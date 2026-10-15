r"""Part IV, Section 4: the server.

Theory
======

The server is a single loop that alternates between *listening* and *working*:

.. code-block:: text

    repeat
        handle every waiting command           (at most ``MAX_BATCH`` per round)
        if a job is running:  advance it by one unit; publish light / frame / autosave
        else:                 wait for the next command

so that, as stated in Section 2, the model is only ever touched by this one
thread and commands take effect between two units of work.  Transport is a ZeroMQ
ROUTER socket: every client is a DEALER whose identity the router sees, which
makes replies (to one client) and events (to every client that has said
``hello``) the same kind of send.

Failure is contained.  A command that raises is answered with ``ok = false`` and
a traceback; a job that raises is ended and reported as an ``error`` event;
neither stops the server.  Only ``shutdown`` does.

Implementation
==============
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import zmq

from . import assemble, protocol
from .jobs import Clock, Job, SeedingJob, TrainingJob
from .sender import Flow, FrameSender, Packet, package
from .session import Session, SessionError

MAX_BATCH = 50
IDLE_POLL_MS = 50

#: commands that change what a save would contain
CHANGES_MODEL = frozenset({"set_values", "set_train", "set_param", "set_loss", "set_regularisation",
                           "set_active_group", "edit", "insert_minima", "seeding_config"})
#: commands after which the model is saved at once (the user has done something that is expensive to redo)
#: commands after which a picture that is waiting to be sent would be wrong
CANCELS_FRAMES = frozenset({"set_values", "set_train", "set_active_group", "edit", "insert_minima"})
SAVES_AT_ONCE = frozenset({"pause", "stop", "edit", "insert_minima"})
LIGHT_INTERVAL = 0.05
#: while training, every this many steps the images of that step are sent (0: only the paced frames)
FRAME_STEPS = 10

Result = Tuple[Dict[str, Any], Dict[str, np.ndarray]]


class Server:
    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555", device: str = "auto",
                 clock: Optional[Clock] = None, frame_steps: int = FRAME_STEPS,
                 workspace: Optional[str] = None):
        self.requested_endpoint = endpoint
        self.endpoint = endpoint
        self.session = Session(device)
        self.session.set_workspace(workspace)
        self.clock = clock or Clock()
        self.frame_steps = int(frame_steps)
        self.flow = Flow()
        self.target_keys: Dict[bytes, int] = {}          # the target each client holds
        self.sender: Optional[FrameSender] = None
        self._since_capture = 0
        self.job: Optional[Job] = None
        self.paused = False
        self.subscribers: List[bytes] = []
        self.frame_extras: Tuple[str, ...] = ()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._socket: Optional[zmq.Socket] = None
        self._error: Optional[BaseException] = None
        self._last_light = 0.0
        self.dirty = False

    # ------------------------------------------------------------------ lifecycle
    def start(self, timeout: float = 30.0) -> str:
        """Run in a background thread; returns the bound endpoint."""

        self._thread = threading.Thread(target=self.serve, name="temsim-server", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("the server did not start")
        if self._error is not None:
            raise self._error
        return self.endpoint

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def serve(self) -> None:
        context = zmq.Context.instance()
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 200)
        try:
            socket.bind(self.requested_endpoint)
            self.endpoint = socket.getsockopt(zmq.LAST_ENDPOINT).decode()
        except BaseException as exc:
            self._error = exc
            self._ready.set()
            socket.close()
            return
        self._socket = socket
        self.sender = FrameSender(Session.compose)
        self._ready.set()
        try:
            self._loop()
        finally:
            self._autosave("shutdown")
            self.sender.close()
            socket.close()
            self._socket = None

    # ------------------------------------------------------------------ loop
    @property
    def running(self) -> bool:
        return self.job is not None and not self.paused and not self.job.done

    def _loop(self) -> None:
        socket = self._socket
        while not self._stop.is_set():
            timeout = 0 if self.running else IDLE_POLL_MS
            handled = 0
            while handled < MAX_BATCH and socket.poll(timeout):
                frames = socket.recv_multipart(copy=False)
                self._handle([bytes(f.buffer) for f in frames])
                handled += 1
                timeout = 0
                if self._stop.is_set():
                    return
            self._pump_frames()
            if self.running:
                self._advance()
            elif self.dirty and self.clock.save_due():
                self._autosave("changes")

    def _fast_frames(self) -> bool:
        """Training sends the images of its own steps; the paced frame is then only for the extras."""

        return self.frame_steps > 0 and isinstance(self.job, TrainingJob)

    def _wants_capture(self) -> bool:
        if not self._fast_frames() or not self.session.is_open:
            return False
        self._since_capture += 1
        return self._since_capture >= self.frame_steps and self._can_send()

    def _can_send(self) -> bool:
        return any(self.flow.can_send(identity) for identity in self.subscribers)

    def _advance(self) -> None:
        job = self.job
        capture = self._wants_capture()
        try:
            light = job.advance(capture)
        except BaseException as exc:
            self.job, self.paused = None, False
            self._emit("error", {"error": str(exc), "trace": traceback.format_exc(), "where": "job"})
            self._emit_status()
            self._autosave("error")
            return
        now = time.monotonic()
        if job.done or now - self._last_light >= LIGHT_INTERVAL:
            self._last_light = now
            self._emit("light", light)
        if capture and job.captured is not None:
            self._since_capture = 0
            self.sender.submit(job.captured)
            job.captured = None
        paced = self.clock.frame_due() and (bool(self.frame_extras) or not self._fast_frames())
        if job.done or self.clock.urgent or paced:
            self._emit_frame()
        self.dirty = True
        if self.clock.save_due():
            self._autosave("interval")
        if job.done:
            self.job = None
            self._autosave("job finished")
            self._emit_status()

    def _emit_frame(self) -> None:
        """A complete frame (with the extras), computed here: it needs a simulation of its own."""

        if not self.subscribers or not self.session.is_open or not self._can_send():
            return
        start = time.monotonic()
        self.sender.cancel()
        try:
            data, arrays = self.session.frame(self.frame_extras)
        except BaseException as exc:
            self._emit("error", {"error": str(exc), "trace": traceback.format_exc(), "where": "frame"})
            return
        data["seq"] = self.sender.next_seq()
        self._send_frame(package(data, arrays))
        self.clock.frame_done(time.monotonic() - start)

    def _pump_frames(self) -> None:
        """Send the message the frame thread has finished, to the clients that can take it."""

        if self.sender is None:
            return
        if self.sender.error is not None:
            error, self.sender.error = self.sender.error, None
            self._emit("error", {"error": f"frame: {error}", "where": "frame",
                                 "trace": "".join(traceback.format_exception(error))})
        if self._can_send():
            packet = self.sender.take()
            if packet is not None:
                self._send_frame(packet)

    def _send_frame(self, packet: Packet) -> None:
        for identity in list(self.subscribers):
            if not self.flow.can_send(identity):
                continue
            if packet.target_frames is not None and self.target_keys.get(identity) != packet.key:
                self._socket.send_multipart([identity] + packet.target_frames, copy=False)
                self.target_keys[identity] = packet.key
            self._socket.send_multipart([identity] + packet.frames, copy=False)
            self.flow.sent(identity)

    def _autosave(self, reason: str = "") -> None:
        """Write ``<out_dir>/autosave``; a failure is reported, never raised."""

        self.clock.save_done()
        session = self.session
        if not session.is_open or not session.assembly.out_dir:
            return
        try:
            path = session.save(os.path.join(session.assembly.out_dir, assemble.AUTOSAVE_DIR))
            self.dirty = False
            self._emit("log", {"text": f"saved {path}" + (f" ({reason})" if reason else "")})
        except BaseException as exc:
            self._emit("error", {"error": f"autosave failed: {exc}", "trace": traceback.format_exc(),
                                 "where": "autosave"})

    # ------------------------------------------------------------------ sending
    def _send(self, identity: bytes, header: Dict[str, Any], arrays=None) -> None:
        self._socket.send_multipart([identity] + protocol.pack(header, arrays), copy=False)

    def _emit(self, name: str, data: Optional[Dict[str, Any]] = None, arrays=None) -> None:
        for identity in list(self.subscribers):
            self._send(identity, protocol.event(name, data), arrays)

    def _emit_status(self) -> None:
        self._emit("status", self.status())

    def status(self) -> Dict[str, Any]:
        state = "idle" if self.job is None else ("paused" if self.paused else "running")
        return {"state": state, "job": None if self.job is None else self.job.status(),
                "open": self.session.is_open}

    # ------------------------------------------------------------------ commands
    def _handle(self, frames: List[bytes]) -> None:
        identity, body = frames[0], frames[1:]
        identifier = -1
        try:
            header, arrays = protocol.unpack(body)
            identifier = int(header.get("id", -1))
            if header.get("kind") != "command":
                raise SessionError(f"expected a command, got {header.get('kind')!r}")
            name = header.get("name")
            if name not in protocol.COMMANDS:
                raise SessionError(f"unknown command {name!r}; known: {sorted(protocol.COMMANDS)}")
            method: Callable = getattr(self, "cmd_" + name)
            result, out = method(identity, header.get("args") or {}, arrays)
            if name not in protocol.NO_REPLY:
                self._send(identity, protocol.reply(identifier, result), out)
            if name in CHANGES_MODEL:
                self.dirty = True
            if name in CANCELS_FRAMES or name.startswith("open_"):
                self.sender.cancel()
            if name in SAVES_AT_ONCE:
                self._autosave(name)
        except BaseException as exc:
            self._send(identity, protocol.failure(identifier, f"{type(exc).__name__}: {exc}",
                                                  traceback.format_exc()))

    def _end_job(self) -> None:
        self.job, self.paused = None, False

    def cmd_hello(self, identity, args, arrays) -> Result:
        if identity not in self.subscribers:
            self.subscribers.append(identity)
        (self.flow.unlimited.discard if args.get("acks") else self.flow.unlimited.add)(identity)
        if "extras" in args:
            self.frame_extras = tuple(args["extras"] or ())
        return {"protocol": protocol.PROTOCOL_VERSION, "commands": protocol.COMMANDS,
                "info": self.session.info(), "status": self.status()}, {}

    def cmd_ack_frame(self, identity, args, arrays) -> Result:
        self.flow.acknowledged(identity)
        return {}, {}

    def cmd_info(self, identity, args, arrays) -> Result:
        return {"info": self.session.info(), "status": self.status()}, {}

    def cmd_workspace(self, identity, args, arrays) -> Result:
        return self.session.workspace_entries(), {}

    def cmd_open_run(self, identity, args, arrays) -> Result:
        self._autosave("before opening")
        self._end_job()
        info = self.session.open_run(args["directory"], args.get("target_path"))
        self.clock.frame_now()
        self._emit_status()
        return {"info": info}, {}

    def cmd_open_dataset(self, identity, args, arrays) -> Result:
        self._autosave("before opening")
        self._end_job()
        info = self.session.open_dataset(args["path"], args.get("route", "auto"), **(args.get("options") or {}))
        self.clock.frame_now()
        self._emit_status()
        return {"info": info}, {}

    def cmd_get_experiment(self, identity, args, arrays) -> Result:
        return {"experiment": protocol.experiment_to_wire(self.session.experiment())}, {}

    def cmd_get_train(self, identity, args, arrays) -> Result:
        wire, out = protocol.train_to_wire(self.session.require_model().settings)
        return {"train": wire}, out

    def cmd_get_state(self, identity, args, arrays) -> Result:
        return {"state": protocol.state_to_wire(self.session.require_model().state())}, {}

    def cmd_get_values(self, identity, args, arrays) -> Result:
        return {"values": self.session.values()}, {}

    def cmd_set_values(self, identity, args, arrays) -> Result:
        values = self.session.set_values(args["values"])
        self.clock.frame_now()
        return {"values": values}, {}

    def cmd_set_train(self, identity, args, arrays) -> Result:
        self.session.set_train(protocol.train_from_wire(args["train"], arrays))
        return {}, {}

    def cmd_set_param(self, identity, args, arrays) -> Result:
        self.session.set_param(args["key"], args.get("enabled"), args.get("lr"))
        return {}, {}

    def cmd_set_loss(self, identity, args, arrays) -> Result:
        self.session.set_loss(**args.get("weights", {}))
        return {}, {}

    def cmd_set_regularisation(self, identity, args, arrays) -> Result:
        self.session.set_regularisation(**args.get("weights", {}))
        return {}, {}

    def cmd_set_active_group(self, identity, args, arrays) -> Result:
        self.session.set_active_group(args.get("name"))
        return {}, {}

    def _start(self, job: Job) -> Result:
        self.job, self.paused = job, False
        self.clock.frame_now()
        self._emit_status()
        return {"status": self.status()}, {}

    def cmd_start_training(self, identity, args, arrays) -> Result:
        self.session.require_model()
        if args.get("refine"):
            self.session.refine(**args["refine"])
        return self._start(TrainingJob(self.session, args.get("steps")))

    def cmd_start_seeding(self, identity, args, arrays) -> Result:
        self.session.require_model()
        if args.get("config"):
            self.session.seeding_config(args["config"])
        keys = ("layer", "trials", "chunk", "tune_every", "tune_steps")
        return self._start(SeedingJob(self.session, **{k: args[k] for k in keys if k in args}))

    def cmd_pause(self, identity, args, arrays) -> Result:
        if self.job is not None:
            self.paused = True
        self._emit_status()
        return {"status": self.status()}, {}

    def cmd_resume(self, identity, args, arrays) -> Result:
        self.paused = False
        self._emit_status()
        return {"status": self.status()}, {}

    def cmd_stop(self, identity, args, arrays) -> Result:
        self._end_job()
        self._emit_status()
        return {"status": self.status()}, {}

    def cmd_frame(self, identity, args, arrays) -> Result:
        data, out = self.session.frame(args.get("extras", self.frame_extras), bool(args.get("standardised")))
        return {"frame": data}, out

    def cmd_edit(self, identity, args, arrays) -> Result:
        report = self.session.edit(args["name"], **(args.get("args") or {}))
        self.clock.frame_now()
        return {"report": report}, {}

    def cmd_minima(self, identity, args, arrays) -> Result:
        points = self.session.minima(**args)
        return {"count": int(len(points))}, {"points": points}

    def cmd_insert_minima(self, identity, args, arrays) -> Result:
        report = self.session.insert_minima(**args)
        self.clock.frame_now()
        return {"report": report}, {}

    def cmd_seeding_config(self, identity, args, arrays) -> Result:
        return {"config": self.session.seeding_config(args.get("patch"))}, {}

    def cmd_save(self, identity, args, arrays) -> Result:
        return {"directory": self.session.save(args.get("directory"), args.get("name"))}, {}

    def cmd_shutdown(self, identity, args, arrays) -> Result:
        self._stop.set()
        return {}, {}
