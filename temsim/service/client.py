r"""Part IV, Section 5: the client.

Theory
======

A client is anything that speaks the protocol of Section 1: the Qt interface, a
script, a test.  It needs neither PyTorch nor the physics; this module imports
numpy, pyzmq and the data layer.

A DEALER socket may be used by one thread only.  The client therefore owns a
*transport thread* that holds the socket; other threads hand it outgoing messages
through an in-process queue (``inproc`` PUSH/PULL pair, which also wakes the
thread) and wait for the reply with the matching ``id``.  Incoming events are
delivered to ``on_event(name, data, arrays)`` *in the transport thread*; a graphical
client must forward them to its own thread (for Qt: emit a signal).

Calls are synchronous (:meth:`Client.call`) and time out; a timed-out call
does not cancel the command, which may still run on the core.
"""

from __future__ import annotations

import itertools
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import zmq

from ..data.schema import Experiment, ModelState, TrainSettings
from . import protocol

EventHandler = Callable[[str, Dict[str, Any], Dict[str, np.ndarray]], None]


class RemoteError(RuntimeError):
    """The core refused or failed a command; ``trace`` is its traceback."""

    def __init__(self, message: str, trace: str = ""):
        super().__init__(message)
        self.trace = trace


@dataclass
class Reply:
    result: Dict[str, Any]
    arrays: Dict[str, np.ndarray]


class _Pending:
    def __init__(self):
        self.done = threading.Event()
        self.header: Optional[Dict[str, Any]] = None
        self.arrays: Dict[str, np.ndarray] = {}


class Client:
    def __init__(self, endpoint: str, on_event: Optional[EventHandler] = None, timeout: float = 120.0):
        self.endpoint = endpoint
        self.on_event = on_event
        self.timeout = timeout
        self._context = zmq.Context.instance()
        self._ids = itertools.count(1)
        self._pending: Dict[int, _Pending] = {}
        self._lock = threading.Lock()
        self._closed = threading.Event()
        wake = f"inproc://temsim-client-{uuid.uuid4().hex}"
        self._pull = self._context.socket(zmq.PULL)
        self._pull.bind(wake)
        self._push = self._context.socket(zmq.PUSH)
        self._push.connect(wake)
        self._push_lock = threading.Lock()
        self._thread = threading.Thread(target=self._transport, name="temsim-client", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ transport
    def _transport(self) -> None:
        dealer = self._context.socket(zmq.DEALER)
        dealer.setsockopt(zmq.LINGER, 200)
        dealer.connect(self.endpoint)
        poller = zmq.Poller()
        poller.register(dealer, zmq.POLLIN)
        poller.register(self._pull, zmq.POLLIN)
        try:
            while not self._closed.is_set():
                ready = dict(poller.poll(200))
                if self._pull in ready:
                    while True:
                        try:
                            dealer.send_multipart(self._pull.recv_multipart(flags=zmq.NOBLOCK), copy=False)
                        except zmq.Again:
                            break
                if dealer in ready:
                    while True:
                        try:
                            frames = dealer.recv_multipart(flags=zmq.NOBLOCK, copy=False)
                        except zmq.Again:
                            break
                        self._dispatch([bytes(f.buffer) for f in frames])
        finally:
            dealer.close()
            self._pull.close()

    def _dispatch(self, frames) -> None:
        try:
            header, arrays = protocol.unpack(frames)
        except Exception:
            return
        if header.get("kind") == "reply":
            with self._lock:
                pending = self._pending.pop(int(header["id"]), None)
            if pending is not None:
                pending.header, pending.arrays = header, arrays
                pending.done.set()
        elif header.get("kind") == "event" and self.on_event is not None:
            try:
                self.on_event(header["name"], header.get("data", {}), arrays)
            except Exception:
                pass

    # ------------------------------------------------------------------ calls
    def call(self, name: str, args: Optional[Dict[str, Any]] = None,
             arrays: Optional[Dict[str, np.ndarray]] = None, timeout: Optional[float] = None) -> Reply:
        if self._closed.is_set():
            raise RemoteError("the client is closed")
        identifier = next(self._ids)
        pending = _Pending()
        with self._lock:
            self._pending[identifier] = pending
        with self._push_lock:
            self._push.send_multipart(protocol.pack(protocol.command(identifier, name, args), arrays), copy=False)
        if not pending.done.wait(self.timeout if timeout is None else timeout):
            with self._lock:
                self._pending.pop(identifier, None)
            raise TimeoutError(f"no reply to {name!r} (id {identifier})")
        header = pending.header
        if not header.get("ok", False):
            raise RemoteError(header.get("error", "unknown error"), header.get("trace", ""))
        return Reply(header.get("result", {}), pending.arrays)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._thread.join(5.0)
        with self._push_lock:
            self._push.close(0)

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ typed helpers
    def hello(self, extras=None) -> Dict[str, Any]:
        args = {} if extras is None else {"extras": list(extras)}
        result = self.call("hello", args).result
        if result.get("protocol") != protocol.PROTOCOL_VERSION:
            raise RemoteError(f"protocol {result.get('protocol')} != {protocol.PROTOCOL_VERSION}")
        return result

    def info(self) -> Dict[str, Any]:
        return self.call("info").result

    def open_run(self, directory: str, target_path: Optional[str] = None) -> Dict[str, Any]:
        return self.call("open_run", {"directory": directory, "target_path": target_path}).result["info"]

    def open_dataset(self, path: str, route: str = "auto", **options) -> Dict[str, Any]:
        return self.call("open_dataset", {"path": path, "route": route, "options": options}).result["info"]

    def get_experiment(self) -> Experiment:
        return protocol.experiment_from_wire(self.call("get_experiment").result["experiment"])

    def get_state(self) -> ModelState:
        return protocol.state_from_wire(self.call("get_state").result["state"])

    def get_values(self) -> Dict[str, Any]:
        return self.call("get_values").result["values"]

    def set_values(self, **values) -> Dict[str, Any]:
        return self.call("set_values", {"values": values}).result["values"]

    def get_train(self) -> TrainSettings:
        reply = self.call("get_train")
        return protocol.train_from_wire(reply.result["train"], reply.arrays)

    def set_train(self, settings: TrainSettings, with_arrays: bool = True) -> None:
        wire, arrays = protocol.train_to_wire(settings)
        self.call("set_train", {"train": wire}, arrays if with_arrays else None)

    def set_param(self, key: str, enabled: Optional[bool] = None, lr: Optional[float] = None) -> None:
        self.call("set_param", {"key": key, "enabled": enabled, "lr": lr})

    def set_loss(self, **weights: float) -> None:
        self.call("set_loss", {"weights": weights})

    def set_regularisation(self, **weights: float) -> None:
        self.call("set_regularisation", {"weights": weights})

    def set_active_group(self, name: Optional[str]) -> None:
        self.call("set_active_group", {"name": name})

    def start_training(self, steps: Optional[int] = None, refine: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.call("start_training", {"steps": steps, "refine": refine}).result["status"]

    def start_seeding(self, **options) -> Dict[str, Any]:
        return self.call("start_seeding", options).result["status"]

    def pause(self) -> None:
        self.call("pause")

    def resume(self) -> None:
        self.call("resume")

    def stop(self) -> None:
        self.call("stop")

    def frame(self, extras=(), standardised: bool = False) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        reply = self.call("frame", {"extras": list(extras), "standardised": standardised})
        return reply.result["frame"], reply.arrays

    def edit(self, name: str, **args) -> Dict[str, Any]:
        return self.call("edit", {"name": name, "args": args}).result["report"]

    def minima(self, threshold: float = 0.5, distance_px: float = 3.0, layer=None,
               atom_distance: bool = True, verify: Optional[Dict[str, bool]] = None) -> np.ndarray:
        return self.call("minima", {"threshold": threshold, "distance_px": distance_px, "layer": layer,
                                    "atom_distance": atom_distance, "verify": verify}).arrays["points"]

    def insert_minima(self, **args) -> Dict[str, Any]:
        return self.call("insert_minima", args).result["report"]

    def seeding_config(self, patch: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.call("seeding_config", {"patch": patch}).result["config"]

    def save(self, directory: Optional[str] = None) -> str:
        return self.call("save", {"directory": directory}).result["directory"]

    def shutdown(self) -> None:
        self.call("shutdown", timeout=10.0)
