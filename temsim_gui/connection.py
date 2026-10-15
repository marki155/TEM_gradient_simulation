r"""Part V, Section 5: the connection to the core.

Theory
======

The interface talks to the core through :class:`temsim.service.client.Client`, which
delivers events in its transport thread and blocks while a command is answered.
Qt wants neither: widgets may only be touched from the GUI thread, and the GUI thread
must not wait for a simulation.  :class:`Connection` bridges the two.

*Events* arrive in the transport thread, are converted (a ``frame`` event becomes a
:class:`~temsim_gui.frame_model.FrameModel`) and re-emitted as Qt signals; Qt queues
a signal that is emitted from another thread, so the slots run in the GUI thread.

A frame is *acknowledged* by the controller once it has been drawn
(:meth:`Connection.acknowledge`); the core sends the next frames only then, so that a slow
laptop or link is never flooded.  A frame that lacks the extras (``partial``) inherits them
from the previous one.

*Commands* are submitted to a single worker thread, in order, and their result is
handed back by a signal.  Because there is one worker, the order in which the user
pressed buttons is the order in which the core sees the commands - the guarantee
of Part IV, Section 1.  :meth:`Connection.submit` takes the callable that performs
the call, a ``done`` callback for the result and an ``failed`` callback for a
:class:`~temsim.service.client.RemoteError`.

Starting a core.  :func:`spawn_core` runs ``python -m temsim.service serve`` as a child
process on a free local port and waits until it answers; the interface then
owns the process and ends it on exit.  Data is opened by a command afterwards, never by
the command line, so that the model has one owner only (Part IV, Section 2).

Implementation
==============
"""

from __future__ import annotations

import queue
import socket
import subprocess
import sys
import threading
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PySide6 import QtCore

from temsim.service.client import Client, RemoteError

from .frame_model import FrameModel


class _Worker(threading.Thread):
    """Runs submitted calls one after the other."""

    def __init__(self, deliver: Callable):
        super().__init__(name="temsim-gui-commands", daemon=True)
        self.jobs: "queue.Queue" = queue.Queue()
        self.deliver = deliver

    def run(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            call, done, failed = job
            try:
                result = call()
            except BaseException as exc:
                self.deliver(failed, exc)
            else:
                self.deliver(done, result)


class Connection(QtCore.QObject):
    """A client of one core, as a Qt object."""

    status = QtCore.Signal(dict)
    light = QtCore.Signal(dict)
    frame = QtCore.Signal(object)
    log = QtCore.Signal(str)
    error = QtCore.Signal(str, str)
    _result = QtCore.Signal(object, object)

    def __init__(self, endpoint: str, extras=(), timeout: float = 600.0, parent=None):
        super().__init__(parent)
        self.endpoint = endpoint
        self.client = Client(endpoint, on_event=self._on_event, timeout=timeout, auto_ack=False)
        self._last_frame: Optional[FrameModel] = None
        self._result.connect(self._deliver)
        self._worker = _Worker(self._result.emit)
        self._worker.start()
        self.hello_info = self.client.hello(extras=list(extras))
        self._extras = tuple(extras)

    # ------------------------------------------------------------------ events
    def _on_event(self, name: str, data: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
        if name == "frame":
            frame = FrameModel.from_wire(data, arrays, self._last_frame)
            self._last_frame = frame
            self.frame.emit(frame)
        elif name == "light":
            self.light.emit(data)
        elif name == "status":
            self.status.emit(data)
        elif name == "log":
            self.log.emit(str(data.get("text", "")))
        elif name == "error":
            self.error.emit(str(data.get("error", "")), str(data.get("trace", "")))

    def remember(self, frame: FrameModel) -> None:
        """Make ``frame`` (a reply to a request) the one that later partial frames inherit extras from."""

        self._last_frame = frame

    def acknowledge(self, frame: FrameModel) -> None:
        """Tell the core that ``frame`` has been drawn; it sends the next one only then."""

        if frame.seq is not None:
            self.client.ack_frame(frame.seq)

    @property
    def extras(self) -> tuple:
        return self._extras

    def set_extras(self, extras) -> None:
        """Tell the core which additional arrays the frame events should carry."""

        extras = tuple(extras)
        if extras != self._extras:
            self._extras = extras
            self.submit(lambda: self.client.hello(extras=list(extras)))

    # ------------------------------------------------------------------ commands
    def submit(self, call: Callable[[], Any], done: Optional[Callable[[Any], None]] = None,
               failed: Optional[Callable[[BaseException], None]] = None) -> None:
        """Run ``call()`` on the command thread; ``done(result)`` / ``failed(exc)`` run in the GUI thread."""

        self._worker.jobs.put((call, done, failed or self._report))

    def _report(self, exc: BaseException) -> None:
        trace = exc.trace if isinstance(exc, RemoteError) else ""
        self.error.emit(f"{type(exc).__name__}: {exc}", trace)

    @QtCore.Slot(object, object)
    def _deliver(self, callback, payload) -> None:
        if callback is not None:
            callback(payload)

    # ------------------------------------------------------------------ shutdown
    def close(self, timeout: float = 60.0) -> None:
        """Let the queued commands finish (at most ``timeout`` seconds), then disconnect."""

        self._worker.jobs.put(None)
        self._worker.join(timeout)
        self.client.close()


class SpawnedCore:
    """A core started by the interface (a child process)."""

    def __init__(self, process: subprocess.Popen, endpoint: str):
        self.process, self.endpoint = process, endpoint

    def stop(self, timeout: float = 20.0) -> None:
        if self.process.poll() is not None:
            return
        try:
            with Client(self.endpoint, timeout=5.0) as client:
                client.shutdown()
            self.process.wait(timeout)
        except Exception:
            self.process.terminate()
            try:
                self.process.wait(5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn_core(device: str = "auto", timeout: float = 120.0,
               extra_args: Optional[List[str]] = None) -> SpawnedCore:
    """Start ``python -m temsim.service serve`` on a free port and wait for it."""

    endpoint = f"tcp://127.0.0.1:{free_port()}"
    command = [sys.executable, "-m", "temsim.service", "serve", "--endpoint", endpoint, "--device", device]
    command += list(extra_args or [])
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=None)
    core = SpawnedCore(process, endpoint)
    deadline = threading.Event()
    timer = threading.Timer(timeout, deadline.set)
    timer.start()
    try:
        while not deadline.is_set():
            if process.poll() is not None:
                raise RuntimeError(f"the core exited with code {process.returncode}")
            try:
                with Client(endpoint, timeout=2.0) as client:
                    client.info()
                return core
            except TimeoutError:
                continue
        core.stop()
        raise TimeoutError("the core did not answer in time")
    finally:
        timer.cancel()
