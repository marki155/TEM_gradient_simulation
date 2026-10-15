r"""Part IV, Section 3 (continued): turning captured steps into messages without stopping the optimiser.

Theory
======

A frame of a large image is 15 to 70 MB.  Copying it from the GPU, masking and
standardising it, writing the bytes of the message and, above all, *transmitting* it
takes longer than an optimiser step.  None of this has to happen in the thread that
optimises, which only has to *hand over* the images of the step it has just computed
(:class:`~temsim.service.session.Capture`: references to tensors nobody changes any
more, plus copies of the few atom tensors).

:class:`FrameSender` is that second thread.  It holds at most one captured step
(*latest wins*: a newer capture replaces an older one that was not yet processed) and
at most one finished message.  The server loop picks up the finished message and
hands it to the socket, because a ZeroMQ socket may be used by one thread only.

*Size.*  The images are sent as float16 (three digits are plenty for a picture) and the
standardised target and the mask, which do not change while one trains, only when their checksum differs from
the one the client already has: they travel in their own ``target`` event (:func:`package`).  A
client sees this only as less traffic (:class:`~temsim.service.client.Client` restores both).

*Flow control.*  A sender that is faster than the link or the reader would pile up
messages in memory.  Each message carries a sequence number ``seq``; a client that has
drawn it answers with ``ack_frame``; the server sends to a client only while fewer than
:data:`MAX_IN_FLIGHT` messages are unanswered (:class:`Flow`).  A client that never
answers (a script) is not stalled for ever: after :data:`ACK_TIMEOUT` seconds the count
starts again.

Implementation
==============
"""

from __future__ import annotations

import threading
import time
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import protocol

MAX_IN_FLIGHT = 2
ACK_TIMEOUT = 10.0


#: images that travel as float16
HALF = ("sim", "diff", "x", "y")
#: arrays that stay the same while one trains and are sent only when they change
STATIC = ("target", "mask")


@dataclass
class Packet:
    """A frame event as frames, and the target event (if the frame has one) that goes with it."""

    frames: List[bytes]
    target_frames: Optional[List[bytes]] = None
    key: Optional[int] = None


def package(data: Dict[str, Any], arrays: Dict[str, Any]) -> Packet:
    arrays = dict(arrays)
    static = {name: arrays.pop(name) for name in STATIC if name in arrays}
    for name in HALF:
        if name in arrays:
            arrays[name] = np.asarray(arrays[name]).astype(np.float16)
    packet = Packet([])
    if static:
        if "target" in static:
            static["target"] = np.ascontiguousarray(static["target"], dtype=np.float16)
        key = 0
        for name in sorted(static):
            static[name] = np.ascontiguousarray(static[name])
            key = zlib.crc32(static[name].view(np.uint8), key)
        packet.key = key
        data = {**data, "target_key": key, "static": sorted(static)}
        packet.target_frames = protocol.pack(protocol.event("target", {"key": key}), static)
    packet.frames = protocol.pack(protocol.event("frame", data), arrays)
    return packet


class Flow:
    """Unanswered frames per client."""

    def __init__(self, limit: int = MAX_IN_FLIGHT, timeout: float = ACK_TIMEOUT, now=time.monotonic):
        self.limit, self.timeout, self.now = limit, timeout, now
        self._open: Dict[bytes, List[Any]] = {}          # identity -> [count, time of last send]
        #: clients that do not say they acknowledge (older versions): never held back
        self.unlimited: set = set()

    def can_send(self, identity: bytes) -> bool:
        entry = self._open.get(identity)
        if identity in self.unlimited or entry is None or entry[0] < self.limit:
            return True
        if self.now() - entry[1] > self.timeout:
            entry[0] = 0
            return True
        return False

    def sent(self, identity: bytes) -> None:
        entry = self._open.setdefault(identity, [0, 0.0])
        entry[0] += 1
        entry[1] = self.now()

    def acknowledged(self, identity: bytes) -> None:
        entry = self._open.get(identity)
        if entry is not None and entry[0] > 0:
            entry[0] -= 1


class FrameSender:
    """Background thread: ``compose(capture) -> (data, arrays)``, then :func:`package`."""

    def __init__(self, compose: Callable[[Any], Tuple[Dict[str, Any], Dict[str, Any]]]):
        self.compose = compose
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pending: Optional[Any] = None
        self._ready: Optional[Packet] = None
        self._generation = 0
        self._seq = 0
        self._stop = False
        self.error: Optional[BaseException] = None
        #: seconds the last message took to prepare
        self.cost = 0.0
        self._thread = threading.Thread(target=self._run, name="temsim-frames", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ called by the server thread
    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def submit(self, capture: Any) -> None:
        """Hand over a capture; an older one that has not been processed is dropped."""

        with self._lock:
            self._pending = capture
        self._wake.set()

    def take(self) -> Optional[Packet]:
        """The finished message or ``None``; each message is returned once."""

        with self._lock:
            ready, self._ready = self._ready, None
        return ready

    def cancel(self) -> None:
        """Drop everything not yet sent (the model was changed; an older picture would be wrong)."""

        with self._lock:
            self._generation += 1
            self._pending, self._ready = None, None

    def close(self) -> None:
        self._stop = True
        self._wake.set()
        self._thread.join(5.0)

    # ------------------------------------------------------------------ the thread
    def _run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop:
                return
            with self._lock:
                capture, generation = self._pending, self._generation
                self._pending = None
            if capture is None:
                continue
            start = time.monotonic()
            try:
                data, arrays = self.compose(capture)
                data["seq"] = self.next_seq()
                packet = package(data, arrays)
            except BaseException as exc:                      # reported by the server, never fatal
                self.error = exc
                continue
            self.cost = time.monotonic() - start
            with self._lock:
                if generation == self._generation:
                    self._ready = packet
