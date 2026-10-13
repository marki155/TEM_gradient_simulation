r"""Part IV, Section 1: the wire protocol between the core and a client.

Theory
======

The reconstruction core and the user interface are two processes.  They share no
memory and no objects; everything that crosses the boundary is one *message*,
which consists of

* a JSON *header* (a dictionary), and
* any number of binary *arrays* whose dtype and shape are listed in the header
  (``header["arrays"]``), each sent as one raw buffer.

On the wire a message is a list of frames, ``[header, buffer_1, ..., buffer_n]``,
so that images are never converted to text.  The transport (ZeroMQ DEALER/ROUTER,
see :mod:`temsim.service.server`) adds an identity frame in front.

Three kinds of messages exist.

``command``  (client -> core)
    ``{"kind": "command", "id": k, "name": ..., "args": {...}}``.  Exactly one
    ``reply`` with the same ``id`` follows.  Commands are executed by the single
    owner of the model *between* two optimisation steps (Section 2); the order in
    which a client sends them is the order in which they take effect.
``reply``  (core -> client)
    ``{"kind": "reply", "id": k, "ok": true, "result": {...}}`` or, on failure,
    ``{"ok": false, "error": "...", "trace": "..."}``.  A failing command never
    changes the model.
``event``  (core -> every client that has said ``hello``)
    ``{"kind": "event", "name": ..., "data": {...}}``.  Events are ``status``
    (idle / running / paused), ``light`` (step and loss, cheap, every step),
    ``frame`` (images and atoms, throttled), ``log`` (text) and ``error``.

Data classes of :mod:`temsim.data.schema` travel as the JSON of
:mod:`temsim.data.codec`; the bulk arrays of the training settings (masks, group
targets) travel as binary arrays named ``mask.<group>`` and ``target.<group>``.

The module imports numpy and the data layer only.  A client therefore needs
neither PyTorch nor the physics.

.. note::
   The server binds to ``127.0.0.1`` by default and accepts commands from anyone
   who can reach the port; a remote core is reached through an SSH tunnel, never
   by opening the port.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..data import codec
from ..data.schema import Experiment, ModelState, TrainSettings

PROTOCOL_VERSION = 1

#: dtypes that may be sent (little endian, C order)
ARRAY_DTYPES = ("float32", "float64", "int32", "int64", "uint8", "bool")

#: the commands of the core, with one line each (the table in the documentation)
COMMANDS: Dict[str, str] = {
    "hello": "register for events; returns the protocol version",
    "info": "description of the loaded model",
    "open_run": "load a run directory (any layout, old or new)",
    "open_dataset": "load a data folder or init.json (lattice start or empty seeding start)",
    "get_experiment": "specification with the current atoms",
    "get_train": "training settings (incl. masks as arrays)",
    "get_state": "trained numbers",
    "get_values": "the numbers a person edits by hand (optics, group transform), flat keys",
    "set_values": "write some of those numbers",
    "set_train": "replace the training settings",
    "set_param": "switch / learning rate of one trainable key",
    "set_loss": "weights of the data terms",
    "set_regularisation": "weights of the priors",
    "set_active_group": "group whose mask, target and layers count",
    "start_training": "begin optimisation",
    "start_seeding": "begin seeding",
    "pause": "stop after the current step, keep the job",
    "resume": "continue a paused job",
    "stop": "end the job",
    "frame": "images, atoms and optional extras now",
    "edit": "model edit between steps (remove low occupancy, reset ..., see Session)",
    "minima": "candidate sites from the difference image",
    "insert_minima": "insert candidate sites that pass the seeding criteria",
    "seeding_config": "read or patch the seeding configuration",
    "save": "write a run directory",
    "shutdown": "end the core",
}


# ---------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------

def pack(header: Dict[str, Any], arrays: Optional[Dict[str, np.ndarray]] = None) -> List[bytes]:
    """Message -> frames ``[header, buffer, ...]``."""

    arrays = arrays or {}
    meta, buffers = [], []
    for name, array in arrays.items():
        array = np.asarray(array)
        if array.dtype.name not in ARRAY_DTYPES:
            array = array.astype(np.float64 if array.dtype.kind == "f" else np.int64)
        meta.append({"name": name, "dtype": array.dtype.name, "shape": list(array.shape)})
        buffers.append(np.ascontiguousarray(array).tobytes())
    head = dict(header)
    head["arrays"] = meta
    return [json.dumps(head, separators=(",", ":")).encode("utf-8")] + buffers


def unpack(frames: List[bytes]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Frames -> ``(header, arrays)``; the header loses its ``arrays`` entry."""

    header = json.loads(bytes(frames[0]).decode("utf-8"))
    meta = header.pop("arrays", [])
    if len(meta) != len(frames) - 1:
        raise ValueError(f"header announces {len(meta)} arrays, message has {len(frames) - 1} buffers")
    arrays = {}
    for entry, buffer in zip(meta, frames[1:]):
        if entry["dtype"] not in ARRAY_DTYPES:
            raise ValueError(f"dtype {entry['dtype']!r} is not allowed")
        shape = tuple(int(n) for n in entry["shape"])
        array = np.frombuffer(bytes(buffer), dtype=np.dtype(entry["dtype"]))
        if array.size != int(np.prod(shape, dtype=np.int64)):
            raise ValueError(f"array {entry['name']!r}: {array.size} values for shape {shape}")
        arrays[entry["name"]] = array.reshape(shape).copy()
    return header, arrays


def command(identifier: int, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"kind": "command", "id": int(identifier), "name": name, "args": args or {}}


def reply(identifier: int, result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"kind": "reply", "id": int(identifier), "ok": True, "result": result or {}}


def failure(identifier: int, error: str, trace: str = "") -> Dict[str, Any]:
    return {"kind": "reply", "id": int(identifier), "ok": False, "error": error, "trace": trace}


def event(name: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"kind": "event", "name": name, "data": data or {}}


# ---------------------------------------------------------------------------
# data classes on the wire
# ---------------------------------------------------------------------------

def experiment_to_wire(experiment: Experiment) -> Dict[str, Any]:
    return codec.encode(experiment)


def experiment_from_wire(data: Dict[str, Any]) -> Experiment:
    return codec.decode(Experiment, data)


def state_to_wire(state: ModelState) -> Dict[str, Any]:
    return codec.encode(state)


def state_from_wire(data: Dict[str, Any]) -> ModelState:
    return codec.decode(ModelState, data)


def train_to_wire(settings: TrainSettings) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """JSON part and the bulk arrays ``mask.<group>`` / ``target.<group>``."""

    arrays = {f"mask.{k}": np.asarray(v, dtype=np.uint8) for k, v in settings.masks.items()}
    arrays.update({f"target.{k}": np.asarray(v, dtype=np.float32) for k, v in settings.targets.items()})
    return codec.encode(settings), arrays


def train_from_wire(data: Dict[str, Any], arrays: Optional[Dict[str, np.ndarray]] = None) -> TrainSettings:
    settings = codec.decode(TrainSettings, data)
    for name, array in (arrays or {}).items():
        kind, _, group = name.partition(".")
        if kind == "mask":
            settings.masks[group] = np.asarray(array) > 0
        elif kind == "target":
            settings.targets[group] = np.asarray(array, dtype=np.float32)
    return settings
