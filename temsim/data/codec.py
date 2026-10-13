"""Generic conversion between dataclasses and JSON-compatible values.

Every persistent object in :mod:`temsim.data.schema` is a plain dataclass.  Rather
than writing a ``to_dict``/``from_dict`` pair per class, one pair of functions
walks the *type annotations*:

* ``encode`` turns a dataclass tree into dicts, lists, numbers and strings,
* ``decode`` rebuilds the tree from such values, guided by the target class.

Rules
-----
* numpy arrays become nested lists and come back as ``float64`` arrays,
* tuples become lists and come back as tuples where the annotation says so,
* fields marked ``metadata={"external": True}`` hold bulk arrays (masks, target
  images) that live in ``.npz`` files; they are skipped here,
* unknown keys are *ignored* by default so that a file written by a newer
  version still loads; pass ``strict=True`` to turn them into errors.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Dict, Tuple, Union, get_args, get_origin

import numpy as np

_NoneType = type(None)


def encode(obj: Any) -> Any:
    """Dataclass tree -> JSON-compatible value."""

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: encode(getattr(obj, f.name))
                for f in dataclasses.fields(obj)
                if not f.metadata.get("external")}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    return obj


def decode(tp: Any, data: Any, *, strict: bool = False, path: str = "") -> Any:
    """JSON-compatible value -> value of annotation ``tp``."""

    origin = get_origin(tp)
    args = get_args(tp)

    if tp is Any:
        return data

    if origin is Union:
        if data is None and _NoneType in args:
            return None
        options = [a for a in args if a is not _NoneType]
        if len(options) == 1:
            return decode(options[0], data, strict=strict, path=path)
        return data

    if data is None:
        return None

    if dataclasses.is_dataclass(tp):
        return _decode_dataclass(tp, data, strict=strict, path=path)

    if tp is np.ndarray:
        return np.asarray(data, dtype=np.float64)

    if origin in (list, typing.List):
        item = args[0] if args else Any
        return [decode(item, v, strict=strict, path=f"{path}[{i}]") for i, v in enumerate(data)]

    if origin in (tuple, typing.Tuple):
        if not args:
            return tuple(data)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(decode(args[0], v, strict=strict, path=path) for v in data)
        return tuple(decode(a, v, strict=strict, path=f"{path}[{i}]")
                     for i, (a, v) in enumerate(zip(args, data)))

    if origin in (dict, typing.Dict):
        value_tp = args[1] if len(args) == 2 else Any
        return {k: decode(value_tp, v, strict=strict, path=f"{path}.{k}") for k, v in data.items()}

    if tp is float:
        return float(data)
    if tp is int:
        return int(data)
    if tp is bool:
        return bool(data)
    if tp is str:
        return str(data)
    return data


def _decode_dataclass(cls, data, *, strict, path):
    if not isinstance(data, dict):
        raise TypeError(f"{path or cls.__name__}: expected an object for {cls.__name__}, "
                        f"got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    known = {f.name: f for f in dataclasses.fields(cls) if not f.metadata.get("external")}
    unknown = sorted(set(data) - set(known))
    if unknown and strict:
        raise KeyError(f"{path or cls.__name__}: unknown fields {unknown}")
    kwargs = {name: decode(hints[name], data[name], strict=strict, path=f"{path}.{name}")
              for name in known if name in data}
    return cls(**kwargs)


def round_trip(obj: Any) -> Any:
    """``decode(type(obj), encode(obj))`` - handy in tests."""

    return decode(type(obj), encode(obj), strict=True)
