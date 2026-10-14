r"""Part IV, Section 2 (continued): the workspace, one directory for opening and saving.

Theory
======

A core started with ``--workspace DIR`` keeps all its data in one directory: the measurements
(data folders, Part III Section 2) and the runs that were saved.  A client does not need to know
paths on the machine of the core; it asks for the *entries* of the workspace, picks one and
sends its path relative to the workspace.  "Save as" likewise only sends a *name*; the run is
written to ``<workspace>/<name>``.

An entry is a folder that is either

* a **run** (it holds an experiment file, :data:`temsim.data.store.EXPERIMENT_FILES`), or
* a **dataset** (it holds a target image, Part III Section 2).  ``autosave`` tells whether a
  dataset has an autosave of an earlier session in one of its sub-folders; opening the
  dataset then continues from it.

The ``autosave`` folders themselves and half-written folders (``*.writing``) are not listed.
The scan looks :data:`MAX_DEPTH` levels deep and lists at most :data:`MAX_ENTRIES` folders.

Implementation
==============
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..data import dataset as datasets
from ..data import store
from .assemble import AUTOSAVE_DIR

MAX_DEPTH = 3
MAX_ENTRIES = 500
FORBIDDEN = set('<>:"|?*\0')


def kind(folder: str) -> Optional[str]:
    if any(os.path.exists(os.path.join(folder, n)) for n in store.EXPERIMENT_FILES):
        return "run"
    if datasets.find_target(folder) is not None:
        return "dataset"
    return None


def _has_autosave(folder: str) -> bool:
    try:
        return any(os.path.isdir(os.path.join(folder, n, AUTOSAVE_DIR)) for n in os.listdir(folder))
    except OSError:
        return False


def scan(root: str) -> List[Dict[str, Any]]:
    """The entries below ``root``: ``{"path" (relative, with "/"), "kind", "autosave", "mtime"}``."""

    entries: List[Dict[str, Any]] = []

    def walk(folder: str, depth: int) -> None:
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            return
        for name in names:
            path = os.path.join(folder, name)
            if len(entries) >= MAX_ENTRIES:
                return
            if name.startswith(".") or name == AUTOSAVE_DIR or name.endswith(".writing") or not os.path.isdir(path):
                continue
            found = kind(path)
            if found is not None:
                entries.append({"path": os.path.relpath(path, root).replace(os.sep, "/"), "kind": found,
                                "autosave": found == "dataset" and _has_autosave(path),
                                "mtime": os.path.getmtime(path)})
            if depth < MAX_DEPTH:
                walk(path, depth + 1)

    walk(root, 1)
    return entries


def resolve(root: Optional[str], path: str) -> str:
    """``path`` as the file system sees it: a relative path is taken relative to the workspace."""

    if root and not os.path.isabs(path):
        return os.path.join(root, path)
    return path


def child(root: str, name: str) -> str:
    """``<root>/<name>`` for a plain relative name (sub-folders allowed, nothing that leaves ``root``)."""

    name = name.strip().replace("\\", "/").strip("/")
    parts = name.split("/")
    if not name or any(p in ("", ".", "..") or p.startswith(".") for p in parts) or FORBIDDEN & set(name):
        raise ValueError(f"{name!r} is not a valid name inside the workspace")
    if parts[-1] == AUTOSAVE_DIR or parts[-1].endswith(".writing"):
        raise ValueError(f"{name!r} is reserved")
    return os.path.join(root, *parts)
