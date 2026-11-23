r"""Part II, Section 8.4: a collection of good reconstructions.

Theory
======

One run teaches the detector one image.  A model that works on every image needs
many, and not every reconstruction is good enough to learn from.  The *collection*
keeps the ones that are: an **entry** is a snapshot of a reconstruction at the
moment it was judged good -- the two input images, the fitted atoms (positions and
intensities), the valid region and the pixel size.  Later changes of the run do not
alter it.

Entries carry **groups** as tags ("Optimiert", "Handverlesen", ...); an entry may
be in several groups or in none.  A training run takes a chosen set of entries,
each as one sample of Section 8.3 with its own label width
:math:`\sigma_{\rm px} = \sigma / \Delta x` (the label width is a physical size,
the pixel size :math:`\Delta x` may differ between entries) and its own validation
tolerances.

Layout on disk (``~/data/ml_collection``, override with ``TEMSIM_COLLECTION``)::

    index.json            {"groups": [...], "entries": {id: {name, source, created, groups, atoms, shape, pixel_A}}}
    entries/<id>.npz      target, second, valid, positions, intensity, pixel_A

Implementation
==============
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .training import Sample, make_sample, validation_scales

DEFAULT_ROOT = os.path.expanduser(os.environ.get("TEMSIM_COLLECTION") or "~/data/ml_collection")
INDEX, ENTRIES = "index.json", "entries"


class CollectionError(ValueError):
    pass


class Collection:
    """The entries and groups under ``root`` (re-read for every call: the core and the tools share it)."""

    def __init__(self, root: Optional[str] = None):
        self.root = root or DEFAULT_ROOT

    # ------------------------------------------------------------------ index
    def _read(self) -> Dict[str, Any]:
        try:
            with open(os.path.join(self.root, INDEX)) as f:
                index = json.load(f)
        except (OSError, ValueError):
            index = {}
        index.setdefault("groups", [])
        index.setdefault("entries", {})
        return index

    def _write(self, index: Dict[str, Any]) -> None:
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, INDEX)
        with open(path + ".partial", "w") as f:
            json.dump(index, f, indent=1)
        os.replace(path + ".partial", path)

    def _path(self, entry_id: str) -> str:
        return os.path.join(self.root, ENTRIES, entry_id + ".npz")

    # ------------------------------------------------------------------ reading
    def groups(self) -> List[str]:
        index = self._read()
        used = {g for e in index["entries"].values() for g in e.get("groups", [])}
        return sorted(set(index["groups"]) | used, key=str.lower)

    def entries(self, group: Optional[str] = None) -> List[Dict[str, Any]]:
        """The entries (newest first), optionally only those of ``group``: ``{id, name, groups, ...}``."""

        found = [{"id": key, **value} for key, value in self._read()["entries"].items()
                 if group is None or group in value.get("groups", [])]
        return sorted(found, key=lambda e: e.get("created", ""), reverse=True)

    def load(self, entry_id: str) -> Dict[str, Any]:
        """The snapshot of an entry (arrays and ``pixel_A``, ``name``, ``source``, ``id``)."""

        meta = self._read()["entries"].get(entry_id)
        if meta is None:
            raise CollectionError(f"no entry {entry_id!r}")
        with np.load(self._path(entry_id)) as data:
            snapshot = {key: data[key] for key in data.files}
        snapshot["pixel_A"] = float(snapshot["pixel_A"])
        snapshot.update(id=entry_id, name=meta.get("name", ""), source=meta.get("source", ""))
        return snapshot

    # ------------------------------------------------------------------ changing
    def add(self, snapshot: Dict[str, Any], name: str = "", groups: Iterable[str] = ()) -> str:
        """Store a snapshot (``target``, ``second``, ``valid``, ``positions``, ``intensity``, ``pixel_A``)."""

        positions = np.asarray(snapshot["positions"], dtype=np.float64).reshape(-1, 2)
        if not len(positions):
            raise CollectionError("no atoms in the snapshot")
        entry_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        os.makedirs(os.path.join(self.root, ENTRIES), exist_ok=True)
        target = np.asarray(snapshot["target"], dtype=np.float32)
        valid = snapshot.get("valid")
        np.savez_compressed(self._path(entry_id), target=target,
                            second=np.asarray(snapshot["second"], dtype=np.float32),
                            valid=np.ones(target.shape, bool) if valid is None else np.asarray(valid, dtype=bool),
                            positions=positions, intensity=np.asarray(snapshot["intensity"], dtype=np.float64),
                            pixel_A=float(snapshot["pixel_A"]))
        groups = _clean(groups)
        index = self._read()
        index["entries"][entry_id] = {"name": name or entry_id, "source": snapshot.get("source", ""),
                                      "created": time.strftime("%Y-%m-%d %H:%M:%S"), "groups": groups,
                                      "atoms": int(len(positions)), "shape": list(target.shape),
                                      "pixel_A": float(snapshot["pixel_A"])}
        index["groups"] = sorted(set(index["groups"]) | set(groups), key=str.lower)
        self._write(index)
        return entry_id

    def remove(self, entry_id: str) -> None:
        index = self._read()
        index["entries"].pop(entry_id, None)
        self._write(index)
        try:
            os.remove(self._path(entry_id))
        except OSError:
            pass

    def create_group(self, group: str) -> None:
        group = _clean([group])
        if not group:
            raise CollectionError("a group needs a name")
        index = self._read()
        index["groups"] = sorted(set(index["groups"]) | set(group), key=str.lower)
        self._write(index)

    def delete_group(self, group: str) -> None:
        """The group disappears; its entries stay (in their other groups, or in none)."""

        index = self._read()
        index["groups"] = [g for g in index["groups"] if g != group]
        for entry in index["entries"].values():
            entry["groups"] = [g for g in entry.get("groups", []) if g != group]
        self._write(index)

    def tag(self, entry_ids: Iterable[str], group: str, member: bool = True) -> None:
        """Put the entries into ``group`` (``member``) or take them out of it."""

        group = (_clean([group]) or [""])[0]
        if not group:
            raise CollectionError("a group needs a name")
        index = self._read()
        for entry_id in entry_ids:
            entry = index["entries"].get(entry_id)
            if entry is None:
                continue
            groups = set(entry.get("groups", []))
            groups = groups | {group} if member else groups - {group}
            entry["groups"] = sorted(groups, key=str.lower)
        if member:
            index["groups"] = sorted(set(index["groups"]) | {group}, key=str.lower)
        self._write(index)


def _clean(groups: Iterable[str]) -> List[str]:
    return sorted({str(g).strip() for g in groups if str(g).strip()}, key=str.lower)


def snapshot_sample(snapshot: Dict[str, Any], scale: float, sigma_A: float, holdout: float) -> Sample:
    """One entry as a training sample: label width and validation tolerances from its own pixel size."""

    pixel_A = float(snapshot["pixel_A"]) or 1.0
    sample = make_sample(snapshot["target"], snapshot["second"], snapshot["positions"], snapshot["intensity"],
                         scale, max(1.0, float(sigma_A) / pixel_A), snapshot.get("valid"), holdout,
                         name=str(snapshot.get("name") or snapshot.get("source") or ""))
    sample.tolerance_px, sample.min_distance_px = validation_scales(snapshot["positions"], pixel_A)
    return sample
