r"""Part V, Section 1: what the interface knows about the model.

Theory
======

The interface owns no model.  Everything it shows is a *frame*: the dictionary and
the arrays of one ``frame`` event (Part IV, Section 1).  This module turns a frame
into a read-only object, :class:`FrameModel`, and derives from it what the plots
need - which atoms are shown, a common histogram of their intensities, the atoms
that lie inside the mask.  None of this needs Qt, so it is tested without a display.

**Visible atoms.**  An atom of layer :math:`l` is shown if its occupancy
:math:`o = \sigma(\vartheta)` is at least the display threshold :math:`o_{\min}` and its
intensity is finite.  The threshold is a property of the *view*; the model is never
changed by it.

**Histogram.**  The intensities :math:`I_i` of all visible atoms of all layers are
binned into :math:`B = 101` bins that span the finite range of all layers together,
so that the curves of different layers can be compared.

**Loss history.**  The core sends one ``light`` event per optimiser step.  The
history keeps at most :data:`HISTORY_LIMIT` values; when the limit is reached every
second point is dropped and the sampling stride doubles, so that a run of millions
of steps stays cheap to draw.

Implementation
==============
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HISTOGRAM_BINS = 101
HISTORY_LIMIT = 20000


@dataclass
class LayerView:
    name: str
    group: str
    element: str
    pos: np.ndarray                    # (n, 2) row, column [px]
    occupancy: np.ndarray              # (n,)
    intensity: np.ndarray              # (n,)
    gradient: Optional[np.ndarray] = None   # (n, 2) d loss / d shift [loss per px]
    potential: Optional[np.ndarray] = None

    def visible(self, min_occupancy: float = 0.0) -> np.ndarray:
        """Boolean mask of the atoms that are drawn."""

        return (self.occupancy >= min_occupancy) & np.isfinite(self.intensity) & np.isfinite(self.pos).all(axis=1)


@dataclass
class FrameModel:
    loss: float
    data: Dict[str, float]             # the weighted terms of the data loss
    prior: Dict[str, float]            # the regularisation terms
    step: int
    mode: str
    active_group: Optional[str]
    revision: int
    sim: np.ndarray
    target: np.ndarray
    diff: np.ndarray
    layers: List[LayerView]
    mask: Optional[np.ndarray] = None
    ctf_phase: Optional[np.ndarray] = None
    ctf_ring: Optional[np.ndarray] = None
    probe: Optional[np.ndarray] = None
    strain: Dict[str, np.ndarray] = field(default_factory=dict)
    seeding: Optional[Dict[str, Any]] = None
    group_names: List[str] = field(default_factory=list)
    seq: Optional[int] = None          # number to acknowledge (None: the frame answered a request)

    @property
    def stem(self) -> bool:
        return self.mode == "stem_haadf"

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.sim.shape)

    # -- construction -----------------------------------------------------------
    @classmethod
    def from_wire(cls, data: Dict[str, Any], arrays: Dict[str, np.ndarray],
                  previous: Optional["FrameModel"] = None) -> "FrameModel":
        """``previous`` supplies the extras (optics, strain, potentials, gradients) that a ``partial`` frame lacks."""

        layers = []
        for i, meta in enumerate(data["layers"]):
            layers.append(LayerView(
                name=meta["name"], group=meta["group"], element=meta.get("element", ""),
                pos=np.asarray(arrays[f"atoms.{i}.pos"], dtype=np.float64).reshape(-1, 2),
                occupancy=np.asarray(arrays[f"atoms.{i}.occ"], dtype=np.float64),
                intensity=np.asarray(arrays[f"atoms.{i}.int"], dtype=np.float64),
                gradient=_optional(arrays, f"grad.{i}", (-1, 2)),
                potential=arrays.get(f"potential.{i}")))
        strain = {key.split(".", 1)[1]: arrays[key] for key in arrays if key.startswith("strain.")}
        mask = arrays.get("mask")
        groups: List[str] = []
        for layer in layers:
            if layer.group not in groups:
                groups.append(layer.group)
        frame = cls(loss=float(data["loss"]), data=_parts(data.get("data")), prior=_parts(data.get("prior")),
                   step=int(data.get("steps", 0)), mode=data.get("mode", "tem"),
                   active_group=data.get("active_group"), revision=int(data.get("revision", 0)),
                   sim=arrays["sim"], target=arrays["target"], diff=arrays["diff"], layers=layers,
                   mask=None if mask is None else np.asarray(mask) > 0,
                   ctf_phase=arrays.get("ctf_phase"), ctf_ring=arrays.get("ctf_ring"), probe=arrays.get("probe"),
                   strain=strain, seeding=data.get("seeding"), group_names=groups, seq=data.get("seq"))
        if data.get("partial") and previous is not None:
            frame._take_extras(previous)
        return frame

    def _take_extras(self, old: "FrameModel") -> None:
        """Keep what a partial frame does not carry; atoms of another count get no stale potentials or arrows."""

        for name in ("ctf_phase", "ctf_ring", "probe"):
            if getattr(self, name) is None:
                setattr(self, name, getattr(old, name))
        if not self.strain:
            self.strain = old.strain
        if len(old.layers) == len(self.layers):
            for new, before in zip(self.layers, old.layers):
                if new.potential is None:
                    new.potential = before.potential
                if new.gradient is None and before.gradient is not None and len(before.gradient) == len(new.pos):
                    new.gradient = before.gradient

    # -- derived views ------------------------------------------------------------
    def active_layers(self) -> List[int]:
        """Layer indices of the active group (all layers if none is active)."""

        if self.active_group is None:
            return list(range(len(self.layers)))
        return [i for i, layer in enumerate(self.layers) if layer.group == self.active_group]

    def atoms(self, min_occupancy: float = 0.0, active_only: bool = True,
              inside_mask: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """``(positions (n, 2), intensities (n,))`` of the shown atoms."""

        indices = self.active_layers() if active_only else range(len(self.layers))
        pos, ints = [], []
        for i in indices:
            layer = self.layers[i]
            keep = layer.visible(min_occupancy)
            pos.append(layer.pos[keep])
            ints.append(layer.intensity[keep])
        if not pos:
            return np.zeros((0, 2)), np.zeros(0)
        pos, ints = np.concatenate(pos), np.concatenate(ints)
        height, width = self.shape
        keep = (pos[:, 0] >= 0) & (pos[:, 0] <= height) & (pos[:, 1] >= 0) & (pos[:, 1] <= width)
        if inside_mask and self.mask is not None and keep.any():
            rows = np.clip(pos[:, 0].astype(int), 0, self.mask.shape[0] - 1)
            cols = np.clip(pos[:, 1].astype(int), 0, self.mask.shape[1] - 1)
            keep &= self.mask[rows, cols]
        return pos[keep], ints[keep]

    def gradients(self, min_occupancy: float = 0.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """``(positions, gradients)`` of the shown atoms of the active group, or ``None``."""

        found_pos, found_grad = [], []
        for i in self.active_layers():
            layer = self.layers[i]
            if layer.gradient is None or len(layer.gradient) != len(layer.pos):
                continue
            keep = layer.visible(min_occupancy) & np.isfinite(layer.gradient).all(axis=1)
            found_pos.append(layer.pos[keep])
            found_grad.append(layer.gradient[keep])
        if not found_pos:
            return None
        return np.concatenate(found_pos), np.concatenate(found_grad)

    def histogram(self, min_occupancy: float = 0.0, bins: int = HISTOGRAM_BINS) -> Tuple[List[np.ndarray], np.ndarray]:
        """Intensity histograms of all layers over common bin edges."""

        values = [layer.intensity[layer.visible(min_occupancy)] for layer in self.layers]
        finite = np.concatenate(values) if values else np.zeros(0)
        if finite.size == 0:
            return [np.zeros(bins) for _ in values], np.linspace(0.0, 1.0, bins + 1)
        low, high = float(finite.min()), float(finite.max())
        if high <= low:
            low, high = low - 0.5, high + 0.5
        edges = np.linspace(low, high, bins + 1)
        return [np.histogram(v, bins=edges)[0].astype(np.float64) for v in values], edges

    def potentials(self) -> List[Optional[np.ndarray]]:
        return [layer.potential for layer in self.layers]


def _parts(value) -> Dict[str, float]:
    return {str(k): float(v) for k, v in value.items()} if isinstance(value, dict) else {}


def _optional(arrays: Dict[str, np.ndarray], key: str, shape) -> Optional[np.ndarray]:
    return None if key not in arrays else np.asarray(arrays[key], dtype=np.float64).reshape(shape)


class LossHistory:
    """Loss against step, bounded in length (see the module docstring)."""

    def __init__(self, limit: int = HISTORY_LIMIT):
        self.limit = int(limit)
        self.reset()

    def reset(self) -> None:
        self.steps: List[int] = []
        self.values: List[float] = []
        self._stride = 1
        self._seen = 0

    def add(self, step: int, loss: float) -> None:
        self._seen += 1
        if self._seen % self._stride:
            return
        self.steps.append(int(step))
        self.values.append(float(loss))
        if len(self.values) > self.limit:
            self.steps, self.values = self.steps[::2], self.values[::2]
            self._stride *= 2

    def __len__(self) -> int:
        return len(self.values)

    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.steps, dtype=np.float64), np.asarray(self.values, dtype=np.float64)
