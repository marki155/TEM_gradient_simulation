r"""Part II, Section 6.1: the atom list of the layer that is being seeded.

Seeding changes the *number* of atoms; the parameter tensors of Section 1
(:class:`~temsim.learning.parameters.LayerParams`) therefore grow and shrink.
:class:`AtomStore` is the thin layer between the seeding algorithm and those
tensors.  It keeps NumPy mirrors of the effective positions and weights (for
neighbour search and for the local image of Part I, Section 10) and makes every
edit reversible: ``apply`` returns an undo token that holds the *old tensors*
(editing creates new ones), so a rejected trial costs one pointer swap.

Identity geometry
-----------------
The seeding works in the frame of the image, where a pixel position is the
effective position.  This is only true if the group of the layer is not
transformed (:math:`\mathbf t = 0, \phi = 0, \mathbf s = 1`, strain zero) in
Eq. (1.2); otherwise the store refuses to start (:class:`SeedingError`).  A new
atom is entered at its image position as initial position
:math:`\mathbf r_0` with zero shift.

Weights
-------
A new atom has occupancy logit ``init_occupancy`` of the layer (default 6,
:math:`\sigma \approx 0.9975`: weight close to one, but with a non-vanishing
gradient) and intensity :math:`I = Z\,w_{\rm rel}` with the reference
:math:`I_0 = Z`, so that its weight is :math:`w_{\rm rel}\,\sigma(o)`,
Eq. (4.3).  The relative weight :math:`w_{\rm rel}\in(0, 1.5]` lets the seeding
place dimmer atoms (thin spots, foreign atoms).  In a mixed layer the intensity
selects the element, so a new atom has the atomic number of the layer element and
:math:`w_{\rm rel} = 1`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ...physics import structure
from ..parameters import ParameterSet

OCCUPIED_THRESHOLD = 0.5
PARK_POSITION = -1.0e4


class SeedingError(RuntimeError):
    pass


@dataclass
class UndoToken:
    tensors: Tuple[torch.Tensor, ...]
    positions: np.ndarray
    weights: np.ndarray
    intensity: np.ndarray


_FIELDS = ("base", "reference", "shifts", "intensity", "logit")


class AtomStore:
    def __init__(self, params: ParameterSet, layer: Union[int, str] = 0, sync: bool = True):
        self.params = params
        self.index = params.layer_index(layer) if isinstance(layer, str) else int(layer)
        self.spec = params.experiment.layers[self.index]
        self.mixed = params.mixed(self.index)
        self.atomic_number = float(structure.atomic_number_of(self.spec.element))
        self.new_logit = float(self.spec.init_occupancy)
        self.new_weight = float(torch.sigmoid(torch.tensor(self.new_logit)))
        self._nodes = ([structure.atomic_number_of(e) for e in structure.sorted_elements(self.spec.element, self.spec.elements)]
                       if self.mixed else None)
        self._elements = (structure.sorted_elements(self.spec.element, self.spec.elements) if self.mixed else None)
        self.check_identity()
        if sync:
            self.sync()
        else:
            self._read_mirror()

    # ------------------------------------------------------------------ access
    @property
    def layer(self):
        return self.params.layers[self.index]

    @property
    def supports_weights(self) -> bool:
        return not self.mixed

    @property
    def n(self) -> int:
        return len(self.positions)

    @property
    def trainable(self) -> bool:
        lp = self.layer
        return lp.shifts.requires_grad or lp.intensity.requires_grad or lp.logit.requires_grad

    def check_identity(self, atol: float = 1e-3) -> None:
        lp = self.layer
        if len(lp) == 0:
            return
        with torch.no_grad():
            error = (self.params.positions(self.index) - (lp.base + lp.shifts)).abs().max()
        if float(error) > atol:
            raise SeedingError(f"layer '{self.spec.name}' lies in a transformed group "
                               f"(deviation {float(error):.3g} px); seeding needs identity geometry")

    # ------------------------------------------------------------------ mirror
    def _read_mirror(self) -> None:
        lp = self.layer
        with torch.no_grad():
            self.positions = (lp.base + lp.shifts).cpu().numpy().astype(np.float64).reshape(-1, 2)
            self.weights = self.params.weights(self.index).cpu().numpy().astype(np.float64)
            self.intensity = lp.intensity.cpu().numpy().astype(np.float64)

    def sync(self) -> None:
        """Read the state from the tensors; atoms with occupancy below one half are dropped."""

        self.check_identity()
        self._read_mirror()
        with torch.no_grad():
            occupied = (torch.sigmoid(self.layer.logit) > OCCUPIED_THRESHOLD).cpu().numpy()
        parked = (self.positions < PARK_POSITION / 2).any(axis=1)
        keep = occupied & ~parked
        if not keep.all():
            self.params.remove_atoms(self.index, keep)
            self._read_mirror()

    def remove_below(self, occupancy: float) -> int:
        """Delete atoms with ``sigmoid(logit) < occupancy``; returns how many."""

        with torch.no_grad():
            occupied = torch.sigmoid(self.layer.logit).cpu().numpy()
        keep = occupied >= occupancy
        removed = int((~keep).sum())
        if removed:
            self.params.remove_atoms(self.index, keep)
        self._read_mirror()
        return removed

    # ------------------------------------------------------------------ editing
    def apply(self, remove: Sequence[int], add: np.ndarray, options: Optional[np.ndarray] = None) -> UndoToken:
        """Remove atoms ``remove`` (indices) and append ``add`` ``(m, 2)`` with relative weights ``options``."""

        lp = self.layer
        token = UndoToken(tuple(getattr(lp, f) for f in _FIELDS), self.positions, self.weights, self.intensity)
        add = np.asarray(add, dtype=np.float64).reshape(-1, 2)
        m = len(add)
        options = self._options(options, m)
        keep = np.ones(self.n, dtype=bool)
        keep[np.asarray(remove, dtype=np.int64)] = False
        if not keep.all():
            self.params.remove_atoms(self.index, keep)
        if m:
            self.params.append_atoms(self.index, add, intensity=self.atomic_number * options,
                                     logit=self.new_logit, reference=self.atomic_number)
        self.positions = np.concatenate([self.positions[keep], add])
        self.weights = np.concatenate([self.weights[keep], options * self.new_weight])
        self.intensity = np.concatenate([self.intensity[keep], self.atomic_number * options])
        return token

    def undo(self, token: UndoToken) -> None:
        lp = self.layer
        for name, tensor in zip(_FIELDS, token.tensors):
            setattr(lp, name, tensor)
            if tensor.grad is not None:
                tensor.grad = None
        self.positions, self.weights, self.intensity = token.positions, token.weights, token.intensity

    def _options(self, options, m) -> np.ndarray:
        if options is None or self.mixed:
            return np.ones(m)
        return np.broadcast_to(np.asarray(options, dtype=np.float64), (m,)).copy()

    # ------------------------------------------------------------------ density change for Part I
    def changes(self, remove: Sequence[int], add: np.ndarray,
                options: Optional[np.ndarray] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Signed density change per element, the input of :meth:`LocalImage.propose`."""

        remove = np.asarray(remove, dtype=np.int64)
        add = np.asarray(add, dtype=np.float64).reshape(-1, 2)
        options = self._options(options, len(add))
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        def put(element, points, weights):
            points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
            weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            if element in out:
                points, weights = np.concatenate([out[element][0], points]), np.concatenate([out[element][1], weights])
            out[element] = (points, weights)

        def shares(z, w):
            parts = structure.mixture_shares(torch.as_tensor(z, dtype=torch.float32), self._nodes)
            weight = torch.as_tensor(w, dtype=torch.float32)
            return {e: (s * weight).numpy() for e, s in zip(self._elements, parts)}

        if self.mixed:
            if len(remove):
                for e, w in shares(self.intensity[remove], self.weights[remove]).items():
                    put(e, self.positions[remove], -w)
            if len(add):
                for e, w in shares(np.full(len(add), self.atomic_number), np.full(len(add), self.new_weight)).items():
                    put(e, add, w)
        else:
            if len(remove):
                put(self.spec.element, self.positions[remove], -self.weights[remove])
            if len(add):
                put(self.spec.element, add, options * self.new_weight)
        return out
