r"""Part V, Section 3: marks drawn over the images.

Theory
======

Three kinds of marks lie over the images: the atoms, the gradient arrows and the
candidate sites.  What is drawn depends on the *zoom*, so the numbers are computed
here, without Qt, and the canvas only draws them.

**Marker size.**  A marker of fixed size becomes dust when one zooms in and a
colour smear when one zooms out.  The area is therefore

.. math::  s = \mathrm{clip}\!\left(\frac{1200}{\Delta},\; 0.5,\; 60\right)\ \mathrm{pt^2},
   \qquad \Delta = \max(\text{width},\,\text{height})\ \text{of the visible window [px]}.   \tag{3.1}

**Gradient arrows.**  For an atom with position :math:`\mathbf r_i` and gradient
:math:`\mathbf g_i = \partial L/\partial \mathbf r_i`, the arrow points along
:math:`-\mathbf g_i` (the direction of descent; optionally :math:`+\mathbf g_i`) and has the length

.. math::  \ell_i = \ell_{\max}\,\frac{|\mathbf g_i|}{g_{\rm ref}},   \tag{3.2}

where :math:`g_{\rm ref}` is the largest :math:`|\mathbf g|` of all atoms (lengths stay
comparable while zooming) or, on request, of the atoms in the visible window.  Only
arrows inside the window are drawn, and at most :data:`MAX_ARROWS` of them (those
with the largest :math:`|\mathbf g|`).

Implementation
==============
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

MAX_ARROWS = 2500


def scatter_size(span: Optional[float]) -> float:
    """Eq. (3.1)."""

    if span is None or not span > 0:
        return 0.5
    return float(np.clip(1200.0 / span, 0.5, 60.0))


def window(xlim, ylim) -> Tuple[float, float, float, float]:
    """``(x0, x1, y0, y1)`` of a possibly inverted view."""

    x0, x1 = sorted(xlim)
    y0, y1 = sorted(ylim)
    return x0, x1, y0, y1


def inside(points: np.ndarray, xlim, ylim) -> np.ndarray:
    """Boolean mask of the ``(row, column)`` points inside the view."""

    x0, x1, y0, y1 = window(xlim, ylim)
    points = np.asarray(points).reshape(-1, 2)
    return (points[:, 1] >= x0) & (points[:, 1] <= x1) & (points[:, 0] >= y0) & (points[:, 0] <= y1)


@dataclass
class Arrows:
    columns: np.ndarray
    rows: np.ndarray
    d_column: np.ndarray
    d_row: np.ndarray
    reference: float


def reference_gradient(gradients: Optional[np.ndarray]) -> float:
    if gradients is None or len(gradients) == 0:
        return 0.0
    return float(np.hypot(gradients[:, 0], gradients[:, 1]).max())


def arrows(positions: np.ndarray, gradients: np.ndarray, xlim, ylim, length: float, descent: bool = True,
           reference: float = 0.0, relative_to_view: bool = False) -> Optional[Arrows]:
    """Eq. (3.2); ``None`` if nothing is to be drawn."""

    visible = np.flatnonzero(inside(positions, xlim, ylim))
    if visible.size == 0:
        return None
    magnitude = np.hypot(gradients[visible, 0], gradients[visible, 1])
    if visible.size > MAX_ARROWS:
        visible = visible[np.argsort(magnitude)[-MAX_ARROWS:]]
        magnitude = np.hypot(gradients[visible, 0], gradients[visible, 1])
    largest = float(magnitude.max()) if relative_to_view else float(reference)
    if not largest > 0.0:
        return None
    scale = (-1.0 if descent else 1.0) * length / largest
    return Arrows(positions[visible, 1], positions[visible, 0],
                  gradients[visible, 1] * scale, gradients[visible, 0] * scale, largest)


def display_limits(array: np.ndarray) -> Tuple[float, float]:
    """Colour limits that are always valid (finite, ``low < high``)."""

    if array is None or array.size == 0 or not np.isfinite(array).any():
        return 0.0, 1.0
    low, high = float(np.nanmin(array)), float(np.nanmax(array))
    if not high > low:
        high = low + 1e-12
    return low, high


def window_limits(first: np.ndarray, second: np.ndarray, xlim, ylim) -> Optional[Tuple[float, float]]:
    """Common colour limits of two images inside a view (the comparison tile)."""

    x0, x1, y0, y1 = window(xlim, ylim)
    r0, r1 = int(max(np.floor(y0), 0)), int(min(np.ceil(y1) + 1, first.shape[0]))
    c0, c1 = int(max(np.floor(x0), 0)), int(min(np.ceil(x1) + 1, first.shape[1]))
    if r1 <= r0 or c1 <= c0:
        return None
    a, b = first[r0:r1, c0:c1], second[r0:r1, c0:c1]
    return display_limits(np.concatenate([a.ravel(), b.ravel()]))
