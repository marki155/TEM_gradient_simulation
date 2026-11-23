r"""Part II, Section 8.1: atom positions as images, and back.

Theory
======

A reconstruction ends with a list of atoms, each with a position
:math:`\mathbf r_k` (pixels) and an intensity :math:`I_k` (Part I, Eq. 4.3).  For a
network that maps an image to an image this list is written as a *heat map*

.. math::

   h(\mathbf x) = \max_k \; \frac{I_k}{s}\,
                  \exp\!\Big(-\frac{|\mathbf x - \mathbf r_k|^2}{2\sigma^2}\Big),      \tag{8.1}

a Gaussian spot of width :math:`\sigma` per atom whose *height* is the intensity in
units of a fixed scale :math:`s` (the median intensity of the first training run).
The maximum instead of the sum keeps the height of a spot equal to
:math:`I_k / s` also where two atoms are close.

The way back is the search for local maxima of :math:`h` above a threshold
:math:`t` that are at least :math:`d_{\min}` apart; the position is refined to a
fraction of a pixel in the :math:`3\times 3`
neighbourhood by a parabola through the logarithms of the three values along each
axis (exact for a Gaussian), and the top of that parabola times :math:`s` is the
predicted intensity.

To compare a prediction with the fitted atoms, both lists are *matched*: a pair
counts if the two points are closer than a tolerance, every point takes part in at
most one pair, and the closest pairs are formed first.  Predicted points without a
partner are atoms the fit lacks; fitted atoms without a partner are atoms the
network does not confirm.

Implementation
==============
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def heatmap(shape, positions, heights, sigma_px: float) -> np.ndarray:
    """Eq. (8.1): ``(H, W)`` float32 with one Gaussian of height ``heights[k]`` at ``positions[k]`` (row, col)."""

    out = np.zeros(tuple(shape), dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    heights = np.asarray(heights, dtype=np.float64).reshape(-1)
    sigma = max(float(sigma_px), 1e-3)
    reach = int(np.ceil(3.0 * sigma))
    n_rows, n_cols = out.shape
    for (row, col), height in zip(positions, heights):
        r0, r1 = max(int(np.floor(row)) - reach, 0), min(int(np.floor(row)) + reach + 2, n_rows)
        c0, c1 = max(int(np.floor(col)) - reach, 0), min(int(np.floor(col)) + reach + 2, n_cols)
        if r0 >= r1 or c0 >= c1:
            continue
        rr = (np.arange(r0, r1) - row)[:, None]
        cc = (np.arange(c0, c1) - col)[None, :]
        spot = (height * np.exp(-(rr ** 2 + cc ** 2) / (2.0 * sigma ** 2))).astype(np.float32)
        np.maximum(out[r0:r1, c0:c1], spot, out=out[r0:r1, c0:c1])
    return out


def find_peaks(heat: np.ndarray, threshold: float, min_distance_px: float = 1.0,
               valid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Local maxima of ``heat`` above ``threshold``, at least ``min_distance_px`` apart.

    Returns ``(positions (n, 2) row/col with sub-pixel centroid, heights (n,))``, highest first.
    """

    heat = np.asarray(heat, dtype=np.float32)
    # local maxima of the 3x3 neighbourhood; the distance is enforced between peaks below.  A maximum filter
    # of radius d would also drop a weak atom whose bright neighbour is a little more than d away, because the
    # flank of that neighbour is higher than the weak peak within d.
    peaks = (heat == ndimage.maximum_filter(heat, size=3, mode="nearest")) & (heat > float(threshold))
    if valid is not None:
        peaks &= np.asarray(valid, dtype=bool)
    rows, cols = np.nonzero(peaks)
    if len(rows) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    padded = np.log(np.clip(np.pad(heat, 1, mode="edge").astype(np.float64), 1e-6, None))
    centre = padded[rows + 1, cols + 1]
    log_height = centre.copy()
    shifts = []
    for dr, dc in ((1, 0), (0, 1)):
        low, high = padded[rows + 1 - dr, cols + 1 - dc], padded[rows + 1 + dr, cols + 1 + dc]
        curve = low - 2.0 * centre + high
        ok = curve < -1e-12
        shift = np.where(ok, 0.5 * (low - high) / np.where(ok, curve, -1.0), 0.0).clip(-0.5, 0.5)
        log_height -= np.where(ok, 0.5 * curve * shift ** 2, 0.0)      # top of the parabola through the three values
        shifts.append(shift)
    heights = np.exp(log_height)
    positions = np.column_stack([rows + shifts[0], cols + shifts[1]]).astype(np.float64)
    # plateaus give several maxima: keep the highest one within the distance
    order = np.argsort(-heights, kind="stable")
    positions, heights = positions[order], heights[order]
    keep = np.ones(len(positions), dtype=bool)
    tree = cKDTree(positions)
    for i in range(len(positions)):
        if keep[i]:
            for j in tree.query_ball_point(positions[i], r=float(min_distance_px) - 1e-9):
                if j > i:
                    keep[j] = False
    return positions[keep], heights[keep]


def match(predicted: np.ndarray, fitted: np.ndarray, tolerance_px: float) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one matching (closest pairs first); returns the boolean masks ``(pred_matched, fit_matched)``."""

    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 2)
    fitted = np.asarray(fitted, dtype=np.float64).reshape(-1, 2)
    pred_ok, fit_ok = np.zeros(len(predicted), bool), np.zeros(len(fitted), bool)
    if len(predicted) == 0 or len(fitted) == 0:
        return pred_ok, fit_ok
    pairs = cKDTree(predicted).sparse_distance_matrix(cKDTree(fitted), float(tolerance_px), output_type="ndarray")
    for entry in pairs[np.argsort(pairs["v"], kind="stable")]:
        i, j = int(entry["i"]), int(entry["j"])
        if not pred_ok[i] and not fit_ok[j]:
            pred_ok[i] = fit_ok[j] = True
    return pred_ok, fit_ok


def scores(predicted: np.ndarray, truth: np.ndarray, tolerance_px: float) -> Tuple[float, float, float]:
    """``(precision, recall, F1)`` of ``predicted`` against ``truth`` (1 for two empty lists)."""

    p, t = match(predicted, truth, tolerance_px)
    hits = int(p.sum())
    precision = hits / len(p) if len(p) else float(len(t) == 0)
    recall = hits / len(t) if len(t) else float(len(p) == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1
