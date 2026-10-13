r"""Part II, Section 7: candidate sites from the difference image.

Where the target is brighter than the simulation an atom is missing.  With the
standardised images :math:`x` (simulation) and :math:`y` (target) of Section 2 the
difference :math:`\delta = x - y` is negative there.  Candidate sites are the local
minima of :math:`\delta` below a fraction of its global minimum,

.. math::

    \delta(\mathbf p) = \min_{|\mathbf q-\mathbf p|_\infty\le r}\delta(\mathbf q)
    \quad\text{and}\quad \delta(\mathbf p)\le\theta\,\min\delta,
    \qquad (7.1)

with the exclusion radius :math:`r` (a minimum within :math:`r` of a deeper one is
dropped) and the threshold :math:`\theta\in(0,1]`.  The sites are sorted
by depth.  Whether a site becomes an atom is decided by the criteria of Section 6
(:meth:`~temsim.learning.seeding.seeder.Seeder.filter_points`); without a
seeder a minimum distance to the existing atoms is applied instead.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def minima_positions(diff: np.ndarray, threshold: float = 0.5, distance_px: float = 3.0,
                     atoms: Optional[np.ndarray] = None) -> np.ndarray:
    """Local minima of ``diff`` (``(n, 2)`` rows, cols; deepest first), Eq. (7.1)."""

    diff = np.nan_to_num(np.asarray(diff, dtype=np.float64))
    lowest = float(diff.min())
    if not lowest < 0.0:
        return np.zeros((0, 2))
    size = 2 * int(round(max(distance_px, 1.0))) + 1
    is_min = (diff == ndimage.minimum_filter(diff, size=size)) & (diff <= threshold * lowest)
    rows, cols = np.nonzero(is_min)
    points = np.column_stack([rows, cols]).astype(np.float64)
    if len(points) == 0:
        return points
    points = points[np.argsort(diff[rows, cols])]
    if atoms is not None and len(atoms):
        distance, _ = cKDTree(np.asarray(atoms)).query(points)
        points = points[distance >= distance_px]
    return points
