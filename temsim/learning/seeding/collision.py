r"""Part II, Section 6.2: collision rule for a new atom.

Two atoms of one layer cannot be closer than about a bond length.  A random
proposal :math:`\mathbf p` is therefore *resolved* against the present atoms with
two radii, :math:`d_{\rm touch} < d_{\rm merge}`:

* an atom within :math:`d_{\rm touch}` of :math:`\mathbf p` is absorbed (removed);
* the first atom :math:`\mathbf q` with :math:`d_{\rm touch} < |\mathbf p - \mathbf q| \le d_{\rm merge}`
  is merged with the proposal,

  .. math::

      \mathbf p \leftarrow \tfrac12(\mathbf p + \mathbf q),
      \qquad (6.1)

  and removed, i.e. the new atom takes the middle position;
* the two steps repeat from the new position until nothing changes (the merge
  may bring further atoms into range).

The result is a position and the list of atoms that disappear if the proposal is
accepted; the loss test of Section 6.5 then decides on the *combined* change.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def resolve_point(positions: np.ndarray, x: float, y: float, touch_px: float, merge_px: float,
                  exclude: Optional[Sequence[int]] = None) -> Tuple[Tuple[float, float], List[int], bool]:
    """Resolve a proposal ``(x, y)`` against ``positions`` ``(n, 2)``.

    Returns ``((x, y), removed indices, merged)``.  ``exclude`` lists atoms that do not take
    part (the atom that is being moved).
    """

    indices = np.arange(len(positions))
    if exclude is not None and len(exclude):
        indices = np.setdiff1d(indices, np.asarray(exclude, dtype=np.int64))
    coords = np.asarray(positions)[indices]
    graveyard: List[int] = []
    merged = False

    for _ in range(len(indices) + 2):
        if len(indices) == 0:
            break
        distance = np.hypot(coords[:, 0] - x, coords[:, 1] - y)
        changed = False

        for n in np.flatnonzero(distance <= touch_px):
            if indices[n] not in graveyard:
                graveyard.append(int(indices[n]))
                changed = True

        near = np.flatnonzero((distance <= merge_px) & (distance > touch_px))
        if len(near):
            n = near[0]
            x = coords[n, 0] + (x - coords[n, 0]) / 2.0
            y = coords[n, 1] + (y - coords[n, 1]) / 2.0
            if indices[n] not in graveyard:
                graveyard.append(int(indices[n]))
                changed = True
            merged = True

        if not changed:
            break

    return (x, y), graveyard, merged
