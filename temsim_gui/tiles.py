r"""Part V, Section 2: the tiles of the plot area.

Theory
======

The plot area is a grid of *tiles*; every tile shows one quantity.  The set and
order of the tiles follows from the model alone:

.. code-block:: text

    loss  sim  diff  target  ctf | pot0 ... pot(L-1) | hist  ux  uy  eps_xx  eps_yy  eps_xy

(``ctf`` is the probe in STEM; ``L`` is the number of layers).  While seeding the
displacement field is not trained, so the five strain tiles are left out.

Three *views* exist: all tiles in a grid of five columns; one tile large with the
loss curve beside it; and the *comparison* ``sim | target | diff`` with a common
zoom and the loss underneath.  :func:`plan` turns a view into rows, columns and
the positions of the tiles, so that the canvas only has to draw.

Implementation
==============
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

LEADING = ("loss", "sim", "diff", "target", "ctf")
TRAILING = ("hist", "ux", "uy", "eps_xx", "eps_yy", "eps_xy")
STRAIN = ("ux", "uy", "eps_xx", "eps_yy", "eps_xy")
IMAGE_TILES = ("sim", "diff", "target", "ctf") + STRAIN
COMPARE = "compare"
COMPARE_TILES = ("sim", "target", "diff", "loss")
COLUMNS = 5


def tile_keys(n_layers: int, seeding: bool = False) -> List[str]:
    """All tiles in grid order."""

    trailing = [k for k in TRAILING if not (seeding and k in STRAIN)]
    return list(LEADING) + [f"pot{i}" for i in range(max(n_layers, 1))] + trailing


def titles(layer_names: Sequence[str], stem: bool = False) -> Dict[str, str]:
    out = {COMPARE: "Zoom-Vergleich (Sim | Target | Diff + Loss)",
           "loss": "Loss", "sim": "Simulated", "diff": "Diff", "target": "Target",
           "ctf": "Probe" if stem else "CTF", "hist": "Histogramm",
           "ux": "ux", "uy": "uy", "eps_xx": "eps xx", "eps_yy": "eps yy", "eps_xy": "eps xy"}
    for i in range(max(len(layer_names), 1)):
        out[f"pot{i}"] = f"Potential {layer_names[i] if i < len(layer_names) else i}"
    return out


@dataclass(frozen=True)
class Plan:
    """Which tiles are drawn and where.

    ``cells`` maps a tile to ``(row, column, row_span, column_span)`` in a grid of
    ``rows x columns``; ``share`` lists tiles whose axes are linked.
    """

    view: Optional[str]
    keys: Tuple[str, ...]
    rows: int
    columns: int
    cells: Dict[str, Tuple[int, int, int, int]]
    height_ratios: Optional[Tuple[float, ...]] = None
    width_ratios: Optional[Tuple[float, ...]] = None
    share: Tuple[str, ...] = ()


def plan(view: Optional[str], n_layers: int, seeding: bool = False) -> Plan:
    """The layout for ``view`` (``None``: all tiles; :data:`COMPARE`; or one tile's key)."""

    keys = tile_keys(n_layers, seeding)
    if view == COMPARE:
        cells = {"sim": (0, 0, 1, 1), "target": (0, 1, 1, 1), "diff": (0, 2, 1, 1), "loss": (1, 0, 1, 3)}
        return Plan(COMPARE, COMPARE_TILES, 2, 3, cells, height_ratios=(3.0, 1.0), share=("sim", "target", "diff"))
    if view in keys:
        if view == "loss":
            return Plan(view, ("loss",), 1, 1, {"loss": (0, 0, 1, 1)})
        return Plan(view, (view, "loss"), 1, 2, {view: (0, 0, 1, 1), "loss": (0, 1, 1, 1)},
                    width_ratios=(2.5, 1.0))
    columns = COLUMNS
    cells = {key: (i // columns, i % columns, 1, 1) for i, key in enumerate(keys)}
    return Plan(None, tuple(keys), -(-len(keys) // columns), columns, cells)
