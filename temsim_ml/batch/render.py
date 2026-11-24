r"""Part V, Section 14: pictures of the found atoms.

Every atom is a dot at its position, coloured by its intensity.  One colour scale
holds for all frames of an input (the 1st to 99th percentile of all intensities of
the input), so that frames of a stack can be compared.  Two versions: on the target
(grey) and on a white background.

* ``atoms_target.tif`` / ``atoms.tif`` -- RGB at the resolution of the image; a stack
  gives a multi-page TIF with one page per frame;
* ``png/`` -- the same with a colour bar, one PNG per frame (single images: next to the TIFs).

The dot radius is 0.3 Å, at least 1.5 pixels.
"""

from __future__ import annotations

import csv
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

COLORMAP = "turbo"
DOT_RADIUS_A = 0.3


def read_atoms(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    positions = np.array([[float(r["row_px"]), float(r["col_px"])] for r in rows]).reshape(-1, 2)
    intensity = np.array([float(r["intensity"]) for r in rows])
    return positions, intensity


def write_atoms(path: str, positions: np.ndarray, intensity: np.ndarray, layers: Optional[np.ndarray] = None) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_px", "col_px", "intensity", "layer"])
        for k, ((row, col), value) in enumerate(zip(positions, intensity)):
            writer.writerow([f"{row:.3f}", f"{col:.3f}", f"{value:.5g}", 0 if layers is None else int(layers[k])])


def color_range(intensities: Sequence[np.ndarray]) -> Tuple[float, float]:
    values = np.concatenate([np.asarray(v, dtype=float) for v in intensities if v is not None and len(v)] or [[0.0]])
    low, high = np.percentile(values, [1, 99]) if len(values) > 1 else (values[0] - 0.5, values[0] + 0.5)
    if high <= low:
        high = low + 1e-6
    return float(low), float(high)


def _figure(shape, colorbar: bool):
    from matplotlib.figure import Figure
    h, w = shape
    dpi = 100.0
    extra = 1.3 if colorbar else 0.0
    fig = Figure(figsize=(w / dpi + extra, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, (w / dpi) / (w / dpi + extra), 1])
    ax.set_axis_off()
    return fig, ax


def draw(target: Optional[np.ndarray], shape, positions, intensity, vmin, vmax, pixel_A: float,
         colorbar: bool = False, title: str = "") -> np.ndarray:
    """RGB ``uint8``; ``target=None``: white background."""

    fig, ax = _figure(shape, colorbar)
    h, w = shape
    if target is None:
        ax.imshow(np.ones((h, w, 3)), interpolation="nearest")
    else:
        lo, hi = np.percentile(target, [0.5, 99.5])
        ax.imshow(target, cmap="gray", vmin=lo, vmax=hi if hi > lo else lo + 1, interpolation="nearest")
    radius_px = max(1.5, DOT_RADIUS_A / max(pixel_A, 1e-9))
    size_pt = 2 * radius_px * 72.0 / fig.dpi
    art = ax.scatter(positions[:, 1] if len(positions) else [], positions[:, 0] if len(positions) else [],
                     c=intensity if len(positions) else [], cmap=COLORMAP, vmin=vmin, vmax=vmax,
                     s=size_pt ** 2, linewidths=0)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    if colorbar:
        width = fig.get_figwidth()
        cax = fig.add_axes([(w / fig.dpi) / width + 0.15 / width, 0.1, 0.25 / width, 0.8])
        bar = fig.colorbar(art, cax=cax)
        bar.set_label("Intensität")
        if title:
            ax.text(4, 4, title, color="yellow", fontsize=9, va="top",
                    bbox={"facecolor": "black", "alpha": 0.5, "linewidth": 0})
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def render_input(folder: str, targets: np.ndarray, atoms: List[Optional[Tuple[np.ndarray, np.ndarray]]],
                 pixel_A: float) -> List[str]:
    """The pictures of one input (``targets``: ``(n, H, W)``; ``atoms[k]`` None: frame failed, no dots)."""

    import tifffile
    stack = len(targets) > 1
    vmin, vmax = color_range([a[1] for a in atoms if a is not None])
    empty = (np.zeros((0, 2)), np.zeros(0))
    png_dir = os.path.join(folder, "png") if stack else folder
    os.makedirs(png_dir, exist_ok=True)
    written = []
    for background, suffix in ((True, "_target"), (False, "")):
        pages = []
        for k, target in enumerate(targets):
            positions, intensity = atoms[k] if atoms[k] is not None else empty
            shown = target if background else None
            pages.append(draw(shown, target.shape, positions, intensity, vmin, vmax, pixel_A))
            title = f"Frame {k}" if stack else ""
            name = f"atoms{suffix}_{k:03d}.png" if stack else f"atoms{suffix}.png"
            picture = draw(shown, target.shape, positions, intensity, vmin, vmax, pixel_A, colorbar=True, title=title)
            path = os.path.join(png_dir, name)
            _write_png(path, picture)
            written.append(path)
        path = os.path.join(folder, f"atoms{suffix}.tif")
        tifffile.imwrite(path, np.stack(pages) if stack else pages[0], photometric="rgb")
        written.append(path)
    return written


def _write_png(path: str, rgb: np.ndarray) -> None:
    from matplotlib import image
    image.imsave(path, rgb)
