r"""Part V, Section 4: the plot canvas.

Theory
======

:class:`PlotCanvas` draws a :class:`~temsim_gui.frame_model.FrameModel` into the tiles
of Section 2.  It follows three rules.

* **Artists live as long as the layout.**  A new frame only replaces the data of
  existing artists, so the zoom (toolbar, mouse wheel) survives updates.  A layout
  is built again only when the view, the seeding flag or the model changes.
* **The loss curve is cheap.**  A ``light`` event redraws only the axes of the
  loss (blitting); drawing 15 panels of megapixel images would cost over a second.
* **Marks follow the zoom.**  Marker sizes and arrows (Section 3) are recomputed
  30 ms after the view stopped changing.

Implementation
==============
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6 import QtCore

from . import overlays, tiles
from .frame_model import FrameModel, LossHistory

MAX_POTENTIAL_PIXELS = 1024


class PlotCanvas(FigureCanvasQTAgg):
    view_changed = QtCore.Signal()

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 8), dpi=100, constrained_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)

        self.layer_names: List[str] = []
        self.stem = False
        self.seeding = False
        self.view: Optional[str] = None

        # settings of the view
        self.show_atoms = True
        self.min_occupancy = 0.5
        self.show_gradients = False
        self.gradient_descent = True
        self.gradient_length = 3.0
        self.gradient_relative_to_view = False
        self.show_minima = False
        self.minima_points = np.zeros((0, 2))
        self.minima_verified = True
        self.ctf_note = ""

        self.frame: Optional[FrameModel] = None
        self.history = LossHistory()
        self._reference = 0.0
        self._arrow_data = None
        self._quivers: list = []
        self._minima_artists: list = []
        self._shape = None

        self.plan = tiles.plan(None, 1)
        self.axes_by_key: Dict[str, object] = {}
        self.images: Dict[str, object] = {}
        self.loss_line = None
        self.target_scatter = None
        self.pot_scatters: List[object] = []
        self.hist_lines: list = []
        self.compare_scatters: Dict[str, object] = {}

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self.refresh_marks)

        self._build()
        self.mpl_connect("scroll_event", self._on_scroll)

    # ------------------------------------------------------------------ layout
    def configure(self, layer_names, stem: bool) -> bool:
        """Adapt to a model; returns True if the layout changed."""

        layer_names, stem = list(layer_names), bool(stem)
        if layer_names == self.layer_names and stem == self.stem:
            return False
        self.layer_names, self.stem = layer_names, stem
        if self.view not in tiles.tile_keys(len(layer_names), self.seeding) and self.view != tiles.COMPARE:
            self.view = None
        self._build()
        return True

    def keys(self) -> List[str]:
        return tiles.tile_keys(len(self.layer_names), self.seeding)

    def titles(self) -> Dict[str, str]:
        return tiles.titles(self.layer_names, self.stem)

    def set_seeding(self, seeding: bool) -> bool:
        seeding = bool(seeding)
        if seeding == self.seeding:
            return False
        self.seeding = seeding
        if self.view not in self.keys() and self.view != tiles.COMPARE:
            self.view = None
        self._build()
        return True

    def set_view(self, view: Optional[str]) -> None:
        if view == self.view:
            return
        self.view = view
        self._build()

    def _build(self) -> None:
        self.fig.clear()
        self.plan = plan = tiles.plan(self.view, len(self.layer_names), self.seeding)
        titles = self.titles()
        grid = self.fig.add_gridspec(plan.rows, plan.columns, height_ratios=plan.height_ratios,
                                     width_ratios=plan.width_ratios)
        self.axes_by_key = {}
        for key in plan.keys:
            row, col, row_span, col_span = plan.cells[key]
            spec = grid[row:row + row_span, col:col + col_span]
            first = self.axes_by_key.get(plan.share[0]) if key in plan.share and plan.share[0] in self.axes_by_key else None
            ax = self.fig.add_subplot(spec, sharex=first, sharey=first) if first is not None else self.fig.add_subplot(spec)
            ax.set_title(titles.get(key, key))
            if key not in ("loss", "hist"):
                ax.set_xticks([])
                ax.set_yticks([])
            self.axes_by_key[key] = ax

        shape = self._shape or (64, 64)
        blank = np.zeros(shape, dtype=np.float32)
        self.images = {}

        def image(key, cmap, data=None, **kwargs):
            ax = self.axes_by_key.get(key)
            if ax is None:
                return None
            artist = ax.imshow(blank if data is None else data, cmap=cmap, origin="lower", interpolation="nearest",
                               **kwargs)
            self.images[key] = artist
            return artist

        self.loss_line = None
        if "loss" in self.axes_by_key:
            ax = self.axes_by_key["loss"]
            self.loss_line, = ax.plot([], [])
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            self._draw_history()

        image("sim", "viridis")
        image("diff", "RdBu_r" if self.view == tiles.COMPARE else "viridis")
        image("target", "viridis")
        self.target_scatter = None
        if "target" in self.axes_by_key:
            self.target_scatter = self.axes_by_key["target"].scatter([], [], c=[], cmap="Wistia", s=2)
        for key in tiles.STRAIN:
            image(key, "RdBu")
        small = np.zeros((10, 10), dtype=np.float32)
        self.ring = None
        if self.stem:
            image("ctf", "viridis", data=small)
        else:
            image("ctf", "RdBu", data=small, vmin=-np.pi, vmax=np.pi)
            if "ctf" in self.axes_by_key:
                self.ring = self.axes_by_key["ctf"].imshow(small, cmap="Greens", origin="lower",
                                                           interpolation="nearest", alpha=0.35)
                self.images["ring"] = self.ring
        self.pot_scatters = [image(f"pot{i}", "RdBu") for i in range(max(len(self.layer_names), 1))]

        for key in ("sim", "diff", "target"):
            ax = self.axes_by_key.get(key)
            if ax is not None:
                ax.set_autoscale_on(False)
                ax.callbacks.connect("xlim_changed", self._on_view_changed)
                ax.callbacks.connect("ylim_changed", self._on_view_changed)

        self.compare_scatters = {}
        if self.view == tiles.COMPARE:
            for key in ("sim", "diff"):
                self.compare_scatters[key] = self.axes_by_key[key].scatter(
                    [], [], s=20, facecolors="none", edgecolors="white", linewidths=0.8)

        self.hist_lines = []
        if "hist" in self.axes_by_key:
            for i in range(max(len(self.layer_names), 1)):
                label = self.layer_names[i] if i < len(self.layer_names) else str(i)
                line, = self.axes_by_key["hist"].plot([], [], label=label)
                self.hist_lines.append(line)
            self.axes_by_key["hist"].legend(fontsize="x-small")

        self._quivers, self._minima_artists = [], []
        if self.frame is not None:
            self.update_frame(self.frame)
        self.draw_idle()

    # ------------------------------------------------------------------ data
    def _draw_history(self) -> None:
        ax = self.axes_by_key.get("loss")
        if ax is None or self.loss_line is None:
            return
        steps, values = self.history.arrays()
        if len(values) > 1:
            self.loss_line.set_data(steps, values)
            ax.relim()
            ax.autoscale_view()

    def update_loss(self) -> bool:
        """Redraw only the loss axes; ``False`` if the whole figure had to be redrawn."""

        ax = self.axes_by_key.get("loss")
        if ax is None or self.loss_line is None:
            return True
        self._draw_history()
        try:
            renderer = self.get_renderer()
            ax.draw(renderer)
            box = ax.get_tightbbox(renderer)
            self.blit(box.expanded(1.15, 1.15) if box is not None else ax.bbox)
            return True
        except Exception:
            self.draw_idle()
            return False

    @staticmethod
    def _set_image(artist, array: np.ndarray, symmetric: bool = False) -> None:
        artist.set_data(array)
        if symmetric:
            low, high = overlays.display_limits(np.abs(array))
            artist.set_clim(-high, high)
        else:
            artist.set_clim(*overlays.display_limits(array))

    def _fit(self, artist, ax, array: np.ndarray) -> None:
        """Set an image together with its extent; reset the view if its shape changed."""

        shape = array.shape
        artist.set_extent((-0.5, shape[1] - 0.5, -0.5, shape[0] - 0.5))
        ax.set_xlim(-0.5, shape[1] - 0.5)
        ax.set_ylim(-0.5, shape[0] - 0.5)

    def update_frame(self, frame: FrameModel) -> None:
        self.frame = frame
        images, axes = self.images, self.axes_by_key
        reset = self._shape != frame.shape
        self._shape = frame.shape
        step = frame.step

        for key, array in (("sim", frame.sim), ("diff", frame.diff), ("target", frame.target)):
            artist = images.get(key)
            if artist is None:
                continue
            self._set_image(artist, array)
            if reset or artist.get_extent()[1] != frame.shape[1] - 0.5:
                self._fit(artist, axes[key], array)
        if self.view == tiles.COMPARE:
            self.images["diff"].set_cmap("RdBu_r")
        if "target" in axes:
            axes["target"].set_title(f"Target (Step {step})")
            self._update_target_atoms()

        for key in tiles.STRAIN:
            if key in images and key in frame.strain:
                self._set_image(images[key], frame.strain[key], symmetric=True)
        if frame.active_group is not None:
            for key in ("ux", "uy"):
                if key in axes:
                    axes[key].set_title(f"{key} ({frame.active_group})")

        self._update_optics(frame)
        for i, artist in enumerate(self.pot_scatters):
            potential = frame.layers[i].potential if i < len(frame.layers) else None
            if artist is not None and potential is not None:
                stride = max(1, int(np.ceil(max(potential.shape) / MAX_POTENTIAL_PIXELS)))
                self._set_image(artist, potential[::stride, ::stride])
                artist.set_extent((-0.5, potential.shape[1] - 0.5, -0.5, potential.shape[0] - 0.5))

        if "hist" in axes:
            counts, edges = frame.histogram(self.min_occupancy)
            centres = 0.5 * (edges[:-1] + edges[1:])
            top = 1.0
            for line, values in zip(self.hist_lines, counts):
                line.set_data(centres, values)
                top = max(top, float(values.max()) if len(values) else 0.0)
            axes["hist"].set_xlim(float(centres[0]), float(centres[-1]))
            axes["hist"].set_ylim(0, top * 1.05)

        self._arrow_data = frame.gradients(self.min_occupancy)
        self._reference = overlays.reference_gradient(None if self._arrow_data is None else self._arrow_data[1])
        self.refresh_marks()

    def _update_optics(self, frame: FrameModel) -> None:
        ax = self.axes_by_key.get("ctf")
        if ax is None:
            return
        name = "Probe" if self.stem else "CTF"
        if self.stem and frame.probe is not None:
            probe = frame.probe
            artist = self.images["ctf"]
            artist.set_data(probe)
            artist.set_extent((-0.5, probe.shape[1] - 0.5, -0.5, probe.shape[0] - 0.5))
            artist.set_clim(*overlays.display_limits(probe))
            ax.set_xlim(-0.5, probe.shape[1] - 0.5)
            ax.set_ylim(-0.5, probe.shape[0] - 0.5)
        elif not self.stem:
            if frame.ctf_phase is not None:
                self.images["ctf"].set_data(frame.ctf_phase)
            if frame.ctf_ring is not None and "ring" in self.images:
                self.images["ring"].set_data(frame.ctf_ring)
        ax.set_title(f"{name} (Step {frame.step})" + (f"\n{self.ctf_note}" if self.ctf_note else ""))

    def _update_target_atoms(self) -> None:
        scatter, ax, frame = self.target_scatter, self.axes_by_key.get("target"), self.frame
        if scatter is None or frame is None:
            return
        pos, ints = frame.atoms(self.min_occupancy, inside_mask=True) if self.show_atoms else (np.zeros((0, 2)), [])
        if len(ints) == 0:
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_array(np.array([], dtype=np.float32))
            return
        scatter.set_offsets(np.column_stack([pos[:, 1], pos[:, 0]]))
        scatter.set_array(ints)
        scatter.set_clim(*overlays.display_limits(ints))
        scatter.set_sizes(np.full(len(ints), self._marker_size(ax)))

    # ------------------------------------------------------------------ marks that follow the zoom
    def _marker_size(self, ax) -> float:
        if ax is None or self._shape is None:
            return 0.5
        x0, x1, y0, y1 = overlays.window(ax.get_xlim(), ax.get_ylim())
        return overlays.scatter_size(max(x1 - x0, y1 - y0))

    def overlay_keys(self) -> List[str]:
        if self.view == tiles.COMPARE:
            return ["sim", "target", "diff"]
        if self.view in ("sim", "diff", "target"):
            return [self.view]
        return ["target"]

    def refresh_marks(self) -> None:
        self._refresh_arrows()
        self._refresh_compare()
        self._refresh_minima()
        if self.target_scatter is not None and len(self.target_scatter.get_offsets()):
            self.target_scatter.set_sizes(np.full(len(self.target_scatter.get_offsets()),
                                                  self._marker_size(self.axes_by_key.get("target"))))
        self.draw_idle()

    @staticmethod
    def _drop(artists: list) -> None:
        for artist in artists:
            try:
                artist.remove()
            except Exception:
                pass
        artists.clear()

    def _refresh_arrows(self) -> None:
        self._drop(self._quivers)
        for key in ("sim", "diff", "target"):
            if key in self.axes_by_key:
                self.axes_by_key[key].set_xlabel("")
        if not self.show_gradients or self._arrow_data is None:
            return
        positions, gradients = self._arrow_data
        for key in self.overlay_keys():
            ax = self.axes_by_key.get(key)
            if ax is None:
                continue
            found = overlays.arrows(positions, gradients, ax.get_xlim(), ax.get_ylim(), self.gradient_length,
                                    self.gradient_descent, self._reference, self.gradient_relative_to_view)
            if found is None:
                continue
            self._quivers.append(ax.quiver(found.columns, found.rows, found.d_column, found.d_row, angles="xy",
                                           scale_units="xy", scale=1.0, color="magenta", width=0.004,
                                           headwidth=4.5, headlength=5, zorder=5))
            ax.set_xlabel(f"{'-' if self.gradient_descent else ''}Gradient, Pfeillaenge ~ |g|: "
                          f"{self.gradient_length:g} px bei |g| = {found.reference:.3g}", fontsize="small")

    def _refresh_compare(self) -> None:
        frame = self.frame
        if self.view != tiles.COMPARE or frame is None:
            return
        pos, _ = frame.atoms(self.min_occupancy, inside_mask=True) if self.show_atoms else (np.zeros((0, 2)), None)
        for key, scatter in self.compare_scatters.items():
            ax = self.axes_by_key[key]
            keep = overlays.inside(pos, ax.get_xlim(), ax.get_ylim()) if len(pos) else np.zeros(0, bool)
            scatter.set_offsets(np.column_stack([pos[keep, 1], pos[keep, 0]]) if keep.any() else np.empty((0, 2)))
            scatter.set_sizes(np.full(int(keep.sum()), self._marker_size(ax) * 2))
        ax = self.axes_by_key["sim"]
        limits = overlays.window_limits(frame.sim, frame.target, ax.get_xlim(), ax.get_ylim())
        if limits is None:
            return
        self.images["sim"].set_clim(*limits)
        self.images["target"].set_clim(*limits)
        x0, x1, y0, y1 = overlays.window(ax.get_xlim(), ax.get_ylim())
        r0, r1 = int(max(np.floor(y0), 0)), int(min(np.ceil(y1) + 1, frame.diff.shape[0]))
        c0, c1 = int(max(np.floor(x0), 0)), int(min(np.ceil(x1) + 1, frame.diff.shape[1]))
        if r1 > r0 and c1 > c0:
            limit = float(np.nanmax(np.abs(frame.diff[r0:r1, c0:c1]))) or 1e-12
            self.images["diff"].set_clim(-limit, limit)
        for key, title in (("sim", "Simulated"), ("target", "Target"), ("diff", "Diff (Sim - Target)")):
            self.axes_by_key[key].set_title(f"{title} (Step {frame.step})")

    def _refresh_minima(self) -> None:
        self._drop(self._minima_artists)
        for ax in self.axes_by_key.values():
            ax.set_title(ax.get_title().split(" [")[0])
        if not self.show_minima:
            return
        points = self.minima_points
        for key in ("diff", "sim", "target"):
            ax = self.axes_by_key.get(key)
            if ax is None or (key != "diff" and self.view != tiles.COMPARE and key != self.view):
                continue
            if len(points):
                self._minima_artists.append(ax.scatter(
                    points[:, 1], points[:, 0], s=self._marker_size(ax) * 3, marker="x",
                    c="lime" if self.minima_verified else "yellow", linewidths=1.2, zorder=6))
            note = "" if self.minima_verified else ", ohne Loss-Test"
            ax.set_title(f"{ax.get_title().split(' [')[0]} [{len(points)} Minima{note}]")

    def set_minima(self, points: Optional[np.ndarray], verified: bool = True) -> None:
        self.minima_points = np.zeros((0, 2)) if points is None else np.asarray(points, dtype=float).reshape(-1, 2)
        self.minima_verified = verified
        self._refresh_minima()
        self.draw_idle()

    # ------------------------------------------------------------------ zoom
    def reset_zoom(self) -> None:
        for key, artist in self.images.items():
            ax = self.axes_by_key.get(key)
            data = artist.get_array()
            if ax is None or data is None or np.ndim(data) < 2 or key == "ring":
                continue
            ax.set_xlim(-0.5, data.shape[1] - 0.5)
            ax.set_ylim(-0.5, data.shape[0] - 0.5)
        self.draw_idle()

    def _on_view_changed(self, _ax) -> None:
        self._timer.start()
        self.view_changed.emit()

    def _on_scroll(self, event) -> None:
        ax = event.inaxes
        if ax is None or event.xdata is None:
            return
        key = next((k for k, a in self.axes_by_key.items() if a is ax), None)
        if key not in self.images:
            return
        factor = 0.8 if event.button == "up" else 1.25
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(event.xdata + (x0 - event.xdata) * factor, event.xdata + (x1 - event.xdata) * factor)
        ax.set_ylim(event.ydata + (y0 - event.ydata) * factor, event.ydata + (y1 - event.ydata) * factor)
        self.draw_idle()
