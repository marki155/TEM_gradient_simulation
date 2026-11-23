r"""Part V, Section 13: a small interface that only predicts.

A trained detector (Part II, Section 8) is a file on disk; it needs neither the
core nor a reconstruction to be used.  This window loads a model (from the global
directory ``~/data/models`` or any folder), the two input images (``target`` and
``i_target``) and shows the atoms the network finds, coloured by their predicted
intensity.

It is used two ways:

* **standalone** -- ``python -m temsim_ml.predict``; the result is written out
  (``ml_detections.csv``, ``ml_heatmap.npy``);
* **from the main interface** -- which starts it with the images of the open run
  (``--target``, ``--i-target``) and a ``--handoff`` file: "Übernehmen" writes the
  confirmed positions and intensities there (``positions``, ``intensity`` in an
  ``.npz``) and closes; the main interface reads them, and the core decides which
  of them the fit is still missing.

Implementation
==============
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Callable, List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

from temsim.data.dataset import read_image
from temsim.learning import detection as det
from temsim_gui.widgets import double_spin

from .models import GLOBAL_MODELS, list_models

IMAGE_FILTER = "Bilder (*.npy *.tif *.tiff)"


def _device(name: str) -> str:
    if name == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


class PredictorWindow(QtWidgets.QMainWindow):
    """Load a model and two images, predict atoms, show, and either export or hand back."""

    def __init__(self, device: str = "auto", on_confirm: Optional[Callable[[np.ndarray, np.ndarray], None]] = None):
        super().__init__()
        self.setWindowTitle("TEM-Detektor: Vorhersage")
        self.resize(1100, 820)
        self.device = _device(device)
        self.on_confirm = on_confirm

        self.net = None
        self.meta: dict = {}
        self.model_dir: Optional[str] = None
        self.target: Optional[np.ndarray] = None
        self.second: Optional[np.ndarray] = None
        self.target_path: Optional[str] = None
        self.heat: Optional[np.ndarray] = None
        self.positions = np.zeros((0, 2))
        self.intensity = np.zeros(0)

        self._build()
        self.refresh_models()
        self._update_enabled()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        side = QtWidgets.QVBoxLayout()
        layout.addLayout(side, 0)

        side.addWidget(QtWidgets.QLabel("<b>Modell (global)</b>"))
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.currentIndexChanged.connect(self._model_chosen)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.model_combo, 1)
        refresh = QtWidgets.QPushButton("↻")
        refresh.setFixedWidth(30)
        refresh.setToolTip("Modellliste neu lesen")
        refresh.clicked.connect(self.refresh_models)
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(30)
        browse.setToolTip("Anderen Modellordner wählen")
        browse.clicked.connect(self.choose_model)
        row.addWidget(refresh)
        row.addWidget(browse)
        side.addLayout(row)
        self.model_label = QtWidgets.QLabel("Kein Modell geladen")
        self.model_label.setWordWrap(True)
        side.addWidget(self.model_label)

        side.addWidget(QtWidgets.QLabel("<b>Bilder</b>"))
        self.target_label = QtWidgets.QLabel("Kein target")
        self.target_label.setWordWrap(True)
        load_target = QtWidgets.QPushButton("target laden …")
        load_target.clicked.connect(lambda: self.choose_image(False))
        self.second_label = QtWidgets.QLabel("Kein i_target")
        self.second_label.setWordWrap(True)
        load_second = QtWidgets.QPushButton("i_target laden …")
        load_second.clicked.connect(lambda: self.choose_image(True))
        for w in (load_target, self.target_label, load_second, self.second_label):
            side.addWidget(w)

        form = QtWidgets.QFormLayout()
        self.threshold = double_spin(0.3, 0.0, 5.0, 0.05, 2,
                                     tip="Mindesthöhe eines Peaks in der Heatmap (Einheit: Intensität/scale).")
        self.min_distance = double_spin(0.8, 0.05, 20.0, 0.05, 2, suffix=" A",
                                        tip="Kleinster erlaubter Abstand zweier Atome. Kleiner als der "
                                            "Abstand benachbarter Atome wählen.")
        form.addRow("Schwelle", self.threshold)
        form.addRow("Mindestabstand", self.min_distance)
        side.addLayout(form)

        self.predict_btn = QtWidgets.QPushButton("Vorhersagen")
        self.predict_btn.clicked.connect(self.predict)
        side.addWidget(self.predict_btn)

        self.show_overlay = QtWidgets.QCheckBox("Atome einblenden")
        self.show_overlay.setChecked(True)
        self.show_overlay.toggled.connect(self._draw)
        side.addWidget(self.show_overlay)

        self.export_btn = QtWidgets.QPushButton("Exportieren (CSV + Heatmap) …")
        self.export_btn.clicked.connect(self.export)
        side.addWidget(self.export_btn)

        # only when driven by the main interface
        self.confirm_btn = QtWidgets.QPushButton("Übernehmen → Haupt-GUI")
        self.confirm_btn.setToolTip("Die vorhergesagten Positionen an die Haupt-GUI übergeben. Dort wird "
                                    "entschieden, welche dem Fit fehlen, und diese werden eingefügt.")
        self.confirm_btn.clicked.connect(self.confirm)
        self.confirm_btn.setVisible(self.on_confirm is not None)
        side.addWidget(self.confirm_btn)

        self.result_label = QtWidgets.QLabel("")
        self.result_label.setWordWrap(True)
        side.addWidget(self.result_label)
        side.addStretch(1)

        plot = QtWidgets.QVBoxLayout()
        layout.addLayout(plot, 1)
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        plot.addWidget(NavigationToolbar2QT(self.canvas, self))
        plot.addWidget(self.canvas, 1)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_axis_off()

    # ------------------------------------------------------------------ models
    def refresh_models(self) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("— wählen —", None)
        for entry in list_models(GLOBAL_MODELS):
            f1 = entry["meta"].get("val_f1")
            label = entry["name"] + (f"  (F1 {f1:.2f})" if isinstance(f1, (int, float)) else "")
            self.model_combo.addItem(label, entry["path"])
        self.model_combo.blockSignals(False)

    def _model_chosen(self) -> None:
        path = self.model_combo.currentData()
        if path:
            self._load_model(path)

    def choose_model(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Modellordner (enthält detector.pt)", GLOBAL_MODELS)
        if directory:
            self._load_model(directory)

    def _load_model(self, directory: str) -> None:
        if not det.is_detector(directory):
            self._warn("Kein detector.pt in diesem Ordner.")
            return
        try:
            self.net, self.meta = det.load_detector(directory, self.device)
        except Exception as error:                                   # noqa: BLE001 - shown to the user
            self._warn(f"Modell nicht ladbar: {error}")
            return
        self.model_dir = directory
        name = os.path.basename(directory.rstrip("/"))
        f1, scale, pixel = self.meta.get("val_f1"), self.meta.get("scale"), self.meta.get("pixel_A")
        parts = [f"Modell: {name}", f"Schritte {self.meta.get('steps', '?')}"]
        if isinstance(f1, (int, float)):
            parts.append(f"F1 {f1:.3f}")
        if pixel:
            parts.append(f"pixel {pixel} A")
        if isinstance(scale, (int, float)):
            parts.append(f"scale {scale:.3g}")
        self.model_label.setText("\n".join(parts))
        self.heat = None
        self._update_enabled()

    # ------------------------------------------------------------------ images
    def set_images(self, target: Optional[np.ndarray], second: Optional[np.ndarray], source: str = "") -> None:
        """Hand the two images in directly (used by the main interface for the open run)."""

        if target is not None:
            self.target = np.asarray(target, dtype=np.float32)
            self.target_path = None
            self.target_label.setText(f"target: {source or 'aus Haupt-GUI'}  {self.target.shape}")
        if second is not None:
            self.second = np.asarray(second, dtype=np.float32)
            self.second_label.setText(f"i_target: {source or 'aus Haupt-GUI'}  {self.second.shape}")
        self.heat = None
        self._update_enabled()
        self._draw()

    def choose_image(self, second: bool) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "i_target wählen" if second else "target wählen", "", IMAGE_FILTER)
        if not path:
            return
        try:
            image = np.asarray(read_image(path), dtype=np.float32)
        except Exception as error:                                   # noqa: BLE001 - shown to the user
            self._warn(f"Bild nicht lesbar: {error}")
            return
        if image.ndim != 2:
            self._warn(f"Erwartet ein 2D-Bild, bekommen {image.shape}.")
            return
        if second:
            self.second, label = image, self.second_label
        else:
            self.target, self.target_path, label = image, path, self.target_label
        label.setText(f"{'i_target' if second else 'target'}: {os.path.basename(path)}  {image.shape}")
        self.heat = None
        self._update_enabled()

    # ------------------------------------------------------------------ predict
    def _pixel_A(self) -> float:
        pixel = self.meta.get("pixel_A")
        return float(pixel) if pixel else 1.0

    def predict(self) -> None:
        if self.net is None or self.target is None or self.second is None:
            return
        if self.target.shape != self.second.shape:
            self._warn(f"target {self.target.shape} und i_target {self.second.shape} haben verschiedene Größen.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            image = det.inputs(self.target, self.second)
            heat = det.predict(self.net, image, self.device)
            min_px = max(1.0, float(self.min_distance.value()) / self._pixel_A())
            positions, heights = det.find_peaks(heat, float(self.threshold.value()), min_px)
            scale = float(self.meta.get("scale", 1.0))
            self.positions, self.intensity, self.heat = positions, heights * scale, heat * scale
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.result_label.setText(
            f"{len(self.positions)} Atome gefunden.\n"
            f"Intensität: Median {np.median(self.intensity):.3g}, "
            f"Bereich {self.intensity.min():.3g}–{self.intensity.max():.3g}"
            if len(self.positions) else "Keine Atome über der Schwelle.")
        self._update_enabled()
        self._draw()

    def _draw(self) -> None:
        self.figure.clear()                                      # also drops any earlier colorbar axes
        self.axes = self.figure.add_subplot(111)
        self.axes.set_axis_off()
        if self.target is not None:
            self.axes.imshow(self.target, cmap="gray", interpolation="nearest")
        if self.show_overlay.isChecked() and len(self.positions):
            scatter = self.axes.scatter(self.positions[:, 1], self.positions[:, 0], c=self.intensity,
                                        cmap="viridis", s=12, linewidths=0.4, edgecolors="black")
            bar = self.figure.colorbar(scatter, ax=self.axes, fraction=0.046, pad=0.04)
            bar.set_label("vorhergesagte Intensität I")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ outputs
    def export(self) -> None:
        if not len(self.positions):
            return
        start = os.path.dirname(self.target_path) if self.target_path else ""
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Zielordner für den Export", start)
        if not directory:
            return
        model = os.path.basename(self.model_dir.rstrip("/")) if self.model_dir else ""
        csv_path = os.path.join(directory, "ml_detections.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["row_px", "col_px", "intensity", "status", "model"])
            for (row, col), value in zip(self.positions, self.intensity):
                writer.writerow([f"{row:.3f}", f"{col:.3f}", f"{value:.4g}", "predicted", model])
        if self.heat is not None:
            np.save(os.path.join(directory, "ml_heatmap.npy"), self.heat.astype(np.float32))
        self.result_label.setText(f"Geschrieben: {csv_path}")

    def confirm(self) -> None:
        """Hand the predicted positions back to the main interface."""

        if self.on_confirm is None or not len(self.positions):
            return
        self.on_confirm(self.positions.copy(), self.intensity.copy())
        self.close()

    # ------------------------------------------------------------------ helpers
    def _update_enabled(self) -> None:
        ready = self.net is not None and self.target is not None and self.second is not None
        self.predict_btn.setEnabled(ready)
        has = len(self.positions) > 0
        self.export_btn.setEnabled(has)
        self.confirm_btn.setEnabled(has and self.on_confirm is not None)

    def _warn(self, message: str) -> None:
        QtWidgets.QMessageBox.warning(self, "Vorhersage", message)


def write_handoff(path: str, positions: np.ndarray, intensity: np.ndarray) -> None:
    """The confirmed prediction for the main interface (written to a temporary name, then renamed)."""

    partial = path + ".partial.npz"
    np.savez(partial, positions=np.asarray(positions, dtype=np.float64),
             intensity=np.asarray(intensity, dtype=np.float64))
    os.replace(partial, path)


def launch_predictor(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="temsim_ml.predict",
                                     description="Eigenständige Atom-Vorhersage mit einem trainierten Detektor")
    parser.add_argument("--model", help="Modellordner (detector.pt + meta.json)")
    parser.add_argument("--target", help="target-Bild (.npy/.tif)")
    parser.add_argument("--i-target", dest="i_target", help="i_target-Bild (.npy/.tif)")
    parser.add_argument("--handoff", help="Datei (.npz), in die 'Übernehmen' die Positionen für die Haupt-GUI schreibt")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    on_confirm = (lambda p, i: write_handoff(args.handoff, p, i)) if args.handoff else None
    window = PredictorWindow(args.device, on_confirm=on_confirm)
    if args.model and det.is_detector(args.model):
        window._load_model(args.model)
    if args.target and os.path.exists(args.target):
        window.target = np.asarray(read_image(args.target), dtype=np.float32)
        window.target_path = args.target
        window.target_label.setText(f"target: {os.path.basename(args.target)}  {window.target.shape}")
    if args.i_target and os.path.exists(args.i_target):
        window.second = np.asarray(read_image(args.i_target), dtype=np.float32)
        window.second_label.setText(f"i_target: {os.path.basename(args.i_target)}  {window.second.shape}")
    window._update_enabled()
    window._draw()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(launch_predictor())
