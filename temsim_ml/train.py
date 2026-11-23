r"""Part V, Section 13: a small interface that trains a detector on one run.

The open reconstruction is labelled data (Part II, Section 8): its atom positions
and intensities teach a U-Net what an atom looks like in this kind of image.  This
window trains such a model on **one run only**, shows a progress bar and the live
validation score, saves the result as a *local* model, and -- once it looks good
-- copies it to the global directory with one button.

It opens the run in a session of its own (a local core, ``python -m temsim_ml.train
--run DIR``), so training uses this machine's device.  Stopping saves what has been
learnt so far.

Implementation
==============
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import List, Optional

from PySide6 import QtCore, QtWidgets

from temsim.learning import detection as det
from temsim.service.session import Session
from temsim_gui.widgets import double_spin, int_spin

from .models import GLOBAL_MODELS, LOCAL_MODELS, promote

CHUNK = 25                         # steps between progress updates


def _device(name: str) -> str:
    if name == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


class TrainWorker(QtCore.QThread):
    """Opens the run, trains, and saves -- off the UI thread."""

    progress = QtCore.Signal(int, int, object, object)     # step, total, val_loss, val_f1
    done = QtCore.Signal(bool, str, str)                   # ok, message, model path

    def __init__(self, run: str, name: str, local_dir: str, steps: int, sigma_A: float,
                 holdout: float, device: str, **config):
        super().__init__()
        self.run_dir, self.name, self.local_dir = run, name, local_dir
        self.steps, self.sigma_A, self.holdout, self.device = steps, sigma_A, holdout, device
        self.config = config                               # further DetectorConfig fields (base, depth, ...)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:                                 # noqa: D401 - QThread entry point
        try:
            session = Session(self.device)
            session.set_workspace(self.local_dir)          # only so the name check has a root
            session.open_run(self.run_dir)
            trainer = session.detection.start_training(
                self.name, steps=self.steps, sigma_A=self.sigma_A, holdout=self.holdout,
                **{"val_every": max(CHUNK, self.steps // 10), "seed": 0, **self.config})
            total = self.steps
            while trainer.steps < total and not self._stop:
                trainer.run(min(CHUNK, total - trainer.steps))
                self.progress.emit(trainer.steps, total, trainer.val_loss, trainer.val_f1)
            trainer.validate()
            meta = dict(session.detection.meta)
            meta.update({"steps": trainer.steps, "val_loss": trainer.val_loss, "val_f1": trainer.val_f1,
                         "loss": trainer.loss, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "history": trainer.history[-200:],
                         "sources": [{"run": self.run_dir, "steps": trainer.steps, "val_f1": trainer.val_f1}]})
            path = det.save_detector(os.path.join(self.local_dir, self.name), trainer.net, meta)
            stopped = " (gestoppt)" if self._stop else ""
            f1 = trainer.val_f1
            self.done.emit(True, f"Fertig{stopped}: {trainer.steps} Schritte, "
                                 f"F1 {f1:.3f}" if f1 is not None else f"Fertig{stopped}: {trainer.steps} Schritte", path)
        except Exception as error:                         # noqa: BLE001 - shown to the user
            self.done.emit(False, f"{type(error).__name__}: {error}", "")
            traceback.print_exc()


class TrainerWindow(QtWidgets.QMainWindow):
    """Train a detector on one run, with a progress bar; then make it global."""

    def __init__(self, run: str = "", device: str = "auto"):
        super().__init__()
        self.setWindowTitle("TEM-Detektor: Training")
        self.resize(560, 420)
        self.device_name = device
        self.worker: Optional[TrainWorker] = None
        self.model_path = ""
        self._build()
        if run:
            self.run_edit.setText(run)
            self._suggest_name(run)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        form = QtWidgets.QFormLayout()
        outer.addLayout(form)

        self.run_edit = QtWidgets.QLineEdit()
        self.run_edit.setPlaceholderText("Pfad des Laufs (Ordner mit experiment/state, target, i_target)")
        self.run_edit.textChanged.connect(lambda _=None: self._update_enabled())
        browse_run = QtWidgets.QPushButton("…")
        browse_run.setFixedWidth(30)
        browse_run.clicked.connect(self.choose_run)
        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(self.run_edit, 1)
        run_row.addWidget(browse_run)
        form.addRow("Lauf", run_row)

        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Modellname (Buchstaben, Ziffern, _ . -)")
        self.name_edit.textChanged.connect(lambda _=None: self._update_enabled())
        form.addRow("Name", self.name_edit)

        self.local_edit = QtWidgets.QLineEdit(LOCAL_MODELS)
        self.local_edit.setToolTip("Lokaler Ordner, in dem das neue Modell gespeichert wird.")
        form.addRow("Lokaler Ordner", self.local_edit)

        self.steps = int_spin(3000, 50, 1_000_000, 100, tip="Trainingsschritte.")
        self.sigma = double_spin(0.3, 0.05, 2.0, 0.05, 2, suffix=" A",
                                 tip="Breite des Gauss-Flecks je Atom im Label.")
        self.holdout = int_spin(15, 0, 50, 5, tip="Unterer Anteil der Zeilen, der zum Validieren zurückgehalten wird [%].")
        form.addRow("Schritte", self.steps)
        form.addRow("Sigma", self.sigma)
        form.addRow("Holdout [%]", self.holdout)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFormat("%v / %m Schritte")
        outer.addWidget(self.progress)
        self.status = QtWidgets.QLabel("bereit")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        buttons = QtWidgets.QHBoxLayout()
        self.train_btn = QtWidgets.QPushButton("Training starten")
        self.train_btn.clicked.connect(self.start)
        self.stop_btn = QtWidgets.QPushButton("Stopp")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        self.global_btn = QtWidgets.QPushButton("Als global setzen")
        self.global_btn.setToolTip(f"Das trainierte Modell nach {GLOBAL_MODELS} kopieren (dann überall nutzbar).")
        self.global_btn.clicked.connect(self.make_global)
        self.global_btn.setEnabled(False)
        buttons.addWidget(self.train_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addWidget(self.global_btn)
        outer.addLayout(buttons)
        outer.addStretch(1)
        self._update_enabled()

    # ------------------------------------------------------------------ actions
    def choose_run(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Lauf wählen")
        if directory:
            self.run_edit.setText(directory)
            self._suggest_name(directory)

    def _suggest_name(self, run: str) -> None:
        if not self.name_edit.text():
            base = os.path.basename(run.rstrip("/")) or "modell"
            self.name_edit.setText("".join(c if (c.isalnum() or c in "_.-") else "_" for c in base)[:40])

    def start(self) -> None:
        run, name = self.run_edit.text().strip(), self.name_edit.text().strip()
        if not os.path.isdir(run):
            self._warn("Der Lauf-Ordner existiert nicht.")
            return
        # the model name is checked against the allowed pattern by the core (start_training); an
        # invalid name comes back as a failed run in on_done.
        self.worker = TrainWorker(run, name, self.local_edit.text().strip() or LOCAL_MODELS,
                                  int(self.steps.value()), float(self.sigma.value()),
                                  int(self.holdout.value()) / 100.0, _device(self.device_name))
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_done)
        self.progress.setRange(0, int(self.steps.value()))
        self.progress.setValue(0)
        self.status.setText("öffnet den Lauf und trainiert …")
        self._set_running(True)
        self.worker.start()

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.status.setText("stoppt … (speichert den aktuellen Stand)")

    def on_progress(self, step: int, total: int, val_loss, val_f1) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(step)
        extra = ""
        if val_f1 is not None:
            extra = f" — val-F1 {val_f1:.3f}, val-Loss {val_loss:.4f}"
        self.status.setText(f"Schritt {step} / {total}{extra}")

    def on_done(self, ok: bool, message: str, path: str) -> None:
        self._set_running(False)
        self.status.setText(message)
        if ok:
            self.progress.setValue(self.progress.maximum())
            self.model_path = path
            self.global_btn.setEnabled(True)

    def make_global(self) -> None:
        if not self.model_path:
            return
        try:
            destination = promote(self.model_path)
        except Exception as error:                           # noqa: BLE001 - shown to the user
            self._warn(f"Konnte nicht global setzen: {error}")
            return
        self.status.setText(f"Global gesetzt: {destination}")
        self.global_btn.setEnabled(False)

    # ------------------------------------------------------------------ helpers
    def _set_running(self, running: bool) -> None:
        self.train_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for w in (self.run_edit, self.name_edit, self.local_edit, self.steps, self.sigma, self.holdout):
            w.setEnabled(not running)
        if running:
            self.global_btn.setEnabled(False)

    def _update_enabled(self) -> None:
        running = self.worker is not None and self.worker.isRunning()
        ready = bool(self.run_edit.text().strip()) and bool(self.name_edit.text().strip())
        self.train_btn.setEnabled(ready and not running)

    def _warn(self, message: str) -> None:
        QtWidgets.QMessageBox.warning(self, "Training", message)

    def closeEvent(self, event) -> None:                     # noqa: N802 - Qt name
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        super().closeEvent(event)


def launch_trainer(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="temsim_ml.train",
                                     description="Detektor auf einem Lauf trainieren")
    parser.add_argument("--run", default="", help="Lauf-Ordner")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = TrainerWindow(args.run, args.device)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(launch_trainer())
