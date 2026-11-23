r"""Part V, Section 13: a small interface that trains a detector.

A reconstruction is labelled data (Part II, Section 8): its atom positions and
intensities teach a U-Net what an atom looks like in this kind of image.  This
window trains on

* **one run** -- the run is opened in a session of its own (a local core), or
* **the collection of good reconstructions** (Part II, Section 8.4) -- the entries
  of a group, each one ticked on or off; entries and groups are managed here too,

either as a **new model** or by **continuing a global model** (its architecture,
label width and intensity scale stay).  A progress bar and the live validation
score show the training; stopping saves what has been learnt so far.  The result
is a *local* model (``~/data/models_local``); "Als global setzen" copies it to the
global directory.

Start it with ``python -m temsim_ml.train [--run DIR]``; training uses this
machine's device.

Implementation
==============
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

from temsim.learning import detection as det
from temsim.service.detection import NAME_PATTERN
from temsim.service.session import Session
from temsim_gui.widgets import double_spin, int_spin

from .models import GLOBAL_MODELS, LOCAL_MODELS, list_models, promote

CHUNK = 25                         # steps between progress updates
ALL = "Alle Einträge"


def _device(name: str) -> str:
    if name == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def run_snapshot(run: str, device: str) -> Dict[str, Any]:
    """Open a run in a session of its own and take its training data (needs ``i_target``)."""

    session = Session(device)
    session.open_run(run)
    snapshot = session.detection.snapshot()
    snapshot["name"] = os.path.basename(os.path.normpath(run))
    return snapshot


def model_sigma_A(meta: Dict[str, Any]) -> Optional[float]:
    """The label width of a saved model in Angstrom (older models only stored it in pixels)."""

    if meta.get("sigma_A"):
        return float(meta["sigma_A"])
    if meta.get("sigma_px") and meta.get("pixel_A"):
        return float(meta["sigma_px"]) * float(meta["pixel_A"])
    return None


class TrainWorker(QtCore.QThread):
    """Gathers the samples, trains, and saves -- off the UI thread."""

    progress = QtCore.Signal(int, int, object, object)     # step, total, val_loss, val_f1
    done = QtCore.Signal(bool, str, str)                   # ok, message, model path

    def __init__(self, run: str, name: str, local_dir: str, steps: int, sigma_A: float, holdout: float,
                 device: str, entries: Optional[List[str]] = None, collection: Optional[str] = None,
                 resume: str = "", **config):
        super().__init__()
        self.run_dir, self.name, self.local_dir = run, name, local_dir
        self.steps, self.sigma_A, self.holdout, self.device = steps, sigma_A, holdout, device
        self.entries, self.collection_root, self.resume = entries, collection, resume
        self.config = config                               # further DetectorConfig fields (base, depth, ...)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _snapshots(self) -> List[Dict[str, Any]]:
        if self.entries is not None:
            collection = det.Collection(self.collection_root)
            return [collection.load(entry) for entry in self.entries]
        return [run_snapshot(self.run_dir, self.device)]

    def run(self) -> None:                                 # noqa: D401 - QThread entry point
        try:
            if not NAME_PATTERN.match(self.name or ""):
                raise ValueError(f"ungültiger Modellname {self.name!r} (Buchstaben, Ziffern, _ . -)")
            snapshots = self._snapshots()
            if not snapshots:
                raise ValueError("keine Trainingsdaten gewählt")
            net, meta = det.load_detector(self.resume, self.device) if self.resume else (None, {})
            scale = float(meta.get("scale") or np.median(np.concatenate([s["intensity"] for s in snapshots])) or 1.0)
            sigma_A = model_sigma_A(meta) or float(self.sigma_A)
            samples = [det.snapshot_sample(s, scale, sigma_A, self.holdout) for s in snapshots]
            config = {"val_every": max(CHUNK, self.steps // 10), "seed": 0, **self.config, "steps": self.steps}
            if net is not None:
                config.update(base=net.config["base"], depth=net.config["depth"])
            trainer = det.DetectorTrainer(samples, det.DetectorConfig(**config), net, self.device)
            total = self.steps
            while trainer.steps < total and not self._stop:
                trainer.run(min(CHUNK, total - trainer.steps))
                self.progress.emit(trainer.steps, total, trainer.val_loss, trainer.val_f1)
            trainer.validate()
            pixel_A = float(np.median([s["pixel_A"] for s in snapshots]))
            data = [{"name": s.get("name", ""), "source": s.get("source", ""), "entry": s.get("id")} for s in snapshots]
            meta = {**meta, "scale": scale, "sigma_A": sigma_A, "sigma_px": sigma_A / pixel_A, "pixel_A": pixel_A,
                    "channels": ["target", "i_target"], "base": trainer.net.config["base"],
                    "depth": trainer.net.config["depth"], "steps": int(meta.get("steps", 0)) + trainer.steps,
                    "val_loss": trainer.val_loss, "val_f1": trainer.val_f1, "loss": trainer.loss,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"), "history": trainer.history[-200:],
                    "sources": list(meta.get("sources", [])) + [{"data": data, "steps": trainer.steps,
                                                                 "val_loss": trainer.val_loss,
                                                                 "val_f1": trainer.val_f1}]}
            path = det.save_detector(os.path.join(self.local_dir, self.name), trainer.net, meta)
            stopped = " (gestoppt)" if self._stop else ""
            score = f", F1 {trainer.val_f1:.3f}" if trainer.val_f1 is not None else ""
            self.done.emit(True, f"Fertig{stopped}: {trainer.steps} Schritte auf {len(samples)} Bild(ern){score}", path)
        except Exception as error:                         # noqa: BLE001 - shown to the user
            self.done.emit(False, f"{type(error).__name__}: {error}", "")
            traceback.print_exc()


class TrainerWindow(QtWidgets.QMainWindow):
    """Train a detector on one run or on chosen entries of the collection; then make it global."""

    def __init__(self, run: str = "", device: str = "auto", collection: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("TEM-Detektor: Training")
        self.resize(760, 760)
        self.device_name = device
        self.collection = det.Collection(collection)
        self.worker: Optional[TrainWorker] = None
        self.model_path = ""
        self._checked: Dict[str, bool] = {}               # entry id -> ticked (kept across group filters)
        self._build()
        self.refresh_collection()
        self.refresh_models()
        if run:
            self.run_edit.setText(run)
            self._suggest_name(os.path.basename(os.path.normpath(run)))

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        # --- training data
        self.data_box = QtWidgets.QGroupBox("Trainingsdaten")
        data = QtWidgets.QVBoxLayout(self.data_box)
        outer.addWidget(self.data_box, 1)
        self.use_run = QtWidgets.QRadioButton("Nur dieser Lauf")
        self.use_collection = QtWidgets.QRadioButton("Sammlung guter Optimierungen")
        self.use_run.setChecked(True)
        self.use_run.toggled.connect(lambda _=None: self._source_changed())
        data.addWidget(self.use_run)

        self.run_edit = QtWidgets.QLineEdit()
        self.run_edit.setPlaceholderText("Pfad des Laufs (Ordner mit experiment/state, target, i_target)")
        self.run_edit.textChanged.connect(lambda _=None: self._update_enabled())
        browse_run = QtWidgets.QPushButton("…")
        browse_run.setFixedWidth(30)
        browse_run.clicked.connect(self.choose_run)
        self.run_row = QtWidgets.QWidget()
        run_layout = QtWidgets.QHBoxLayout(self.run_row)
        run_layout.setContentsMargins(20, 0, 0, 0)
        run_layout.addWidget(self.run_edit, 1)
        run_layout.addWidget(browse_run)
        data.addWidget(self.run_row)
        data.addWidget(self.use_collection)

        self.collection_box = QtWidgets.QWidget()
        coll = QtWidgets.QVBoxLayout(self.collection_box)
        coll.setContentsMargins(20, 0, 0, 0)
        data.addWidget(self.collection_box, 1)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Gruppe"))
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.setToolTip("Welche Einträge gezeigt werden. Trainiert wird auf den gezeigten, angehakten.")
        self.filter_combo.currentIndexChanged.connect(lambda _=None: self._fill_entries())
        filter_row.addWidget(self.filter_combo, 1)
        new_group = QtWidgets.QPushButton("Neue Gruppe …")
        new_group.clicked.connect(self.new_group)
        self.delete_group_btn = QtWidgets.QPushButton("Gruppe löschen")
        self.delete_group_btn.setToolTip("Die gezeigte Gruppe auflösen (die Einträge bleiben erhalten).")
        self.delete_group_btn.clicked.connect(self.delete_group)
        filter_row.addWidget(new_group)
        filter_row.addWidget(self.delete_group_btn)
        coll.addLayout(filter_row)

        self.entry_list = QtWidgets.QListWidget()
        self.entry_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.entry_list.setToolTip("Häkchen: wird trainiert. Markieren (anklicken, Strg/Shift): für die Knöpfe unten.")
        self.entry_list.itemChanged.connect(self._entry_changed)
        coll.addWidget(self.entry_list, 1)
        self.selection_label = QtWidgets.QLabel("")
        coll.addWidget(self.selection_label)

        tick_row = QtWidgets.QHBoxLayout()
        for label, state in (("Alle an", True), ("Alle aus", False)):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda _=None, s=state: self._tick_all(s))
            tick_row.addWidget(button)
        delete_entry = QtWidgets.QPushButton("Markierte löschen")
        delete_entry.clicked.connect(self.delete_entries)
        tick_row.addStretch(1)
        tick_row.addWidget(delete_entry)
        coll.addLayout(tick_row)

        tag_row = QtWidgets.QHBoxLayout()
        tag_row.addWidget(QtWidgets.QLabel("Markierte"))
        self.tag_combo = QtWidgets.QComboBox()
        self.tag_combo.setEditable(True)
        self.tag_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.tag_combo.setToolTip("Gruppe für 'in Gruppe' / 'aus Gruppe' und für neue Einträge; ein neuer Name legt sie an.")
        add_tag = QtWidgets.QPushButton("in Gruppe")
        add_tag.clicked.connect(lambda: self.tag_selected(True))
        remove_tag = QtWidgets.QPushButton("aus Gruppe")
        remove_tag.clicked.connect(lambda: self.tag_selected(False))
        tag_row.addWidget(add_tag)
        tag_row.addWidget(remove_tag)
        tag_row.addWidget(self.tag_combo, 1)
        coll.addLayout(tag_row)
        self.add_run_btn = QtWidgets.QPushButton("Lauf (oben) als Eintrag aufnehmen")
        self.add_run_btn.setToolTip("Den jetzigen Stand des Laufs als Schnappschuss in die Sammlung legen, in die Gruppe\n"
                                    "rechts neben 'aus Gruppe'.")
        self.add_run_btn.clicked.connect(self.add_run)
        coll.addWidget(self.add_run_btn)

        # --- model
        self.model_box = QtWidgets.QGroupBox("Modell")
        form = QtWidgets.QFormLayout(self.model_box)
        outer.addWidget(self.model_box)
        self.new_model = QtWidgets.QRadioButton("Neues Modell")
        self.resume_model = QtWidgets.QRadioButton("Globales Modell weitertrainieren")
        self.new_model.setChecked(True)
        self.resume_model.toggled.connect(lambda _=None: self._mode_changed())
        self.global_combo = QtWidgets.QComboBox()
        self.global_combo.setToolTip(f"Modelle in {GLOBAL_MODELS}. Architektur, Fleckbreite und Intensitätsskala\n"
                                     "bleiben die des Modells.")
        self.global_combo.currentIndexChanged.connect(lambda _=None: self._mode_changed())
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(self.new_model)
        mode_row.addWidget(self.resume_model)
        mode_row.addWidget(self.global_combo, 1)
        form.addRow(mode_row)

        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Modellname (Buchstaben, Ziffern, _ . -)")
        self.name_edit.textChanged.connect(lambda _=None: self._update_enabled())
        form.addRow("Name", self.name_edit)
        self.local_edit = QtWidgets.QLineEdit(LOCAL_MODELS)
        self.local_edit.setToolTip("Lokaler Ordner, in dem das Ergebnis gespeichert wird.")
        form.addRow("Lokaler Ordner", self.local_edit)
        self.steps = int_spin(3000, 50, 1_000_000, 100, tip="Trainingsschritte.")
        self.sigma = double_spin(0.3, 0.05, 2.0, 0.05, 2, suffix=" A",
                                 tip="Breite des Gauss-Flecks je Atom im Label (beim Weitertrainieren die des Modells).")
        self.holdout = int_spin(15, 0, 50, 5, tip="Unterer Anteil der Zeilen jedes Bildes, der zum Validieren\n"
                                                  "zurückgehalten wird [%].")
        form.addRow("Schritte", self.steps)
        form.addRow("Sigma", self.sigma)
        form.addRow("Holdout [%]", self.holdout)

        # --- run
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
        self._source_changed()
        self._mode_changed()

    # ------------------------------------------------------------------ collection
    def refresh_collection(self) -> None:
        groups = self.collection.groups()
        shown = self.filter_combo.currentData()
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem(ALL, None)
        for group in groups:
            self.filter_combo.addItem(group, group)
        self.filter_combo.setCurrentIndex(max(self.filter_combo.findData(shown), 0))
        self.filter_combo.blockSignals(False)
        text = self.tag_combo.currentText()
        self.tag_combo.clear()
        self.tag_combo.addItems(groups)
        self.tag_combo.setEditText(text)
        self._fill_entries()

    def _fill_entries(self) -> None:
        group = self.filter_combo.currentData()
        self.delete_group_btn.setEnabled(group is not None)
        self.entry_list.blockSignals(True)
        self.entry_list.clear()
        for entry in self.collection.entries(group):
            groups = ", ".join(entry.get("groups", [])) or "ohne Gruppe"
            item = QtWidgets.QListWidgetItem(
                f"{entry.get('name')}  —  {entry.get('atoms', '?')} Atome, {entry.get('pixel_A', 0):.3g} A/px  "
                f"[{groups}]  {entry.get('created', '')}")
            item.setData(QtCore.Qt.UserRole, entry["id"])
            item.setData(QtCore.Qt.UserRole + 1, int(entry.get("atoms", 0)))
            item.setToolTip(entry.get("source", ""))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if self._checked.get(entry["id"], True) else QtCore.Qt.Unchecked)
            self.entry_list.addItem(item)
        self.entry_list.blockSignals(False)
        self._update_selection()

    def _items(self) -> List[QtWidgets.QListWidgetItem]:
        return [self.entry_list.item(k) for k in range(self.entry_list.count())]

    def chosen_entries(self) -> List[str]:
        """The shown entries that are ticked: the training data of the collection."""

        return [i.data(QtCore.Qt.UserRole) for i in self._items() if i.checkState() == QtCore.Qt.Checked]

    def _entry_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        self._checked[item.data(QtCore.Qt.UserRole)] = item.checkState() == QtCore.Qt.Checked
        self._update_selection()

    def _tick_all(self, state: bool) -> None:
        for item in self._items():
            item.setCheckState(QtCore.Qt.Checked if state else QtCore.Qt.Unchecked)

    def _update_selection(self) -> None:
        ticked = [i for i in self._items() if i.checkState() == QtCore.Qt.Checked]
        atoms = sum(int(i.data(QtCore.Qt.UserRole + 1)) for i in ticked)
        self.selection_label.setText(f"{len(ticked)} von {self.entry_list.count()} Einträgen gewählt, {atoms} Atome")
        self._update_enabled()

    def _marked(self) -> List[str]:
        ids = [i.data(QtCore.Qt.UserRole) for i in self.entry_list.selectedItems()]
        if not ids:
            self._warn("Erst Einträge markieren (anklicken, mit Strg/Shift mehrere).")
        return ids

    def new_group(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Neue Gruppe", "Name der Gruppe (z.B. Optimiert, Handverlesen):")
        if ok and name.strip():
            self.collection.create_group(name.strip())
            self.refresh_collection()
            self.filter_combo.setCurrentIndex(self.filter_combo.findData(name.strip()))

    def delete_group(self) -> None:
        group = self.filter_combo.currentData()
        if group is None:
            return
        answer = QtWidgets.QMessageBox.question(self, "Gruppe löschen",
                                                f"Gruppe '{group}' auflösen? Die Einträge bleiben erhalten.")
        if answer == QtWidgets.QMessageBox.Yes:
            self.collection.delete_group(group)
            self.filter_combo.setCurrentIndex(0)
            self.refresh_collection()

    def tag_selected(self, member: bool) -> None:
        group = self.tag_combo.currentText().strip()
        if not group:
            self._warn("Erst rechts eine Gruppe wählen oder eintippen.")
            return
        ids = self._marked()
        if ids:
            self.collection.tag(ids, group, member)
            self.refresh_collection()

    def delete_entries(self) -> None:
        ids = self._marked()
        if not ids:
            return
        answer = QtWidgets.QMessageBox.question(self, "Einträge löschen",
                                                f"{len(ids)} Eintrag/Einträge endgültig aus der Sammlung löschen?")
        if answer == QtWidgets.QMessageBox.Yes:
            for entry in ids:
                self.collection.remove(entry)
            self.refresh_collection()

    def add_run(self) -> None:
        run = self.run_edit.text().strip()
        if not os.path.isdir(run):
            self._warn("Oben erst einen Lauf wählen.")
            return
        group = self.tag_combo.currentText().strip()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            snapshot = run_snapshot(run, _device(self.device_name))
            self.collection.add(snapshot, snapshot["name"], [group] if group else [])
        except Exception as error:                         # noqa: BLE001 - shown to the user
            QtWidgets.QApplication.restoreOverrideCursor()
            self._warn(f"Nicht aufgenommen: {error}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.status.setText(f"Aufgenommen: {snapshot['name']} ({len(snapshot['positions'])} Atome)"
                            + (f" in Gruppe '{group}'" if group else ""))
        self.refresh_collection()

    # ------------------------------------------------------------------ models
    def refresh_models(self) -> None:
        current = self.global_combo.currentData()
        self.global_combo.blockSignals(True)
        self.global_combo.clear()
        for entry in list_models(GLOBAL_MODELS):
            f1 = entry["meta"].get("val_f1")
            label = entry["name"] + (f"  (F1 {f1:.2f})" if isinstance(f1, (int, float)) else "")
            self.global_combo.addItem(label, entry["path"])
        self.global_combo.setCurrentIndex(max(self.global_combo.findData(current), 0))
        self.global_combo.blockSignals(False)
        self.resume_model.setEnabled(self.global_combo.count() > 0)
        self._mode_changed()

    def _mode_changed(self) -> None:
        resume = self.resume_model.isChecked() and self.global_combo.count() > 0
        self.global_combo.setEnabled(resume)
        self.sigma.setEnabled(not resume)
        if resume:                                         # same name: "Als global setzen" replaces it
            self.name_edit.setText(os.path.basename(self.global_combo.currentData().rstrip("/")))
        self._update_enabled()

    # ------------------------------------------------------------------ training
    def choose_run(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Lauf wählen")
        if directory:
            self.run_edit.setText(directory)
            self._suggest_name(os.path.basename(os.path.normpath(directory)))

    def _suggest_name(self, base: str) -> None:
        if not self.name_edit.text():
            self.name_edit.setText("".join(c if (c.isalnum() or c in "_.-") else "_" for c in base)[:40] or "modell")

    def _source_changed(self) -> None:
        collection = self.use_collection.isChecked()
        self.run_edit.setEnabled(not collection)
        self.collection_box.setEnabled(collection)
        self.add_run_btn.setEnabled(collection)
        if collection:
            self._suggest_name("sammlung_" + (self.filter_combo.currentData() or "alle"))
        self._update_enabled()

    def start(self) -> None:
        run, name = self.run_edit.text().strip(), self.name_edit.text().strip()
        entries = None
        if self.use_collection.isChecked():
            entries = self.chosen_entries()
            if not entries:
                self._warn("Keine Einträge angehakt.")
                return
        elif not os.path.isdir(run):
            self._warn("Der Lauf-Ordner existiert nicht.")
            return
        resume = self.global_combo.currentData() if self.resume_model.isChecked() else ""
        self.worker = TrainWorker(run, name, self.local_edit.text().strip() or LOCAL_MODELS,
                                  int(self.steps.value()), float(self.sigma.value()),
                                  int(self.holdout.value()) / 100.0, _device(self.device_name),
                                  entries=entries, collection=self.collection.root, resume=resume or "")
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_done)
        self.progress.setRange(0, int(self.steps.value()))
        self.progress.setValue(0)
        what = f"{len(entries)} Einträge der Sammlung" if entries is not None else "den Lauf"
        self.status.setText(f"lädt {what} und trainiert …")
        self._set_running(True)
        self.worker.start()

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.status.setText("stoppt … (speichert den aktuellen Stand)")

    def on_progress(self, step: int, total: int, val_loss, val_f1) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(step)
        extra = f" — val-F1 {val_f1:.3f}, val-Loss {val_loss:.4f}" if val_f1 is not None else ""
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
        name = os.path.basename(self.model_path.rstrip("/"))
        if os.path.exists(os.path.join(GLOBAL_MODELS, name)):
            answer = QtWidgets.QMessageBox.question(self, "Als global setzen",
                                                    f"Das globale Modell '{name}' ersetzen?")
            if answer != QtWidgets.QMessageBox.Yes:
                return
        try:
            destination = promote(self.model_path)
        except Exception as error:                         # noqa: BLE001 - shown to the user
            self._warn(f"Konnte nicht global setzen: {error}")
            return
        self.status.setText(f"Global gesetzt: {destination}")
        self.global_btn.setEnabled(False)
        self.refresh_models()

    # ------------------------------------------------------------------ helpers
    def _set_running(self, running: bool) -> None:
        self.stop_btn.setEnabled(running)
        for box in (self.data_box, self.model_box):
            box.setEnabled(not running)
        if running:
            self.global_btn.setEnabled(False)
        self._update_enabled()

    def _update_enabled(self) -> None:
        if not hasattr(self, "train_btn"):
            return
        running = self.worker is not None and self.worker.isRunning()
        if self.use_collection.isChecked():
            ready = bool(self.chosen_entries())
        else:
            ready = bool(self.run_edit.text().strip())
        self.train_btn.setEnabled(ready and bool(self.name_edit.text().strip()) and not running)

    def _warn(self, message: str) -> None:
        QtWidgets.QMessageBox.warning(self, "Training", message)

    def closeEvent(self, event) -> None:                   # noqa: N802 - Qt name
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        super().closeEvent(event)


def launch_trainer(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="temsim_ml.train", description="Detektor trainieren (Lauf oder Sammlung)")
    parser.add_argument("--run", default="", help="Lauf-Ordner")
    parser.add_argument("--collection", help="Ordner der Sammlung (Standard ~/data/ml_collection)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = TrainerWindow(args.run, args.device, args.collection)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(launch_trainer())
