r"""Part V, Section 14: the window of the batch tool.

Three parts:

* **Eingaben** -- data folders or file pairs (target + i_target, optional mask);
  either image may be a stack.  Per input: name, pixel size, voltage (empty: from
  ``init.json`` or the template) and, for stacks, what a frame takes over from the
  previous one.  The **template** (an ``init.json`` and default values) holds for
  inputs without their own file.
* **Pipeline** -- the steps in order; the parameters of the selected step are
  edited below the list.  Pipelines can be saved as named templates
  (``~/data/batch_pipelines``).  Default: ML prediction with the global model,
  then training of all aberrations, positions and intensities.
* **Auftrag** -- the output folder, the device, start and stop, progress and log.

"Starten" writes ``job.json`` into the output folder and starts the runner as a
process of its own (:func:`temsim_ml.batch.runner.launch`); the window may be
closed, the job goes on.  "Auftrag öffnen" shows a job again (also one that has
finished or was stopped; starting it again continues with the frames not yet done).

Implementation
==============
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Any, Dict, List, Optional

from PySide6 import QtCore, QtWidgets

from temsim_gui.widgets import double_spin, int_spin

from . import job as jobs
from . import runner

COLUMNS = ["Name", "Art", "Pixel (nm)", "Spannung (kV)", "Start aus Vorgänger", "Quelle"]
MODES = {"": "aus init.json", "stem_haadf": "STEM (HAADF)", "tem": "TEM"}


class FilesDialog(QtWidgets.QDialog):
    """target, i_target and an optional mask."""

    def __init__(self, parent=None, start: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Bilder hinzufügen")
        self.resize(640, 160)
        self.start = start
        form = QtWidgets.QFormLayout(self)
        self.edits = {}
        for key, label in (("target", "target (Bild oder Stack)"), ("i_target", "i_target"),
                           ("mask", "Maske (optional)")):
            edit = QtWidgets.QLineEdit()
            button = QtWidgets.QPushButton("…")
            button.setFixedWidth(30)
            button.clicked.connect(lambda _=None, k=key: self._choose(k))
            row = QtWidgets.QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(button)
            form.addRow(label, row)
            self.edits[key] = edit
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _choose(self, key: str) -> None:
        current = self.edits["target"].text() or self.start
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, key, os.path.dirname(current) if current else "",
                                                        "Bilder (*.tif *.tiff *.npy)")
        if path:
            self.edits[key].setText(path)
            if key == "target" and not self.edits["i_target"].text():
                folder = os.path.dirname(path)
                for name in jobs.I_TARGET_NAMES:
                    if os.path.exists(os.path.join(folder, name)):
                        self.edits["i_target"].setText(os.path.join(folder, name))
                        break

    def values(self):
        return tuple(self.edits[k].text().strip() for k in ("target", "i_target", "mask"))


class BatchWindow(QtWidgets.QMainWindow):
    def __init__(self, job_dir: str = ""):
        super().__init__()
        self.setWindowTitle("TEM: Automatik (Stapelverarbeitung)")
        self.resize(1250, 860)
        self.inputs: List[Dict[str, Any]] = []
        self.pipeline: List[Dict[str, Any]] = jobs.default_pipeline()
        self.job_dir = ""
        self._log_size = -1
        self._filling = False
        self._build()
        self.refresh_pipeline()
        self.refresh_presets()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(2000)
        if job_dir:
            self.open_job(job_dir)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(split, 3)

        # --- inputs and template
        self.input_box = QtWidgets.QGroupBox("Eingaben")
        left = QtWidgets.QVBoxLayout(self.input_box)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(self._cell_changed)
        left.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        for text, slot, tip in (("Ordner …", self.add_folders, "Datenordner (target, i_target, Maske, init.json)"),
                                ("Dateien …", self.add_files, "target und i_target einzeln, Stacks erlaubt"),
                                ("Entfernen", self.remove_inputs, "")):
            button = QtWidgets.QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(lambda _=None, s=slot: s())
            row.addWidget(button)
        row.addStretch(1)
        self.count_label = QtWidgets.QLabel("")
        row.addWidget(self.count_label)
        left.addLayout(row)

        template = QtWidgets.QGroupBox("Vorlage (für Eingaben ohne eigene init.json; leere Felder: aus der Datei)")
        form = QtWidgets.QFormLayout(template)
        self.init_edit = QtWidgets.QLineEdit()
        self.init_edit.setPlaceholderText("init.json (optional)")
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self.choose_init)
        line = QtWidgets.QHBoxLayout()
        line.addWidget(self.init_edit, 1)
        line.addWidget(browse)
        form.addRow("init.json", line)
        self.pixel_spin = double_spin(0.0, 0.0, 1.0, 0.0001, 5, " nm", tip="0: aus init.json")
        self.voltage_spin = double_spin(80.0, 0.0, 1000.0, 10.0, 1, " kV", tip="0: aus init.json")
        self.mode_combo = QtWidgets.QComboBox()
        for key, label in MODES.items():
            self.mode_combo.addItem(label, key)
        self.mode_combo.setCurrentIndex(self.mode_combo.findData("stem_haadf"))
        form.addRow("Pixelgröße", self.pixel_spin)
        form.addRow("Spannung", self.voltage_spin)
        form.addRow("Modus", self.mode_combo)
        left.addWidget(template)
        split.addWidget(self.input_box)

        # --- pipeline
        self.pipe_box = QtWidgets.QGroupBox("Pipeline (pro Bild, von oben nach unten)")
        right = QtWidgets.QVBoxLayout(self.pipe_box)
        self.step_list = QtWidgets.QListWidget()
        self.step_list.currentRowChanged.connect(lambda _=None: self.show_params())
        self.step_list.setMaximumHeight(160)
        right.addWidget(self.step_list)
        row = QtWidgets.QHBoxLayout()
        self.kind_combo = QtWidgets.QComboBox()
        for kind, spec in jobs.STEP_TYPES.items():
            self.kind_combo.addItem(spec["label"], kind)
        row.addWidget(self.kind_combo, 1)
        for text, slot in (("Hinzufügen", self.add_step), ("Entfernen", self.remove_step),
                           ("↑", lambda: self.move_step(-1)), ("↓", lambda: self.move_step(1)),
                           ("Standard", self.default_steps)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda _=None, s=slot: s())
            row.addWidget(button)
        right.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Vorlage:"))
        self.preset_combo = QtWidgets.QComboBox()
        row.addWidget(self.preset_combo, 1)
        load = QtWidgets.QPushButton("Laden")
        load.clicked.connect(self.load_preset)
        save = QtWidgets.QPushButton("Speichern unter …")
        save.clicked.connect(self.save_preset)
        row.addWidget(load)
        row.addWidget(save)
        right.addLayout(row)
        self.param_box = QtWidgets.QGroupBox("Parameter")
        self.param_form = QtWidgets.QFormLayout(self.param_box)
        self.param_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.param_box)
        right.addWidget(scroll, 3)
        split.addWidget(self.pipe_box)
        split.setSizes([700, 550])

        # --- job
        box = QtWidgets.QGroupBox("Auftrag")
        lay = QtWidgets.QVBoxLayout(box)
        outer.addWidget(box, 2)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Ausgabeordner"))
        self.output_edit = QtWidgets.QLineEdit()
        row.addWidget(self.output_edit, 1)
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self.choose_output)
        row.addWidget(browse)
        row.addWidget(QtWidgets.QLabel("Gerät"))
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        row.addWidget(self.device_combo)
        self.start_btn = QtWidgets.QPushButton("Starten")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QtWidgets.QPushButton("Stoppen")
        self.stop_btn.clicked.connect(self.stop)
        self.open_btn = QtWidgets.QPushButton("Auftrag öffnen …")
        self.open_btn.clicked.connect(self.choose_job)
        for button in (self.start_btn, self.stop_btn, self.open_btn):
            row.addWidget(button)
        lay.addLayout(row)
        self.total_bar = QtWidgets.QProgressBar()
        self.total_bar.setFormat("%v / %m Bilder")
        self.step_bar = QtWidgets.QProgressBar()
        self.state_label = QtWidgets.QLabel("kein Auftrag")
        self.state_label.setWordWrap(True)
        lay.addWidget(self.total_bar)
        lay.addWidget(self.state_label)
        lay.addWidget(self.step_bar)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        lay.addWidget(self.log_view, 1)
        self.stop_btn.setEnabled(False)

    # ------------------------------------------------------------------ inputs
    def add_folders(self) -> None:
        dialog = QtWidgets.QFileDialog(self, "Datenordner wählen")
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        for view in dialog.findChildren((QtWidgets.QListView, QtWidgets.QTreeView)):
            view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        if dialog.exec():
            self.add_folder_inputs(dialog.selectedFiles())

    def add_folder_inputs(self, paths) -> None:
        problems = []
        for path in paths:
            try:
                self.add_input(jobs.folder_input(path))
            except (jobs.JobError, OSError, ValueError) as exc:
                problems.append(str(exc))
        if problems:
            QtWidgets.QMessageBox.warning(self, "Eingaben", "\n".join(problems))

    def add_files(self) -> None:
        dialog = FilesDialog(self, self.inputs[-1]["target"] if self.inputs else "")
        if not dialog.exec():
            return
        target, second, mask = dialog.values()
        try:
            self.add_input(jobs.file_input(target, second, mask))
        except (jobs.JobError, OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Eingaben", str(exc))

    def add_input(self, entry: Dict[str, Any]) -> None:
        names = {e["name"] for e in self.inputs}
        base, k = entry["name"], 2
        while entry["name"] in names:
            entry["name"] = f"{base}_{k}"
            k += 1
        self.inputs.append(entry)
        self.refresh_inputs()

    def remove_inputs(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.inputs[r]
        self.refresh_inputs()

    def refresh_inputs(self) -> None:
        self._filling = True
        self.table.setRowCount(len(self.inputs))
        for r, e in enumerate(self.inputs):
            stack = int(e["frames"]) > 1
            cells = [e["name"], f"Stack ({e['frames']})" if stack else "Bild",
                     "" if not e.get("pixel_size_nm") else f"{e['pixel_size_nm']:g}",
                     "" if not e.get("voltage_kV") else f"{e['voltage_kV']:g}", "",
                     e.get("folder") or e["target"]]
            for c, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if c in (1, 4, 5):
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                item.setToolTip("\n".join(f"{k}: {e.get(k) or '—'}" for k in ("target", "i_target", "mask", "init")))
                self.table.setItem(r, c, item)
            if stack:
                combo = QtWidgets.QComboBox()
                for key, label in jobs.WARM_STARTS.items():
                    combo.addItem(label, key)
                combo.setCurrentIndex(max(0, combo.findData(e.get("warm_start", "optics"))))
                combo.currentIndexChanged.connect(lambda _=None, entry=e, c=combo: entry.update(warm_start=c.currentData()))
                self.table.setCellWidget(r, 4, combo)
            else:
                self.table.removeCellWidget(r, 4)
                self.table.item(r, 4).setText("—")
        self.table.resizeColumnsToContents()
        frames = sum(int(e["frames"]) for e in self.inputs)
        self.count_label.setText(f"{len(self.inputs)} Eingaben, {frames} Bilder")
        self._filling = False

    def _cell_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._filling:
            return
        e = self.inputs[item.row()]
        text = item.text().strip().replace(",", ".")
        column = item.column()
        if column == 0:
            e["name"] = text or e["name"]
        elif column in (2, 3):
            key = "pixel_size_nm" if column == 2 else "voltage_kV"
            try:
                e[key] = float(text) if text else None
            except ValueError:
                pass
        self.refresh_inputs()

    def choose_init(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "init.json", "", "JSON (*.json)")
        if path:
            self.init_edit.setText(path)

    def template(self) -> Dict[str, Any]:
        return {"init": self.init_edit.text().strip(), "pixel_size_nm": self.pixel_spin.value() or None,
                "voltage_kV": self.voltage_spin.value() or None, "mode": self.mode_combo.currentData() or None}

    # ------------------------------------------------------------------ pipeline
    def refresh_pipeline(self, select: Optional[int] = None) -> None:
        current = self.step_list.currentRow() if select is None else select
        self.step_list.blockSignals(True)
        self.step_list.clear()
        for k, step in enumerate(self.pipeline):
            self.step_list.addItem(f"{k + 1}. {jobs.step_title(step)}")
        self.step_list.blockSignals(False)
        if self.pipeline:
            self.step_list.setCurrentRow(min(max(current, 0), len(self.pipeline) - 1))
        self.show_params()

    def add_step(self) -> None:
        self.pipeline.append(jobs.make_step(self.kind_combo.currentData()))
        self.refresh_pipeline(len(self.pipeline) - 1)

    def remove_step(self) -> None:
        row = self.step_list.currentRow()
        if 0 <= row < len(self.pipeline):
            del self.pipeline[row]
            self.refresh_pipeline(row)

    def move_step(self, delta: int) -> None:
        row = self.step_list.currentRow()
        new = row + delta
        if 0 <= row < len(self.pipeline) and 0 <= new < len(self.pipeline):
            self.pipeline[row], self.pipeline[new] = self.pipeline[new], self.pipeline[row]
            self.refresh_pipeline(new)

    def default_steps(self) -> None:
        self.pipeline = jobs.default_pipeline()
        self.refresh_pipeline(0)

    def show_params(self) -> None:
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        row = self.step_list.currentRow()
        if not 0 <= row < len(self.pipeline):
            self.param_box.setTitle("Parameter")
            return
        step = self.pipeline[row]
        self.param_box.setTitle(f"Parameter: {jobs.STEP_TYPES[step['kind']]['label']}")
        params = step["params"]
        for key, label, kind, default, extra in jobs.STEP_TYPES[step["kind"]]["params"]:
            value = params.get(key, default)
            widget = self._param_widget(kind, value, extra, lambda v, k=key: self._set_param(row, k, v))
            self.param_form.addRow(label, widget)

    def _param_widget(self, kind, value, extra, store):
        if kind == "bool":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(store)
        elif kind == "int":
            low, high, step = extra
            w = int_spin(int(value), low, high, step)
            w.valueChanged.connect(store)
        elif kind == "float":
            low, high, step, decimals = extra
            w = double_spin(float(value), low, high, step, decimals)
            w.valueChanged.connect(store)
        elif kind == "choice":
            w = QtWidgets.QComboBox()
            for key, label in extra.items():
                w.addItem(label, key)
            w.setCurrentIndex(max(0, w.findData(value)))
            w.currentIndexChanged.connect(lambda _=None, c=w: store(c.currentData()))
        else:                                             # a detector of the global directory
            from temsim_ml.models import GLOBAL_MODELS, list_models
            w = QtWidgets.QComboBox()
            w.setEditable(True)
            w.addItem("", "")
            w.setItemText(0, "")
            for m in list_models(GLOBAL_MODELS):
                f1 = m["meta"].get("val_f1")
                w.addItem(m["name"], m["name"])
                w.setItemData(w.count() - 1, f"F1 {f1:.3f}" if f1 is not None else "", QtCore.Qt.ToolTipRole)
            w.lineEdit().setPlaceholderText(f"neuestes in {GLOBAL_MODELS}")
            w.setEditText(str(value or ""))
            w.editTextChanged.connect(lambda text: store(text.strip()))
        return w

    def _set_param(self, row: int, key: str, value) -> None:
        if 0 <= row < len(self.pipeline):
            self.pipeline[row]["params"][key] = value
            item = self.step_list.item(row)
            if item is not None:
                item.setText(f"{row + 1}. {jobs.step_title(self.pipeline[row])}")

    def refresh_presets(self) -> None:
        self.preset_combo.clear()
        self.preset_combo.addItems(jobs.list_pipelines())

    def save_preset(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Pipeline speichern", "Name:",
                                                  text=self.preset_combo.currentText())
        if ok and name.strip():
            jobs.save_pipeline(name.strip(), self.pipeline)
            self.refresh_presets()
            self.preset_combo.setCurrentText(jobs.safe_name(name.strip()))

    def load_preset(self) -> None:
        name = self.preset_combo.currentText()
        if not name:
            return
        try:
            self.pipeline = jobs.load_pipeline(name)
        except jobs.JobError as exc:
            QtWidgets.QMessageBox.warning(self, "Pipeline", str(exc))
            return
        self.refresh_pipeline(0)

    # ------------------------------------------------------------------ the job
    def choose_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Ausgabeordner", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def job(self) -> Dict[str, Any]:
        return {"output": os.path.abspath(os.path.expanduser(self.output_edit.text().strip()))
                if self.output_edit.text().strip() else "",
                "device": self.device_combo.currentText(), "template": self.template(),
                "inputs": copy.deepcopy(self.inputs), "pipeline": copy.deepcopy(self.pipeline)}

    def start(self, confirm: bool = True) -> bool:
        job = self.job()
        problems = jobs.check_job(job)
        if problems:
            QtWidgets.QMessageBox.warning(self, "Auftrag", "Nicht startbar:\n" + "\n".join(problems))
            return False
        out = job["output"]
        if jobs.is_running(out):
            QtWidgets.QMessageBox.warning(self, "Auftrag", "In diesem Ordner läuft schon ein Auftrag.")
            return False
        if confirm and os.path.isdir(out) and os.listdir(out):
            done = sum(jobs.frame_done(out, e, k) for e in job["inputs"] for k in range(int(e["frames"])))
            text = (f"{out} ist nicht leer.\n{done} Bilder sind dort schon fertig und werden übersprungen; "
                    "der Rest wird (neu) bearbeitet.\n\nFortsetzen?")
            if QtWidgets.QMessageBox.question(self, "Auftrag", text) != QtWidgets.QMessageBox.Yes:
                return False
        jobs.save_job(job)
        runner.launch(out)
        self.job_dir = out
        self._log_size = -1
        self.log_view.clear()
        self.poll()
        return True

    def stop(self) -> None:
        if self.job_dir:
            runner.request_stop(self.job_dir)
            self.state_label.setText("Stopp angefordert – endet nach dem laufenden Arbeitsschritt …")

    def choose_job(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Auftragsordner (mit job.json)", self.output_edit.text())
        if path:
            self.open_job(path)

    def open_job(self, path: str) -> None:
        try:
            job = jobs.load_job(path)
        except jobs.JobError as exc:
            QtWidgets.QMessageBox.warning(self, "Auftrag", str(exc))
            return
        self.inputs = job.get("inputs", [])
        self.pipeline = jobs.normalize_pipeline(job.get("pipeline", []))
        template = job.get("template", {})
        self.init_edit.setText(template.get("init") or "")
        self.pixel_spin.setValue(float(template.get("pixel_size_nm") or 0.0))
        self.voltage_spin.setValue(float(template.get("voltage_kV") or 0.0))
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(template.get("mode") or "")))
        self.device_combo.setCurrentText(job.get("device", "auto"))
        self.output_edit.setText(job["output"])
        self.refresh_inputs()
        self.refresh_pipeline(0)
        self.job_dir = job["output"]
        self._log_size = -1
        self.log_view.clear()
        self.poll()

    def poll(self) -> None:
        if not self.job_dir:
            self._set_running(False)
            return
        state = jobs.status(self.job_dir)
        running = state.get("state") == "running"
        self._set_running(running)
        total = int(state.get("total") or sum(int(e["frames"]) for e in self.inputs) or 1)
        self.total_bar.setRange(0, max(1, total))
        self.total_bar.setValue(int(state.get("done") or 0))
        names = {"running": "läuft", "finished": "fertig", "stopped": "gestoppt", "failed": "abgebrochen (Fehler)",
                 "abgebrochen": "abgebrochen (Prozess beendet)"}
        text = f"{names.get(state.get('state'), state.get('state') or 'noch nicht gestartet')}"
        if state.get("failed"):
            text += f", {state['failed']} Fehler"
        current = state.get("current") or {}
        if running and current:
            text += f" — {current.get('input', '')}"
            if int(current.get("frames") or 1) > 1:
                text += f", Frame {current.get('frame', 0) + 1}/{current['frames']}"
            if current.get("title"):
                text += f" — Schritt {current.get('step', 0) + 1}/{current.get('steps', 0)}: {current['title']}"
            if current.get("loss") is not None:
                text += f", Loss {current['loss']:.4g}"
        self.state_label.setText(text + f"   [{self.job_dir}]")
        progress = current.get("progress") if running else None
        if progress and progress[1]:
            self.step_bar.setRange(0, int(progress[1]))
            self.step_bar.setValue(int(progress[0]))
        else:
            self.step_bar.setRange(0, 1)
            self.step_bar.setValue(0)
        log, size = jobs.tail(os.path.join(self.job_dir, jobs.LOG_FILE))
        if size != self._log_size:
            self._log_size = size
            self.log_view.setPlainText(log)
            self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _set_running(self, running: bool) -> None:
        self.input_box.setEnabled(not running)
        self.pipe_box.setEnabled(not running)
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.output_edit.setEnabled(not running)


def launch_window(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m temsim_ml.batch",
                                     description="Automatik: Bilder und Stacks über Nacht rekonstruieren")
    parser.add_argument("job_dir", nargs="?", default="", help="einen bestehenden Auftrag öffnen")
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = BatchWindow(args.job_dir)
    window.show()
    return app.exec()
