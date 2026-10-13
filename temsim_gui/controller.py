r"""Part V, Section 10: the controller.

Theory
======

The interface is a *client*: it holds no model and computes nothing.  What a person does
- pressing a button, moving a spin box - becomes a command to the core; what the core
reports - a ``status``, a ``light`` or a ``frame`` event - becomes a change of the
panels and the canvas.  :class:`Controller` is the one place where this translation
happens, so that the panels (Section 9) stay free of the protocol and the connection
(Section 5) stays free of the panels.

**Who changes what.**  The core is the only owner of the state.  The controller never
assumes that a command had an effect: after every edit it asks for a new frame
(``render``) and shows that.  The only state kept here is *view* state - the zoom,
the display threshold of the occupancy, the loss history - and the memory of what was
last done, which decides two conveniences of the old interface:

* after a seeding, the first start of training releases the positions and intensities
  of the seeded atoms (unless the person has chosen the trainable quantities);
* the seeding view leaves out the strain tiles, because the displacement field is
  not trained while seeding.

**Where files live.**  Paths in commands (open, save) are read and written by the *core*.
If the core runs on another machine, a local file dialog would offer the wrong file
system; the controller then asks for the path as text (:attr:`Controller.remote`).

**Candidate sites.**  While the optimiser runs, the core would have to build a seeder
to test the sites, which costs time that belongs to the optimisation.  Candidates are
therefore requested *without* the loss test while a job runs (yellow marks) and with
it when the core is idle or paused (green marks), as in the old interface.

Implementation
==============
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from PySide6 import QtCore, QtWidgets

from temsim.data.schema import Experiment, TrainSettings

from . import tiles
from .canvas import PlotCanvas
from .connection import Connection
from .frame_model import FrameModel
from .panels import ControlPanel, MinimaPanel, SeedingPanel, StrainPanel
from .train_editor import TrainEditor
from .values_editor import ValuesEditor

BASE_EXTRAS = ("optics", "strain", "potentials")
VALUES_INTERVAL = 2.0
MINIMA_DELAY_MS = 250


class Controller(QtCore.QObject):
    notified = QtCore.Signal(str)

    def __init__(self, connection: Connection, canvas: PlotCanvas, control: ControlPanel, minima: MinimaPanel,
                 seeding: SeedingPanel, strain: StrainPanel, values: ValuesEditor, train: TrainEditor,
                 parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.connection, self.canvas, self.control = connection, canvas, control
        self.minima, self.seeding, self.strain = minima, seeding, strain
        self.values, self.train = values, train
        self.info: Dict[str, Any] = {}
        self.experiment: Optional[Experiment] = None
        self.state, self.job_kind = "idle", None
        self.seeded = False
        self.remote = False
        self.errors: List[str] = []
        self._minima_pending = False
        self._values_asked = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(MINIMA_DELAY_MS)
        self._timer.timeout.connect(self.request_minima)
        self._connect()
        self.connection.set_extras(BASE_EXTRAS)

    # ------------------------------------------------------------------ wiring
    def _connect(self) -> None:
        c, p = self.connection, self.control
        c.frame.connect(self.on_frame)
        c.light.connect(self.on_light)
        c.status.connect(self.on_status)
        c.error.connect(self.on_error)
        p.start_btn.clicked.connect(self.start_training)
        p.seed_btn.clicked.connect(self.start_seeding)
        p.pause_btn.clicked.connect(lambda: self.send(self.client.pause))
        p.resume_btn.clicked.connect(lambda: self.send(self.client.resume))
        p.stop_btn.clicked.connect(lambda: self.send(self.client.stop))
        p.render_btn.clicked.connect(self.render)
        p.tile_combo.currentIndexChanged.connect(lambda _=None: self.canvas.set_view(p.tile_combo.currentData()))
        p.show_atoms.toggled.connect(self.on_show_atoms)
        p.show_occ_spin.valueChanged.connect(self.on_show_occupancy)
        p.reset_occ_btn.clicked.connect(lambda: self.edit("reset_occupancies"))
        p.reset_int_btn.clicked.connect(lambda: self.edit("reset_intensities"))
        p.reset_free_btn.clicked.connect(lambda: self.edit("reset_free_positions"))
        p.reset_losses_btn.clicked.connect(self.reset_losses)
        p.reset_zoom_btn.clicked.connect(self.canvas.reset_zoom)
        p.remove_btn.clicked.connect(self.remove_low_occupancy)
        for widget in (p.gradients, p.descent, p.relative):
            widget.toggled.connect(self.on_gradient_settings)
        p.length_spin.valueChanged.connect(self.on_gradient_settings)

        m = self.minima
        m.show_check.toggled.connect(self.on_show_minima)
        m.threshold.valueChanged.connect(self._later)
        m.distance.valueChanged.connect(self._later)
        for check in (*m.checks.values(), m.atom_distance):
            check.toggled.connect(self._later)
        for key, spin in m.spins.items():
            spin.valueChanged.connect(lambda value, k=key: self.on_minima_radius(k, value))
        m.insert_btn.clicked.connect(self.insert_minima)

        self.seeding.config_changed.connect(self.on_seeding_config)
        self.strain.apply_clicked.connect(self.apply_control_points)
        self.train.active_group_changed.connect(lambda _=None: self.render())
        self.values.save_clicked.connect(lambda: self.save(None))
        self.values.save_as_clicked.connect(self.save_as)
        self.values.load_clicked.connect(self.load)

    @property
    def client(self):
        return self.connection.client

    def send(self, call: Callable, done: Optional[Callable] = None) -> None:
        self.connection.submit(call, done)

    def notify(self, text: str) -> None:
        self.notified.emit(text)

    # ------------------------------------------------------------------ opening
    def refresh_model(self) -> None:
        """Read everything the panels show from the core."""

        def read():
            client = self.client
            info = client.info()["info"]
            if not info.get("open"):
                return info, None, None, client.seeding_config()
            return info, client.get_experiment(), client.get_train(), client.seeding_config()

        self.send(read, self.on_model)

    def on_model(self, result) -> None:
        info, experiment, settings, config = result
        self.info = info
        self.experiment = experiment
        self.control.set_state(self.state, self.job_kind, open_=bool(info.get("open")))
        self.seeding.show_values(config)
        self.minima.spins["touch_A"].blockSignals(True)
        for key, spin in self.minima.spins.items():
            if key in config:
                spin.setValue(float(config[key]))
        self.minima.spins["touch_A"].blockSignals(False)
        if experiment is None:
            return
        self.seeding.set_stem(info.get("mode") == "stem_haadf")
        self.strain.build({g.name: list(g.control_points) for g in experiment.groups})
        self.train.load(experiment, settings)
        self.request_values()
        self.render()

    def open_run(self, directory: str, target_path: Optional[str] = None) -> None:
        self.send(lambda: self.client.open_run(directory, target_path), lambda _info: self.refresh_model())

    def open_dataset(self, path: str, **options) -> None:
        self.send(lambda: self.client.open_dataset(path, **options), lambda _info: self.refresh_model())

    # ------------------------------------------------------------------ events
    def on_status(self, data: Dict[str, Any]) -> None:
        previous = self.state
        job = data.get("job") or {}
        self.state = data.get("state", "idle")
        self.job_kind = job.get("kind") if self.state != "idle" else None
        self.control.set_state(self.state, self.job_kind, open_=bool(data.get("open", True)))
        if previous != "idle" and self.state == "idle":
            self.render()
            self.request_values()
        if not self.info.get("open") and data.get("open"):
            self.refresh_model()

    def on_light(self, data: Dict[str, Any]) -> None:
        if data.get("kind") == "seeding":
            step, loss = data.get("trials"), data.get("loss")
        else:
            step, loss = data.get("step"), data.get("loss")
        self.control.set_progress(step, loss)
        if loss is not None and step is not None:
            self.canvas.history.add(step, loss)
            self.canvas.update_loss()

    def on_frame(self, frame: FrameModel) -> None:
        if self.canvas.configure([layer.name for layer in frame.layers], frame.stem):
            self.fill_tiles()
        self.canvas.update_frame(frame)
        if self.state == "idle":
            self.control.set_progress(frame.step, frame.loss)
        now = time.monotonic()
        if now - self._values_asked > VALUES_INTERVAL:
            self.request_values()
        self.request_minima()

    def on_error(self, message: str, trace: str) -> None:
        self.errors.append(message)
        self.notify(message)
        if trace:
            print(trace)

    # ------------------------------------------------------------------ tiles and view
    def fill_tiles(self) -> None:
        titles = self.canvas.titles()
        entries = [("alle Kacheln", None), (titles[tiles.COMPARE], tiles.COMPARE)]
        entries += [(titles.get(key, key), key) for key in self.canvas.keys()]
        self.control.fill_tiles(entries)

    def set_seeding_view(self, seeding: bool) -> None:
        if self.canvas.set_seeding(seeding):
            self.fill_tiles()

    def on_show_atoms(self, on: bool) -> None:
        self.canvas.show_atoms = bool(on)
        self._redraw()

    def on_show_occupancy(self, value: float) -> None:
        self.canvas.min_occupancy = float(value)
        self._redraw()

    def _redraw(self) -> None:
        if self.canvas.frame is not None:
            self.canvas.update_frame(self.canvas.frame)

    def on_gradient_settings(self, *_args) -> None:
        p, c = self.control, self.canvas
        want = p.gradients.isChecked()
        c.gradient_descent = p.descent.isChecked()
        c.gradient_length = float(p.length_spin.value())
        c.gradient_relative_to_view = p.relative.isChecked()
        if want != c.show_gradients:
            c.show_gradients = want
            self.connection.set_extras(BASE_EXTRAS + (("gradients",) if want else ()))
            self.render()
        else:
            c.refresh_marks()

    def reset_losses(self) -> None:
        self.canvas.history.reset()
        self.canvas.update_loss()

    # ------------------------------------------------------------------ jobs
    def render(self) -> None:
        extras = self.connection.extras

        def done(result) -> None:
            data, arrays = result
            self.on_frame(FrameModel.from_wire(data, arrays))

        if self.info.get("open"):
            self.send(lambda: self.client.frame(extras=extras), done)

    def start_training(self) -> None:
        if self.seeded and not self.train.touched:
            self.train.enable_after_seeding()
        self.seeded = False
        self.set_seeding_view(False)
        self.send(self.client.start_training)

    def start_seeding(self) -> None:
        self.seeded = True
        self.set_seeding_view(True)
        options = self.seeding.job_options()
        self.send(lambda: self.client.start_seeding(**options))

    def edit(self, name: str, **args) -> None:
        def done(report) -> None:
            self.notify(f"{name}: {report}")
            self.render()
        self.send(lambda: self.client.edit(name, **args), done)

    def remove_low_occupancy(self) -> None:
        self.edit("remove_low_occupancy", threshold=float(self.control.remove_spin.value()))

    def apply_control_points(self) -> None:
        if self.experiment is None:
            return
        wanted = self.strain.values()
        for group in self.experiment.groups:
            points = wanted.get(group.name)
            if points is not None and list(points) != list(group.control_points):
                self.edit("resample_control", group=group.name, control_points=points)
                group.control_points = list(points)

    # ------------------------------------------------------------------ values
    def request_values(self) -> None:
        self._values_asked = time.monotonic()
        self.send(self.client.get_values, self.on_values)

    def on_values(self, values: Dict[str, Any]) -> None:
        self.values.show_values(values)
        c10, c30 = values.get("optics.C10"), values.get("optics.C30")
        if c10 is not None and c30 is not None:
            self.canvas.ctf_note = f"C10 = {c10:.1f} nm, C30 = {c30:.2f} um"

    # ------------------------------------------------------------------ candidate sites
    def _later(self, *_args) -> None:
        if self.minima.show_check.isChecked():
            self._timer.start()

    def on_show_minima(self, on: bool) -> None:
        self.canvas.show_minima = bool(on)
        if on:
            self.request_minima()
        else:
            self.canvas.set_minima(None)

    def on_minima_radius(self, key: str, value: float) -> None:
        self.send(lambda: self.client.seeding_config({key: float(value)}))
        if key in self.seeding.spins:
            self.seeding.show_values({key: float(value)})
        self._later()

    def request_minima(self) -> None:
        if not self.minima.show_check.isChecked() or self._minima_pending or not self.info.get("open"):
            return
        verified = self.state != "running"
        options = self.minima.insert_options()
        verify = self.minima.verify() if verified else None
        self._minima_pending = True

        def call():
            return self.client.minima(options["threshold"], options["distance_px"], None,
                                      options["atom_distance"], verify)

        def done(points) -> None:
            self._minima_pending = False
            self.canvas.set_minima(points, verified)

        def failed(exc: BaseException) -> None:
            self._minima_pending = False
            self.on_error(f"{type(exc).__name__}: {exc}", "")

        self.connection.submit(call, done, failed)

    def insert_minima(self) -> None:
        options = self.minima.insert_options()

        def done(report) -> None:
            self.notify(f"{report['accepted']} von {report['candidates']} Minima eingefuegt, "
                        f"{report['atoms']} Atome")
            if self.minima.save_check.isChecked():
                self.save(None)
            self.render()

        self.send(lambda: self.client.insert_minima(**options), done)

    def on_seeding_config(self, patch: Dict[str, Any]) -> None:
        self.send(lambda: self.client.seeding_config(patch))
        for key in ("touch_A", "merge_A"):
            if key in patch:
                self.minima.spins[key].blockSignals(True)
                self.minima.spins[key].setValue(float(patch[key]))
                self.minima.spins[key].blockSignals(False)

    # ------------------------------------------------------------------ files
    def save(self, directory: Optional[str] = None) -> None:
        self.send(lambda: self.client.save(directory), lambda path: self.notify(f"gespeichert: {path}"))

    def ask_directory(self, title: str) -> str:
        """A directory *on the machine of the core*: browsed locally, typed in if the core is remote."""

        if self.remote:
            text, ok = QtWidgets.QInputDialog.getText(self.canvas, title, "Verzeichnis auf dem Core-Rechner:")
            return text.strip() if ok else ""
        return QtWidgets.QFileDialog.getExistingDirectory(self.canvas, title)

    def save_as(self) -> None:
        directory = self.ask_directory("Speichern unter ...")
        if directory:
            self.save(directory)

    def load(self) -> None:
        directory = self.ask_directory("Run laden")
        if directory:
            self.open_run(directory)

    def finish(self, save: bool) -> None:
        """Stop the job and, if asked, save; the queued commands are completed by ``Connection.close``."""

        if self.state != "idle":
            self.connection.submit(self.client.stop)
        if save and self.info.get("open"):
            self.connection.submit(self.client.save)
