r"""Part V, Section 9: the control panels.

Theory
======

The right-hand column of the window is organised by *what one is doing*.  A panel is
*only* a set of widgets: it knows its defaults, how to show a state and how to hand its
settings out, but nothing about the core; the :class:`~temsim_gui.controller.Controller`
connects panels, canvas and connection.

Two **modes** exist, chosen in :class:`ControlPanel`, and only what the mode needs is shown:

* *Training* - the optimiser runs; the gradient arrows, the editing of the model (resets,
  removal of weak atoms) and the displacement-field grid (:class:`StrainPanel`) belong here.
* *Atome einfuegen* - new atoms are put into the model; :class:`InsertPanel` merges the two ways
  of doing so behind one choice of *method*: random trials that are kept when the correlation
  improves (:class:`SeedingPanel`, a job) or candidate sites from the minima of the difference
  image (:class:`MinimaPanel`, a single action).  What both share - the collision rule
  (``touch_A`` removes neighbours, ``merge_A`` pulls them into the middle) - is entered once.

Inside a panel, an option is shown only if it can matter: the bond lengths only for the
neighbour strategy, the temperature only for the gradient strategy, the relaxation
only for STEM, the margin only if the margin check is on; the buttons Pause / Resume
appear according to the state of the job.

Implementation
==============
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtWidgets

from .widgets import CollapsibleSection, double_spin, int_spin, row_of, set_quietly

TRAINING, INSERT = "training", "insert"


class ControlPanel(QtWidgets.QWidget):
    """Status, mode, job buttons, view options and (in the training mode) editing of the model."""

    MODES = (("Training", TRAINING), ("Atome einfuegen", INSERT))

    mode_changed = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._inserting, self._seeding_method = False, True
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._job_box())
        self.view_section = self._view_section()
        self.edit_section = self._edit_section()
        layout.addWidget(self.view_section)
        layout.addWidget(self.edit_section)
        self.save_on_close = QtWidgets.QCheckBox("Beim Schliessen speichern")
        self.save_on_close.setChecked(True)
        self.save_on_close.setToolTip("Speichert beim Schliessen des Fensters den Stand im Ausgabeordner des Kerns.")
        layout.addWidget(self.save_on_close)
        self.set_state("idle")
        self.show_actions(False, True)

    # ------------------------------------------------------------------ building
    def _job_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Lauf")
        layout = QtWidgets.QVBoxLayout(box)
        self.state_label = QtWidgets.QLabel("idle")
        self.step_label = QtWidgets.QLabel("step: -")
        self.loss_label = QtWidgets.QLabel("loss: -")
        status = row_of(self.state_label, self.step_label, self.loss_label)
        status.layout().insertStretch(1)
        layout.addWidget(status)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.mode_combo.setMinimumContentsLength(12)
        for label, key in self.MODES:
            self.mode_combo.addItem(label, key)
        self.mode_combo.setToolTip("Training: Optimierer laufen lassen.\n"
                                   "Atome einfuegen: neue Atome per Seeding oder aus den Diff-Minima setzen.")
        self.mode_combo.currentIndexChanged.connect(lambda _=None: self.mode_changed.emit(self.mode))
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Modus", self.mode_combo)
        layout.addLayout(form)
        button = QtWidgets.QPushButton
        self.start_btn, self.seed_btn = button("Training starten"), button("Seeding starten")
        self.pause_btn, self.resume_btn, self.stop_btn = button("Pause"), button("Resume"), button("Stop")
        self.render_btn = button("Render")
        self.seed_btn.setToolTip("Atome zufaellig einfuegen und behalten, wenn die Korrelation besser wird.\n"
                                 "Pause/Resume/Stop gelten auch hierfuer.")
        self.render_btn.setToolTip("Bild vom Kern neu anfordern.")
        buttons = row_of(self.start_btn, self.seed_btn, self.pause_btn, self.resume_btn, self.stop_btn, self.render_btn)
        layout.addWidget(buttons)
        return box

    def _view_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Ansicht", expanded=True)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        section.content_layout.addLayout(form)
        self.tile_combo = QtWidgets.QComboBox()
        self.tile_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.tile_combo.setMinimumContentsLength(12)
        self.tile_combo.setToolTip("Eine Kachel gross zeigen - Simulation, Diff, Target, CTF, Potential ...\n"
                                   "Zoomen und Verschieben mit der Toolbar unter dem Bild.")
        form.addRow("Kachel", self.tile_combo)
        self.show_atoms = QtWidgets.QCheckBox("Atompositionen auf dem Target zeigen")
        self.show_atoms.setChecked(True)
        form.addRow(self.show_atoms)
        self.show_occ_spin = double_spin(0.5, 0.0, 1.0, 0.05, 2,
                                         tip="Nur Atome mit sigmoid(Occupancy) ab diesem Wert werden angezeigt "
                                             "(Positionen, Pfeile,\nHistogramm). 0 = alle. Das Modell bleibt unveraendert.")
        form.addRow("Anzeige ab Occ", self.show_occ_spin)
        self.reset_zoom_btn = QtWidgets.QPushButton("Zoom zuruecksetzen")
        self.reset_losses_btn = QtWidgets.QPushButton("Loss-Verlauf leeren")
        section.content_layout.addWidget(row_of(self.reset_zoom_btn, self.reset_losses_btn))

        self.gradients = QtWidgets.QCheckBox("Gradient-Vektoren je Atom")
        self.gradients.setToolTip("Pfeil je Atom: in welche Richtung der Loss sich beim Verschieben des Atoms\n"
                                  "aendert. Der laengste sichtbare Pfeil hat die eingestellte Laenge, alle anderen\n"
                                  "sind dazu proportional. Mausrad zoomt in die Bilder.")
        self.descent = QtWidgets.QCheckBox("Abstiegsrichtung (-Gradient)")
        self.descent.setChecked(True)
        self.length_spin = double_spin(3.0, 0.2, 200.0, 0.5, 1, suffix=" px",
                                       tip="Das Atom mit dem groessten Gradienten bekommt diese Laenge.")
        self.relative = QtWidgets.QCheckBox("auf Ausschnitt normieren")
        self.relative.setToolTip("Aus: Bezug ist das Maximum aller Atome.\nAn: Bezug ist das Maximum im sichtbaren Ausschnitt.")
        self.gradient_details = row_of(self.descent, self.length_spin, self.relative)
        self.gradient_details.setVisible(False)
        self.gradients.toggled.connect(self.gradient_details.setVisible)
        self.gradient_box = QtWidgets.QWidget()
        box_layout = QtWidgets.QVBoxLayout(self.gradient_box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.addWidget(self.gradients)
        box_layout.addWidget(self.gradient_details)
        section.content_layout.addWidget(self.gradient_box)
        return section

    def _edit_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Modell bearbeiten (aktive Gruppe)", expanded=False)
        button = QtWidgets.QPushButton
        self.reset_occ_btn = button("Occupancies zuruecksetzen")
        self.reset_int_btn = button("Intensitaeten zuruecksetzen")
        self.reset_int_btn.setToolTip("Intensitaeten zurueck auf die Startwerte -\n"
                                      "nicht auf 0, die Intensitaet ist die Ordnungszahl.")
        self.reset_free_btn = button("Freie Positionen zuruecksetzen")
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self.reset_occ_btn, 0, 0)
        grid.addWidget(self.reset_int_btn, 0, 1)
        grid.addWidget(self.reset_free_btn, 1, 0)
        section.content_layout.addLayout(grid)
        self.remove_btn = button("Atome unter Occ entfernen")
        self.remove_btn.setToolTip("Loescht alle Atome der aktiven Gruppe, deren sigmoid(Occupancy) unter dem Wert\n"
                                   "rechts liegt, endgueltig aus dem Modell (Adam-Zustand wird zurueckgesetzt).")
        self.remove_spin = double_spin(0.5, 0.0, 1.0, 0.05, 2, prefix="Occ < ")
        section.content_layout.addWidget(row_of(self.remove_btn, self.remove_spin))
        return section

    def add_mode_panel(self, widget: QtWidgets.QWidget) -> None:
        """Put the panel of a mode directly below the job box."""

        self.layout().insertWidget(1, widget)

    # ------------------------------------------------------------------ mode
    @property
    def mode(self) -> str:
        return self.mode_combo.currentData()

    def set_mode(self, mode: str) -> None:
        index = self.mode_combo.findData(mode)
        if index >= 0 and index != self.mode_combo.currentIndex():
            self.mode_combo.setCurrentIndex(index)

    def show_actions(self, inserting: bool, seeding_method: bool) -> None:
        """Mode dependent parts: the start button that fits, the training-only options."""

        self._inserting, self._seeding_method = inserting, seeding_method
        self.start_btn.setVisible(not inserting)
        self.seed_btn.setVisible(inserting and seeding_method)
        self.gradient_box.setVisible(not inserting)
        self.edit_section.setVisible(not inserting)

    # ------------------------------------------------------------------ showing
    def set_state(self, state: str, job: Optional[str] = None, open_: bool = True) -> None:
        """Label, enabled and visible buttons for ``state`` in ``idle | running | paused``."""

        text = state if job is None or state == "idle" else f"{state} ({job})"
        self.state_label.setText(text)
        idle, running, paused = state == "idle", state == "running", state == "paused"
        self.start_btn.setEnabled(idle and open_)
        self.seed_btn.setEnabled(idle and open_)
        self.pause_btn.setEnabled(running)
        self.resume_btn.setEnabled(paused)
        self.stop_btn.setEnabled(running or paused)
        self.pause_btn.setVisible(not idle and not paused)
        self.resume_btn.setVisible(paused)
        self.stop_btn.setVisible(not idle)
        self.mode_combo.setEnabled(idle)

    def set_progress(self, step: Optional[int], loss: Optional[float]) -> None:
        self.step_label.setText("step: -" if step is None else f"step: {step}")
        self.loss_label.setText("loss: -" if loss is None else f"loss: {loss:.6g}")

    def fill_tiles(self, entries: Sequence[Tuple[str, Optional[str]]]) -> None:
        """``(label, key)`` of the tile combo; the chosen tile stays if it still exists."""

        current = self.tile_combo.currentData() if self.tile_combo.count() else None
        self.tile_combo.blockSignals(True)
        self.tile_combo.clear()
        for label, key in entries:
            self.tile_combo.addItem(label, key)
        self.tile_combo.setCurrentIndex(max(self.tile_combo.findData(current), 0))
        self.tile_combo.blockSignals(False)


class MinimaPanel(QtWidgets.QWidget):
    """Candidate atom sites from the minima of the difference image."""

    CHECKS = (("margin", "Rand beachten", True, "Minima naeher am Bildrand als 'Rand [px]' entfallen."),
              ("collisions", "Kollisionsregel", True,
               "Atome naeher als 'Entfernen bis' werden beim Einfuegen entfernt,\n"
               "Atome bis 'Verschieben bis' ziehen das neue Atom in die Mitte."),
              ("loss", "Loss darf nicht schlechter werden", True,
               "Ein Minimum zaehlt nur, wenn das neue Atom den Loss nicht verschlechtert."))

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.show_check = QtWidgets.QCheckBox("Minima im Diff-Bild zeigen")
        self.show_check.setToolTip("Lokale Minima von Sim - Target: dort ist das Target heller, also fehlt ein Atom.")
        self.threshold = double_spin(0.5, 0.0, 1.0, 0.05, 2,
                                     tip="Ein Minimum zaehlt, wenn es tiefer als dieser Bruchteil des tiefsten Minimums liegt.")
        self.distance = double_spin(3.0, 1.0, 50.0, 0.5, 1, suffix=" px", tip="Mindestabstand der Minima untereinander.")
        self.checks: Dict[str, QtWidgets.QCheckBox] = {}
        for key, label, default, tip in self.CHECKS:
            check = QtWidgets.QCheckBox(label)
            check.setChecked(default)
            check.setToolTip(tip)
            self.checks[key] = check
        self.atom_distance = QtWidgets.QCheckBox("Abstand zu Atomen >= Mindestabstand")
        self.atom_distance.setToolTip("Minima naeher als 'Mindestabstand' an einem vorhandenen Atom entfallen.")
        margin = double_spin(0.0, 0.0, 100.0, 0.05, 2, tip="Mindestabstand zum Bildrand.")
        self.spins: Dict[str, QtWidgets.QDoubleSpinBox] = {"margin_px": margin}
        self.save_check = QtWidgets.QCheckBox("Nach dem Einfuegen speichern")
        self.save_check.setChecked(True)
        self.insert_btn = QtWidgets.QPushButton("Minima als Atome einfuegen")
        self.insert_btn.setToolTip("Setzt an jedem gezeigten Minimum ein neues Atom in die aktive Gruppe\n"
                                   "(Occupancy/Intensitaet wie beim Seeding).")
        form.addRow(self.show_check)
        form.addRow("Schwelle (x tiefstes Minimum)", self.threshold)
        form.addRow("Mindestabstand", self.distance)
        form.addRow(self.atom_distance)
        form.addRow(self.checks["margin"])
        form.addRow("Rand [px]", margin)
        form.addRow(self.checks["collisions"])
        form.addRow(self.checks["loss"])
        form.addRow(self.save_check)
        form.addRow(self.insert_btn)
        self._form = form
        self.checks["margin"].toggled.connect(self._show_margin)
        self._show_margin(self.checks["margin"].isChecked())

    def _show_margin(self, on: bool) -> None:
        self._form.setRowVisible(self.spins["margin_px"], bool(on))

    def verify(self) -> Dict[str, bool]:
        """Keys of ``Session.minima(verify=...)``."""

        return {"check_margin": self.checks["margin"].isChecked(),
                "check_collisions": self.checks["collisions"].isChecked(),
                "check_loss": self.checks["loss"].isChecked()}

    def config_patch(self) -> Dict[str, float]:
        return {key: float(spin.value()) for key, spin in self.spins.items()}

    def insert_options(self) -> Dict[str, Any]:
        return {"threshold": float(self.threshold.value()), "distance_px": float(self.distance.value()),
                "atom_distance": self.atom_distance.isChecked(), **self.verify()}


class SeedingPanel(QtWidgets.QWidget):
    """Options of the random seeding; changes go to the core at once, also while it runs."""

    #: (key, label, minimum, maximum, step, decimals, tooltip); keys of SeedConfig or of the job
    MAIN = (
        ("max_atoms", "max. Atome (0 = frei)", 0, 1_000_000, 100, 0, "Seeding endet, sobald so viele Atome gesetzt sind."),
        ("trials", "max. Versuche", 1, 100_000_000, 1000, 0, ""),
    )
    NEIGHBOURS = (
        ("bond_A", "Bindungslaenge [A]", 0.0, 10.0, 0.05, 2, "Wahrscheinlichster Abstand zum Nachbaratom."),
        ("bond_sigma_A", "Bindung sigma [A]", 0.01, 5.0, 0.05, 2, "Breite der Gauss-Verteilung um die Bindungslaenge."),
        ("neighbour_min_A", "Nachbar min. Abstand [A]", 0.0, 10.0, 0.05, 2, "Harter Cutoff nach unten."),
        ("neighbour_max_A", "Nachbar max. Abstand [A]", 0.0, 10.0, 0.05, 2, "Harter Cutoff nach oben."),
    )
    GRADIENT = (
        ("gradient_temperature", "Gradient Temperatur", 0.01, 5.0, 0.05, 2,
         "Klein: fast immer die beste Stelle der Landschaft; gross: breiter gestreut."),
    )
    MOVES = (
        ("move_prob", "P(Atom verschieben)", 0.0, 1.0, 0.05, 2,
         "Anteil der Versuche, ein vorhandenes Atom zu verschieben statt eins einzufuegen."),
        ("remove_prob", "P(Atom entfernen)", 0.0, 1.0, 0.05, 2, "Anteil der Versuche, ein vorhandenes Atom wegzunehmen."),
        ("move_sigma_A", "Verschiebung sigma [A]", 0.0, 10.0, 0.05, 2, "Streuung der Gauss-Verschiebung eines Atoms."),
    )
    RELAX = (
        ("relax_prob", "P(Einfuegen + Ausweichen)", 0.0, 1.0, 0.05, 2,
         "Anteil der Versuche: neues Atom an einer von einem Nachbarn blockierten Stelle, der Nachbar\n"
         "sucht sich den besten neuen Platz. Nur STEM."),
        ("relax_A", "Ausweich-Reichweite [A]", 0.0, 10.0, 0.05, 2, "Nachbaratome in diesem Abstand duerfen ausweichen."),
    )
    TUNING = (
        ("tune_every", "Training alle N Versuche (0 = aus)", 0, 1_000_000, 100, 0,
         "Zwischendurch ein paar Gradientenschritte auf Sonde/Mikroskop-Parametern."),
        ("tune_steps", "Schritte je Training", 1, 1000, 1, 0, ""),
    )
    FIELDS = MAIN + NEIGHBOURS + GRADIENT + MOVES + RELAX + TUNING
    JOB_KEYS = ("trials", "tune_every", "tune_steps")
    JOB_DEFAULTS = {"trials": 100_000, "tune_every": 0, "tune_steps": 3}

    #: how new positions are chosen: key -> (label, neighbour_seeding, gradient_seeding)
    STRATEGIES = {"uniform": ("gleichverteilt", False, False),
                  "neighbour": ("um vorhandene Atome", True, False),
                  "gradient": ("Loss-Landschaft (STEM)", False, True),
                  "both": ("um Atome + Landschaft (STEM)", True, True)}

    config_changed = QtCore.Signal(dict)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.stem = True
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.form = form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(form)
        self.strategy = QtWidgets.QComboBox()
        self.strategy.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.strategy.setMinimumContentsLength(12)
        self.strategy.setToolTip("Wo neue Atome gesetzt werden:\n"
                                 "gleichverteilt ueber das Bild; der Reihe nach im Bindungsabstand um vorhandene Atome;\n"
                                 "gezogen aus der Loss-Landschaft (angenommen wird weiterhin nur nach der exakten lokalen Loss).")
        self._fill_strategies()
        self.strategy.currentIndexChanged.connect(self._on_strategy)
        form.addRow("Positionen", self.strategy)
        self.weights = QtWidgets.QLineEdit("1")
        self.weights.setToolTip("Gewichte fuer neue Atome, getrennt durch Leerzeichen (z.B. 1 0.7 0.5 0.3).\n"
                                "Je Einfuegen gewinnt das Gewicht mit dem besten Loss.")
        self.weights.editingFinished.connect(self._on_weights)
        form.addRow("Gewichte", self.weights)
        self.spins: Dict[str, QtWidgets.QAbstractSpinBox] = {}
        self._add_fields(form, self.NEIGHBOURS + self.GRADIENT + self.MAIN)
        self.moves = CollapsibleSection("Verschieben / Entfernen / Ausweichen", expanded=False)
        moves_form = QtWidgets.QFormLayout()
        moves_form.setContentsMargins(0, 0, 0, 0)
        self.moves.content_layout.addLayout(moves_form)
        self._add_fields(moves_form, self.MOVES + self.RELAX)
        self.moves_form = moves_form
        outer.addWidget(self.moves)
        self.tuning = CollapsibleSection("Training zwischendurch", expanded=False)
        tuning_form = QtWidgets.QFormLayout()
        tuning_form.setContentsMargins(0, 0, 0, 0)
        self.tuning.content_layout.addLayout(tuning_form)
        self._add_fields(tuning_form, self.TUNING)
        outer.addWidget(self.tuning)
        self.show_values(dict(self.JOB_DEFAULTS))
        self._show_rows()

    def _add_fields(self, form: QtWidgets.QFormLayout, fields) -> None:
        for key, label, low, high, step, decimals, tip in fields:
            spin = double_spin(low, low, high, step, decimals, tip=tip) if decimals else int_spin(int(low), int(low), int(high), int(step), tip)
            spin.valueChanged.connect(lambda value, k=key: self._on_value(k, value))
            form.addRow(label, spin)
            self.spins[key] = spin

    def _fill_strategies(self) -> None:
        current = self.strategy.currentData()
        self.strategy.blockSignals(True)
        self.strategy.clear()
        for key, (label, _neighbour, gradient) in self.STRATEGIES.items():
            if self.stem or not gradient:
                self.strategy.addItem(label, key)
        index = self.strategy.findData(current)
        self.strategy.setCurrentIndex(max(index, 0))
        self.strategy.blockSignals(False)

    # ------------------------------------------------------------------ showing only what matters
    def _flags(self) -> Tuple[bool, bool]:
        _label, neighbour, gradient = self.STRATEGIES[self.strategy.currentData() or "uniform"]
        return neighbour, gradient

    def _show_rows(self) -> None:
        neighbour, gradient = self._flags()
        for key, *_ in self.NEIGHBOURS:
            self.form.setRowVisible(self.spins[key], neighbour)
        for key, *_ in self.GRADIENT:
            self.form.setRowVisible(self.spins[key], gradient)
        for key, *_ in self.RELAX:
            self.moves_form.setRowVisible(self.spins[key], self.stem)

    def _on_strategy(self, _index: int = 0) -> None:
        neighbour, gradient = self._flags()
        self._show_rows()
        self.config_changed.emit({"neighbour_seeding": neighbour, "gradient_seeding": gradient})

    def _on_weights(self) -> None:
        try:
            values = sorted({float(t) for t in self.weights.text().replace(",", " ").split()}, reverse=True)
        except ValueError:
            values = []
        if values and all(0.0 < v <= 1.5 for v in values):
            self.config_changed.emit({"weight_options": values})
        else:
            self.weights.setText(self._weights_text(self._weights))

    @staticmethod
    def _weights_text(values: Sequence[float]) -> str:
        return " ".join(f"{w:g}" for w in values)

    _weights: Sequence[float] = (1.0,)

    def _on_value(self, key: str, value) -> None:
        if key in self.JOB_KEYS:
            return
        if key == "max_atoms":
            self.config_changed.emit({"max_atoms": int(value) or None})
        else:
            self.config_changed.emit({key: float(value)})

    def show_values(self, config: Dict[str, Any]) -> None:
        """Show a SeedConfig dictionary (and job options) without sending anything."""

        for key, spin in self.spins.items():
            if key in config and config[key] is not None:
                set_quietly(spin, config[key])
            elif key == "max_atoms" and key in config:
                set_quietly(spin, 0)
        if "neighbour_seeding" in config or "gradient_seeding" in config:
            neighbour, gradient = self._flags()
            neighbour = bool(config.get("neighbour_seeding", neighbour))
            gradient = bool(config.get("gradient_seeding", gradient)) and self.stem
            for key, (_label, n, g) in self.STRATEGIES.items():
                if (n, g) == (neighbour, gradient):
                    index = self.strategy.findData(key)
                    self.strategy.blockSignals(True)
                    self.strategy.setCurrentIndex(max(index, 0))
                    self.strategy.blockSignals(False)
            self._show_rows()
        if "weight_options" in config:
            self._weights = tuple(config["weight_options"])
            set_quietly(self.weights, self._weights_text(self._weights))

    def job_options(self) -> Dict[str, int]:
        return {key: int(self.spins[key].value()) for key in self.JOB_KEYS}

    def set_stem(self, stem: bool) -> None:
        """The loss landscape and the relaxation need STEM (Part II, Section 5)."""

        self.stem = bool(stem)
        self._fill_strategies()
        self._show_rows()


class InsertPanel(QtWidgets.QGroupBox):
    """Insertion of atoms: one choice of method, the shared collision rule, the options of the method."""

    METHODS = (("Zufaellig wuerfeln (Seeding)", "seeding"), ("Aus Diff-Minima", "minima"))
    COLLISION = (("touch_A", "Entfernen bis [A]", "Atome, die naeher als das am neuen Atom liegen, werden entfernt."),
                 ("merge_A", "Verschieben bis [A]",
                  "Zwischen 'Entfernen bis' und diesem Radius wird nicht neu eingefuegt:\n"
                  "das Nachbaratom wandert in die Mitte zum neuen Punkt."))

    config_changed = QtCore.Signal(dict)
    method_changed = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__("Atome einfuegen", parent)
        self.seeding, self.minima = SeedingPanel(), MinimaPanel()
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(form)
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.method_combo.setMinimumContentsLength(12)
        for label, key in self.METHODS:
            self.method_combo.addItem(label, key)
        self.method_combo.setToolTip("Seeding: zufaellige Versuche, behalten wird, was die Korrelation verbessert (ein Lauf).\n"
                                     "Diff-Minima: Atome an den Stellen setzen, an denen das Target heller ist (einmalig).")
        form.addRow("Methode", self.method_combo)
        self.collision: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        for key, label, tip in self.COLLISION:
            spin = double_spin(0.5 if key == "touch_A" else 1.0, 0.0, 10.0, 0.05, 2, tip=tip)
            spin.valueChanged.connect(lambda value, k=key: self.config_changed.emit({k: float(value)}))
            form.addRow(label, spin)
            self.collision[key] = spin
        layout.addWidget(self.seeding)
        layout.addWidget(self.minima)
        self.seeding.config_changed.connect(self.config_changed.emit)
        self.method_combo.currentIndexChanged.connect(lambda _=None: self._method_changed())
        self._active = True
        self.set_active(False)

    @property
    def method(self) -> str:
        return self.method_combo.currentData()

    def _method_changed(self) -> None:
        self._apply()
        self.method_changed.emit(self.method)

    def set_active(self, active: bool) -> None:
        """Shown in the insertion mode only; the minima marks vanish with it."""

        self._active = bool(active)
        self._apply()

    def _apply(self) -> None:
        self.setVisible(self._active)
        self.seeding.setVisible(self.method == "seeding")
        self.minima.setVisible(self.method == "minima")
        if not self._active or self.method != "minima":
            self.minima.show_check.setChecked(False)

    def show_values(self, config: Dict[str, Any]) -> None:
        """Show the configuration of the core without sending anything."""

        for key, spin in self.collision.items():
            if key in config:
                set_quietly(spin, float(config[key]))
        if "margin_px" in config:
            set_quietly(self.minima.spins["margin_px"], float(config["margin_px"]))
        self.seeding.show_values(config)


class StrainPanel(QtWidgets.QWidget):
    """Control points of the displacement field, per group."""

    apply_clicked = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setToolTip("Zahl der Kontrollpunkte je Gruppe. Beim Uebernehmen wird das vorhandene Feld interpoliert -\n"
                        "Verfeinern kostet fast nichts, Vergroebern verliert Struktur.")
        self.form = QtWidgets.QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.spins: Dict[str, List[QtWidgets.QSpinBox]] = {}

    def build(self, control_points: Dict[str, Sequence[int]]) -> None:
        """One row per group; the rows are rebuilt when a different run is opened."""

        self.spins.clear()
        while self.form.rowCount() > 0:
            self.form.removeRow(0)
        for name, points in control_points.items():
            holder = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(holder)
            layout.setContentsMargins(0, 0, 0, 0)
            spins = []
            for index, value in enumerate(points):
                spin = int_spin(int(value), 2, 200)
                layout.addWidget(spin)
                if index == 0:
                    layout.addWidget(QtWidgets.QLabel("x"))
                spins.append(spin)
            self.spins[name] = spins
            self.form.addRow(name, holder)
        self.apply_btn = QtWidgets.QPushButton("uebernehmen (interpoliert)")
        self.apply_btn.clicked.connect(self.apply_clicked.emit)
        self.form.addRow("", self.apply_btn)

    def values(self) -> Dict[str, List[int]]:
        return {name: [int(s.value()) for s in spins] for name, spins in self.spins.items()}
