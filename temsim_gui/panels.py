r"""Part V, Section 9: the control panels.

Theory
======

The right-hand column of the window consists of four panels.  A panel is *only* a set of
widgets: it knows its defaults, how to show a state and how to hand its settings
out, but nothing about the core; the :class:`~temsim_gui.controller.Controller` connects
panels, canvas and connection.  This is what makes the panels testable without a core
and keeps the interface replaceable: a new button needs a widget here and one line there.

* :class:`ControlPanel` - state, training buttons, view options, resets, gradient
  arrows, removal of weak atoms.
* :class:`MinimaPanel` - candidate sites from the minima of the difference image
  (Part II, Section 5) and the checks that a candidate has to pass.
* :class:`SeedingPanel` - the options of the random seeding; the numbers are those of
  :class:`~temsim.learning.seeding.seeder.SeedConfig` plus the three job options.
* :class:`StrainPanel` - the number of control points of the displacement field of every
  group.

Implementation
==============
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtWidgets

from .widgets import double_spin, int_spin, set_quietly


class ControlPanel(QtWidgets.QWidget):
    """Status, buttons and view options."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._status_box())
        layout.addWidget(self._control_box())
        self.set_state("idle")

    def _status_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Status")
        form = QtWidgets.QFormLayout(box)
        self.state_label = QtWidgets.QLabel("idle")
        self.step_label = QtWidgets.QLabel("step: -")
        self.loss_label = QtWidgets.QLabel("loss: -")
        form.addRow("State", self.state_label)
        form.addRow("Step", self.step_label)
        form.addRow("Loss", self.loss_label)
        return box

    def _control_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Training control")
        grid = QtWidgets.QGridLayout(box)
        button = QtWidgets.QPushButton
        self.start_btn, self.pause_btn, self.resume_btn = button("Start"), button("Pause"), button("Resume")
        self.stop_btn, self.render_btn, self.seed_btn = button("Stop"), button("Render"), button("Start Seeding")
        self.seed_btn.setToolTip("Atome zufaellig einfuegen und behalten, wenn die Korrelation besser wird.\n"
                                 "Pause/Resume/Stop gelten auch hierfuer.")
        self.tile_combo = QtWidgets.QComboBox()
        self.tile_combo.setToolTip("Eine Kachel gross zeigen - Simulation, Diff, Target, CTF, Potential ...\n"
                                   "Zoomen und Verschieben mit der Toolbar unter dem Bild.")
        self.show_atoms = QtWidgets.QCheckBox("Show atom positions on target")
        self.show_atoms.setChecked(True)
        self.reset_occ_btn = button("Reset occupancies (aktive Gruppe)")
        self.reset_int_btn = button("Reset intensities (aktive Gruppe)")
        self.reset_int_btn.setToolTip("Intensitaeten zurueck auf die Startwerte -\n"
                                      "nicht auf 0, die Intensitaet ist die Ordnungszahl.")
        self.reset_free_btn = button("Reset free positions (aktive Gruppe)")
        self.reset_losses_btn = button("Reset losses")
        self.reset_zoom_btn = button("Zoom zuruecksetzen")
        self.remove_btn = button("Atome unter Occ entfernen")
        self.remove_btn.setToolTip("Loescht alle Atome der aktiven Gruppe, deren sigmoid(Occupancy) unter dem Wert\n"
                                   "rechts liegt, endgueltig aus dem Modell (Adam-Zustand wird zurueckgesetzt).")
        self.remove_spin = double_spin(0.5, 0.0, 1.0, 0.05, 2, prefix="Occ < ")
        self.show_occ_spin = double_spin(0.5, 0.0, 1.0, 0.05, 2, prefix="Anzeige ab Occ ",
                                         tip="Nur Atome mit sigmoid(Occupancy) ab diesem Wert werden angezeigt "
                                             "(Positionen, Pfeile,\nHistogramm). 0 = alle. Das Modell bleibt unveraendert.")
        self.save_on_close = QtWidgets.QCheckBox("Beim Schliessen speichern")
        self.save_on_close.setChecked(True)
        self.save_on_close.setToolTip("Speichert beim Schliessen des Fensters den Stand im Ausgabeordner des Kerns.")
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
        for b in (self.pause_btn, self.resume_btn, self.stop_btn):
            b.setEnabled(False)
        for row, widget in enumerate((self.start_btn, self.pause_btn, self.resume_btn, self.stop_btn,
                                      self.render_btn, self.seed_btn)):
            grid.addWidget(widget, row, 0)
        for row, widget in enumerate((self.tile_combo, self.show_atoms, self.reset_occ_btn, self.reset_int_btn,
                                      self.reset_free_btn, self.reset_losses_btn, self.reset_zoom_btn)):
            grid.addWidget(widget, row, 1)
        grid.addWidget(self.gradients, 6, 0)
        row = QtWidgets.QHBoxLayout()
        for widget in (self.descent, self.length_spin, self.relative):
            row.addWidget(widget)
        grid.addLayout(row, 7, 0, 1, 2)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.remove_btn)
        row.addWidget(self.remove_spin)
        grid.addLayout(row, 8, 0, 1, 2)
        grid.addWidget(self.show_occ_spin, 9, 0, 1, 2)
        grid.addWidget(self.save_on_close, 10, 0, 1, 2)
        return box

    # ------------------------------------------------------------------ showing
    def set_state(self, state: str, job: Optional[str] = None, open_: bool = True) -> None:
        """Label and enabled buttons for ``state`` in ``idle | running | paused``."""

        text = state if job is None or state == "idle" else f"{state} ({job})"
        self.state_label.setText(text)
        idle, running, paused = state == "idle", state == "running", state == "paused"
        self.start_btn.setEnabled(idle and open_)
        self.seed_btn.setEnabled(idle and open_)
        self.pause_btn.setEnabled(running)
        self.resume_btn.setEnabled(paused)
        self.stop_btn.setEnabled(running or paused)

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


class MinimaPanel(QtWidgets.QGroupBox):
    """Candidate atom sites from the minima of the difference image."""

    CHECKS = (("margin", "Rand beachten", True, "Minima naeher am Bildrand als 'Rand [px]' entfallen."),
              ("collisions", "Kollisionsregel", True,
               "Atome naeher als 'Entfernen bis' werden beim Einfuegen entfernt,\n"
               "Atome bis 'Verschieben bis' ziehen das neue Atom in die Mitte."),
              ("loss", "Loss darf nicht schlechter werden", True,
               "Ein Minimum zaehlt nur, wenn das neue Atom den Loss nicht verschlechtert."))
    SPINS = (("touch_A", "Entfernen bis [A]", 0.5, "Atome naeher als das werden beim Einfuegen entfernt."),
             ("merge_A", "Verschieben bis [A]", 1.0, "Bis hierher rutscht das Atom mit dem Nachbarn in die Mitte."),
             ("margin_px", "Rand [px]", 0.0, "Mindestabstand zum Bildrand."))

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__("Diff-Minima als neue Atome", parent)
        grid = QtWidgets.QGridLayout(self)
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
        self.spins: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        grid.addWidget(self.show_check, 0, 0, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Schwelle (x tiefstes Minimum)"), 1, 0)
        grid.addWidget(self.threshold, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Mindestabstand"), 2, 0)
        grid.addWidget(self.distance, 2, 1)
        row = 3
        for widget in (*self.checks.values(), self.atom_distance):
            grid.addWidget(widget, row, 0, 1, 2)
            row += 1
        for key, label, default, tip in self.SPINS:
            spin = double_spin(default, 0.0, 100.0, 0.05, 2, tip=tip)
            self.spins[key] = spin
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            grid.addWidget(spin, row, 1)
            row += 1
        self.save_check = QtWidgets.QCheckBox("Nach dem Einfuegen speichern")
        self.save_check.setChecked(True)
        self.insert_btn = QtWidgets.QPushButton("Minima als Atome einfuegen")
        self.insert_btn.setToolTip("Setzt an jedem gezeigten Minimum ein neues Atom in die aktive Gruppe\n"
                                   "(Occupancy/Intensitaet wie beim Seeding).")
        grid.addWidget(self.save_check, row, 0, 1, 2)
        grid.addWidget(self.insert_btn, row + 1, 0, 1, 2)

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


class SeedingPanel(QtWidgets.QGroupBox):
    """Options of the random seeding; changes go to the core at once, also while it runs."""

    #: (key, label, minimum, maximum, step, decimals, tooltip); keys of SeedConfig or of the job
    FIELDS = (
        ("touch_A", "Entfernen bis [A]", 0.0, 10.0, 0.05, 2, "Atome, die naeher als das am neuen Atom liegen, werden entfernt."),
        ("merge_A", "Verschieben bis [A]", 0.0, 10.0, 0.05, 2,
         "Zwischen 'Entfernen bis' und diesem Radius wird nicht neu eingefuegt:\n"
         "das Nachbaratom wandert in die Mitte zum neuen Punkt."),
        ("move_prob", "P(Atom verschieben)", 0.0, 1.0, 0.05, 2,
         "Anteil der Versuche, ein vorhandenes Atom zu verschieben statt eins einzufuegen."),
        ("remove_prob", "P(Atom entfernen)", 0.0, 1.0, 0.05, 2, "Anteil der Versuche, ein vorhandenes Atom wegzunehmen."),
        ("move_sigma_A", "Verschiebung sigma [A]", 0.0, 10.0, 0.05, 2, "Streuung der Gauss-Verschiebung eines Atoms."),
        ("gradient_temperature", "Gradient Temperatur", 0.01, 5.0, 0.05, 2,
         "Klein: fast immer die beste Stelle der Landschaft; gross: breiter gestreut."),
        ("relax_prob", "P(Einfuegen + Ausweichen)", 0.0, 1.0, 0.05, 2,
         "Anteil der Versuche: neues Atom an einer von einem Nachbarn blockierten Stelle, der Nachbar\n"
         "sucht sich den besten neuen Platz. Nur STEM."),
        ("relax_A", "Ausweich-Reichweite [A]", 0.0, 10.0, 0.05, 2, "Nachbaratome in diesem Abstand duerfen ausweichen."),
        ("bond_A", "Bindungslaenge [A]", 0.0, 10.0, 0.05, 2, "Wahrscheinlichster Abstand zum Nachbaratom."),
        ("bond_sigma_A", "Bindung sigma [A]", 0.01, 5.0, 0.05, 2, "Breite der Gauss-Verteilung um die Bindungslaenge."),
        ("neighbour_min_A", "Nachbar min. Abstand [A]", 0.0, 10.0, 0.05, 2, "Harter Cutoff nach unten."),
        ("neighbour_max_A", "Nachbar max. Abstand [A]", 0.0, 10.0, 0.05, 2, "Harter Cutoff nach oben."),
        ("max_atoms", "max. Atome (0 = frei)", 0, 1_000_000, 100, 0, "Seeding endet, sobald so viele Atome gesetzt sind."),
        ("trials", "max. Versuche", 1, 100_000_000, 1000, 0, ""),
        ("tune_every", "Training alle N Versuche (0 = aus)", 0, 1_000_000, 100, 0,
         "Zwischendurch ein paar Gradientenschritte auf Sonde/Mikroskop-Parametern."),
        ("tune_steps", "Schritte je Training", 1, 1000, 1, 0, ""),
    )
    JOB_KEYS = ("trials", "tune_every", "tune_steps")
    JOB_DEFAULTS = {"trials": 100_000, "tune_every": 0, "tune_steps": 3}

    config_changed = QtCore.Signal(dict)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__("Seeding-Parameter", parent)
        form = QtWidgets.QFormLayout(self)
        self.neighbour = QtWidgets.QCheckBox("Nachbar-Seeding")
        self.neighbour.setToolTip("Neue Atome der Reihe nach im Bindungsabstand um jedes vorhandene Atom setzen\n"
                                  "statt gleichverteilt ueber das Bild. Das erste Atom wird zufaellig gesetzt.")
        self.neighbour.toggled.connect(lambda on: self.config_changed.emit({"neighbour_seeding": bool(on)}))
        form.addRow(self.neighbour)
        self.gradient = QtWidgets.QCheckBox("Gradienten-Seeding (nur STEM)")
        self.gradient.setToolTip("Positionen nach der Loss-Landschaft fuers Einfuegen ziehen statt gleichverteilt.\n"
                                 "Angenommen wird weiterhin nur nach der exakten lokalen Loss.")
        self.gradient.toggled.connect(lambda on: self.config_changed.emit({"gradient_seeding": bool(on)}))
        form.addRow(self.gradient)
        self.weights = QtWidgets.QLineEdit("1")
        self.weights.setToolTip("Gewichte fuer neue Atome, getrennt durch Leerzeichen (z.B. 1 0.7 0.5 0.3).\n"
                                "Je Einfuegen gewinnt das Gewicht mit dem besten Loss.")
        self.weights.editingFinished.connect(self._on_weights)
        form.addRow("Gewichte", self.weights)
        self.spins: Dict[str, QtWidgets.QAbstractSpinBox] = {}
        for key, label, low, high, step, decimals, tip in self.FIELDS:
            spin = double_spin(low, low, high, step, decimals, tip=tip) if decimals else int_spin(int(low), int(low), int(high), int(step), tip)
            spin.valueChanged.connect(lambda value, k=key: self._on_value(k, value))
            form.addRow(label, spin)
            self.spins[key] = spin
        self.show_values(dict(self.JOB_DEFAULTS))

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
        if "neighbour_seeding" in config:
            set_quietly(self.neighbour, config["neighbour_seeding"])
        if "gradient_seeding" in config:
            set_quietly(self.gradient, config["gradient_seeding"])
        if "weight_options" in config:
            self._weights = tuple(config["weight_options"])
            set_quietly(self.weights, self._weights_text(self._weights))

    def job_options(self) -> Dict[str, int]:
        return {key: int(self.spins[key].value()) for key in self.JOB_KEYS}

    def set_stem(self, stem: bool) -> None:
        """Gradient seeding needs the STEM landscape (Part II, Section 5)."""

        self.gradient.setEnabled(stem)
        if not stem:
            set_quietly(self.gradient, False)


class StrainPanel(QtWidgets.QGroupBox):
    """Control points of the displacement field, per group."""

    apply_clicked = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__("Strain-Kontrollgitter", parent)
        self.setToolTip("Zahl der Kontrollpunkte je Gruppe. Beim Uebernehmen wird das vorhandene Feld interpoliert -\n"
                        "Verfeinern kostet fast nichts, Vergroebern verliert Struktur.")
        self.form = QtWidgets.QFormLayout(self)
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
