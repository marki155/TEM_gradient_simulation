r"""Part V, Section 8: what is trained, and how it is weighted.

Theory
======

The optimiser of Part II is steered by :class:`~temsim.data.schema.TrainSettings`: for
every trainable quantity a switch and a learning rate, the weights of the loss terms
(Part II, Section 1) and of the priors (Section 3), the flags that select single
aberrations and envelope terms, and the *active group* - the group whose mask and
target count and which, if the group asks for it, is simulated alone.

:class:`TrainEditor` shows these settings and changes them.  It keeps a *copy* of the
settings, built from the experiment and the settings that it is given, and sends every
change as the smallest command that expresses it:

* a switch or learning rate  ->  ``set_param``,
* a loss or prior weight     ->  ``set_loss`` / ``set_regularisation``,
* the active group           ->  ``set_active_group``,
* the flag dictionaries and the two presets, which change many things at once
  ->  ``set_train`` without the bulk arrays (the core keeps its masks and targets).

Two presets are common enough to have buttons.  :meth:`TrainEditor.train_only_group`
freezes everything except the rigid transformation (shift, rotation, scale) of one
group, makes that group active and keeps the optics free unless the group is simulated
in isolation - without the rest of the structure the optics cannot be determined.
:meth:`TrainEditor.free_positions_training` freezes everything except the positions
of the individual atoms of all layers.

Implementation
==============
"""

from __future__ import annotations

import copy
from dataclasses import fields
from typing import Callable, Dict, List, Optional

from PySide6 import QtCore, QtWidgets

from temsim.data.schema import Experiment, TrainSettings

from .connection import Connection
from .widgets import double_spin, set_quietly

MAIN_ENTRY = "Main (alle Gruppen)"


class TrainEditor(QtWidgets.QGroupBox):
    changed = QtCore.Signal()
    active_group_changed = QtCore.Signal(object)

    def __init__(self, connection: Connection, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__("TrainMethod", parent)
        self.connection = connection
        self.experiment: Optional[Experiment] = None
        self.settings = TrainSettings()
        self.rows: Dict[str, Dict[str, QtWidgets.QWidget]] = {}
        self.flags: Dict[str, Dict[str, QtWidgets.QCheckBox]] = {}
        self.weights: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        self.combo: Optional[QtWidgets.QComboBox] = None
        self.touched = False
        self.root = QtWidgets.QVBoxLayout(self)

    # ------------------------------------------------------------------ building
    def load(self, experiment: Experiment, settings: TrainSettings) -> None:
        """Show ``settings`` (a copy is kept) for ``experiment``."""

        self.experiment = experiment
        self.settings = copy.deepcopy(settings)
        self.settings.masks, self.settings.targets = {}, {}
        self.settings.ensure_for(experiment)
        self.touched = False
        while self.root.count():
            item = self.root.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.rows.clear(), self.flags.clear(), self.weights.clear()
        self._build_active_group()
        self._build_groups()
        self._build_layers()
        self._build_global()
        self._build_weights("Loss-Gewichte", "loss", self.settings.loss, self._send_loss)
        self._build_weights("Regularisierung", "regularisation", self.settings.regularisation, self._send_regularisation)
        self._build_flags("Aberrationen (Betrag + Winkel)", "aberration_flags")
        self._build_flags("Envelope-Parameter", "envelope_flags")

    def _group_names(self) -> List[str]:
        return self.experiment.group_names if self.experiment is not None else list(self.settings.groups)

    def _build_active_group(self) -> None:
        box = QtWidgets.QGroupBox("Aktive Gruppe")
        form = QtWidgets.QFormLayout(box)
        self.combo = QtWidgets.QComboBox()
        self.combo.addItem(MAIN_ENTRY, None)
        for name in self._group_names():
            self.combo.addItem(name, name)
        self.show_active_group()
        self.combo.currentIndexChanged.connect(self._on_active_group)
        form.addRow("Gruppe", self.combo)
        presets = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(presets)
        layout.setContentsMargins(0, 0, 0, 0)
        for name in self._group_names():
            button = QtWidgets.QPushButton(f"nur {name}")
            button.setToolTip(f"Nur die Geometrie der Gruppe {name} trainieren")
            button.clicked.connect(lambda _=False, n=name: self.train_only_group(n))
            layout.addWidget(button)
        free = QtWidgets.QPushButton("freie Positionen")
        free.setToolTip("Alle Gruppen fix, dafuer freie Atompositionen trainieren")
        free.clicked.connect(self.free_positions_training)
        layout.addWidget(free)
        form.addRow("Presets", presets)
        self.root.addWidget(box)

    def _build_groups(self) -> None:
        if not self.settings.groups:
            return
        box = QtWidgets.QGroupBox("Gruppen (Shift / Rotation / Scale / Strain)")
        outer = QtWidgets.QVBoxLayout(box)
        for gname, train in self.settings.groups.items():
            sub = QtWidgets.QGroupBox(gname)
            form = QtWidgets.QFormLayout(sub)
            for f in fields(train):
                form.addRow(f.name, self._param_row(f"groups.{gname}.{f.name}"))
            outer.addWidget(sub)
        self.root.addWidget(box)

    def _layer_title(self, name: str) -> str:
        if self.experiment is not None:
            for layer in self.experiment.layers:
                if layer.name == name:
                    return f"{layer.name}  ({layer.element}, z={layer.z_nm:g} nm, Gruppe {layer.group})"
        return name

    def _build_layers(self) -> None:
        if not self.settings.layers:
            return
        box = QtWidgets.QGroupBox("Schichten (Positionen / Intensitaeten / Occupancies)")
        outer = QtWidgets.QVBoxLayout(box)
        for lname, train in self.settings.layers.items():
            sub = QtWidgets.QGroupBox(self._layer_title(lname))
            form = QtWidgets.QFormLayout(sub)
            for f in fields(train):
                form.addRow(f.name, self._param_row(f"layers.{lname}.{f.name}"))
            outer.addWidget(sub)
        self.root.addWidget(box)

    def _build_global(self) -> None:
        box = QtWidgets.QGroupBox("Global")
        form = QtWidgets.QFormLayout(box)
        for name in TrainSettings.GLOBAL_PARAMS:
            form.addRow(name, self._param_row(name))
        self.root.addWidget(box)

    def _param_row(self, key: str) -> QtWidgets.QWidget:
        param = self.settings.param(key)
        holder = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        check = QtWidgets.QCheckBox()
        check.setChecked(bool(param.enabled))
        spin = double_spin(float(param.lr), 0.0, 1e12, 1e-5, 10)
        spin.setMinimumWidth(120)
        check.stateChanged.connect(lambda _=None, k=key, w=check: self._on_enabled(k, w.isChecked()))
        spin.valueChanged.connect(lambda _=None, k=key, w=spin: self._on_rate(k, float(w.value())))
        layout.addWidget(check, 0)
        layout.addWidget(spin, 1)
        self.rows[key] = {"enabled": check, "lr": spin}
        return holder

    def _build_weights(self, title: str, attr: str, target, send: Callable) -> None:
        box = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(box)
        for f in fields(target):
            spin = double_spin(float(getattr(target, f.name)), 0.0, 1e9, 0.05, 6)
            spin.valueChanged.connect(lambda value, n=f.name, a=attr: send(a, n, float(value)))
            form.addRow(f.name, spin)
            self.weights[f"{attr}.{f.name}"] = spin
        self.root.addWidget(box)

    def _build_flags(self, title: str, attr: str) -> None:
        flags = getattr(self.settings, attr)
        if not flags:
            return
        box = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(box)
        self.flags[attr] = {}
        for key, value in flags.items():
            check = QtWidgets.QCheckBox()
            check.setChecked(bool(value))
            check.stateChanged.connect(lambda _=None, a=attr, k=key, w=check: self._on_flag(a, k, w.isChecked()))
            label = f"{key} (+ phi{key[1:]})" if attr == "aberration_flags" and key[1:2] != "" and key[-1:] != "0" else key
            form.addRow(label, check)
            self.flags[attr][key] = check
        self.root.addWidget(box)

    # ------------------------------------------------------------------ changes
    def _send(self, call: Callable) -> None:
        self.touched = True
        self.connection.submit(call)
        self.changed.emit()

    def _on_enabled(self, key: str, value: bool) -> None:
        self.settings.param(key).enabled = bool(value)
        self._send(lambda: self.connection.client.set_param(key, enabled=bool(value)))

    def _on_rate(self, key: str, value: float) -> None:
        self.settings.param(key).lr = float(value)
        self._send(lambda: self.connection.client.set_param(key, lr=float(value)))

    def _send_loss(self, attr: str, name: str, value: float) -> None:
        setattr(self.settings.loss, name, value)
        self._send(lambda: self.connection.client.set_loss(**{name: value}))

    def _send_regularisation(self, attr: str, name: str, value: float) -> None:
        setattr(self.settings.regularisation, name, value)
        self._send(lambda: self.connection.client.set_regularisation(**{name: value}))

    def _on_flag(self, attr: str, key: str, value: bool) -> None:
        getattr(self.settings, attr)[key] = bool(value)
        self.push()

    def _on_active_group(self, _index: int) -> None:
        name = self.combo.currentData()
        self.settings.active_group = name
        self._send(lambda: self.connection.client.set_active_group(name))
        self.active_group_changed.emit(name)

    def push(self) -> None:
        """Send the whole local copy (without masks and targets)."""

        settings = copy.deepcopy(self.settings)
        self._send(lambda: self.connection.client.set_train(settings, with_arrays=False))

    # ------------------------------------------------------------------ showing
    def show_active_group(self) -> None:
        index = self.combo.findData(self.settings.active_group)
        set_quietly_index(self.combo, max(index, 0))

    def refresh(self) -> None:
        """Show the local copy in the widgets."""

        self.show_active_group()
        for key, row in self.rows.items():
            param = self.settings.param(key)
            if param is not None:
                set_quietly(row["enabled"], param.enabled)
                set_quietly(row["lr"], float(param.lr))
        for attr, boxes in self.flags.items():
            for key, check in boxes.items():
                set_quietly(check, getattr(self.settings, attr).get(key, False))
        for key, spin in self.weights.items():
            attr, name = key.split(".", 1)
            set_quietly(spin, float(getattr(getattr(self.settings, attr), name)))

    def adopt(self, settings: TrainSettings) -> None:
        """Take over settings that the core reports (e.g. after loading a run)."""

        keep = self.touched
        self.settings = copy.deepcopy(settings)
        self.settings.masks, self.settings.targets = {}, {}
        if self.experiment is not None:
            self.settings.ensure_for(self.experiment)
        self.refresh()
        self.touched = keep

    def reload(self) -> None:
        self.connection.submit(self.connection.client.get_train, self.adopt)

    # ------------------------------------------------------------------ presets
    def _disable_all(self) -> None:
        for param in self.settings.params().values():
            param.enabled = False

    def train_only_group(self, group: str) -> None:
        """Freeze everything but the rigid transformation of ``group`` (module docstring)."""

        s = self.settings
        self._disable_all()
        train = s.group(group)
        train.shift.enabled = train.rotation.enabled = train.scale.enabled = True
        s.active_group = group
        isolated = False
        if self.experiment is not None:
            try:
                isolated = bool(self.experiment.group(group).isolate_when_active)
            except KeyError:
                isolated = False
        s.aberrations.enabled = s.envelope.enabled = not isolated
        s.normalisation.enabled = True
        self.refresh()
        self.push()
        self.active_group_changed.emit(group)

    def free_positions_training(self) -> None:
        """Freeze everything but the individual atom positions of all layers."""

        s = self.settings
        self._disable_all()
        for layer in s.layers.values():
            layer.free_positions.enabled = True
        s.active_group = None
        s.normalisation.enabled = True
        self.refresh()
        self.push()
        self.active_group_changed.emit(None)

    def enable_after_seeding(self) -> None:
        """Seeded atoms start out free: release their positions and intensities."""

        for layer in self.settings.layers.values():
            layer.free_positions.enabled = True
            layer.intensities.enabled = True
        self.refresh()
        self.push()


def set_quietly_index(combo: QtWidgets.QComboBox, index: int) -> None:
    combo.blockSignals(True)
    combo.setCurrentIndex(index)
    combo.blockSignals(False)
