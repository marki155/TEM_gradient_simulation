r"""Part V, Section 7: the numbers that are edited by hand.

Theory
======

A reconstruction has a few dozen numbers that a person may want to set directly: the
amplitude contrast, the energy spread, :math:`C_c`, the aberration coefficients, the
two normalisation constants of the image, and for every group its shift, rotation,
scale and contrast.  The core publishes exactly these under flat keys
(:meth:`temsim.service.session.Session.values`), always as magnitude and angle for the
aberrations, whatever it uses internally.

:class:`ValuesEditor` builds one spin box per component from the dictionary it is
given, so that new groups or aberrations appear without a change here, and writes a
changed key back with ``set_values``.  The values keep changing while the optimiser
runs, so :meth:`ValuesEditor.refresh` shows new numbers; a spin box that has the focus
is left alone, so that typing is never overwritten.

Implementation
==============
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

from .connection import Connection
from .widgets import CollapsibleSection, double_spin, set_quietly

DECIMALS = 6
LIMIT = 1.0e9


def _components(value: Any) -> List[float]:
    return [float(v) for v in np.atleast_1d(np.asarray(value, dtype=float))]


def _pairs(keys: List[str]) -> List[tuple]:
    """``[(magnitude key, angle key or None)]`` of the aberrations, in the given order."""

    names = [k[len("optics."):] for k in keys if k.startswith("optics.")]
    scalars = {"amp_contrast", "delta_E", "Cc_um"}
    out = []
    for name in names:
        if name in scalars or name.startswith("phi"):
            continue
        angle = "phi" + name[1:]
        out.append((f"optics.{name}", f"optics.{angle}" if angle in names else None))
    return out


class ValuesEditor(QtWidgets.QFrame):
    """Spin boxes for :meth:`Session.values`; ``save``, ``save_as`` and ``load`` are asked of the window."""

    save_clicked = QtCore.Signal()
    save_as_clicked = QtCore.Signal()
    load_clicked = QtCore.Signal()
    changed = QtCore.Signal()

    def __init__(self, connection: Connection, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.connection = connection
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.spins: Dict[str, List[QtWidgets.QDoubleSpinBox]] = {}
        self._keys: List[str] = []
        self._build_frame()

    def _build_frame(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        buttons = QtWidgets.QHBoxLayout()
        for label, signal in (("Laden ...", self.load_clicked), ("Speichern", self.save_clicked),
                              ("Speichern unter ...", self.save_as_clicked)):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda _=False, s=signal: s.emit())
            buttons.addWidget(button)
        outer.addLayout(buttons)
        self.section = CollapsibleSection("Parameter", expanded=False)
        outer.addWidget(self.section)
        self.body = QtWidgets.QHBoxLayout()
        self.section.content_layout.addLayout(self.body)

    # ------------------------------------------------------------------ building
    def build(self, values: Dict[str, Any]) -> None:
        """Replace all widgets by those of ``values``."""

        while self.body.count():
            item = self.body.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.spins.clear()
        self._keys = list(values)
        keys = self._keys
        optics = QtWidgets.QGroupBox("Optik")
        form = QtWidgets.QFormLayout(optics)
        for key, label in (("optics.amp_contrast", "amp_contrast"), ("optics.delta_E", "delta_E"),
                           ("optics.Cc_um", "Cc [um]")):
            if key in values:
                form.addRow(label, self._spin(key, 0))
        for magnitude, angle in _pairs(keys):
            row = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._spin(magnitude, 0))
            if angle is not None:
                layout.addWidget(QtWidgets.QLabel(angle.split(".", 1)[1] + " [rad]"))
                layout.addWidget(self._spin(angle, 0))
            form.addRow(magnitude.split(".", 1)[1], row)
        self.body.addWidget(optics)
        if "normalisation" in values:
            box = QtWidgets.QGroupBox("Normierung")
            norm = QtWidgets.QFormLayout(box)
            norm.addRow("a / b", self._vector("normalisation", 2))
            self.body.addWidget(box)
        for name in self._group_names(keys):
            box = QtWidgets.QGroupBox(name)
            form = QtWidgets.QFormLayout(box)
            form.addRow("shift", self._vector(f"groups.{name}.shift", 2))
            form.addRow("rotation", self._vector(f"groups.{name}.rotation", 1))
            form.addRow("scale", self._vector(f"groups.{name}.scale", 2))
            form.addRow("contrast", self._vector(f"groups.{name}.contrast", 1))
            self.body.addWidget(box)
        self.body.addStretch()
        self.refresh(values)

    @staticmethod
    def _group_names(keys: List[str]) -> List[str]:
        names: List[str] = []
        for key in keys:
            if key.startswith("groups."):
                name = key.split(".")[1]
                if name not in names:
                    names.append(name)
        return names

    def _spin(self, key: str, index: int) -> QtWidgets.QDoubleSpinBox:
        spin = double_spin(0.0, -LIMIT, LIMIT, 0.01, DECIMALS)
        spin.setMinimumWidth(110)
        spin.valueChanged.connect(lambda _=None, k=key: self._write(k))
        self.spins.setdefault(key, []).append(spin)
        return spin

    def _vector(self, key: str, n: int) -> QtWidgets.QWidget:
        holder = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        for i in range(n):
            layout.addWidget(self._spin(key, i))
        return holder

    # ------------------------------------------------------------------ talking to the core
    def _write(self, key: str) -> None:
        values = [spin.value() for spin in self.spins[key]]
        payload = values if len(values) > 1 else values[0]
        self.connection.submit(lambda: self.connection.client.set_values(**{key: payload}))
        self.changed.emit()

    def reload(self) -> None:
        """Ask the core for the numbers and show them."""

        self.connection.submit(self.connection.client.get_values, self.show_values)

    def show_values(self, values: Dict[str, Any]) -> None:
        if list(values) != self._keys:
            self.build(values)
        else:
            self.refresh(values)

    def refresh(self, values: Dict[str, Any]) -> None:
        """Show ``values``; a box that has the focus keeps what is being typed."""

        for key, spins in self.spins.items():
            if key not in values:
                continue
            for spin, component in zip(spins, _components(values[key])):
                if not spin.hasFocus():
                    set_quietly(spin, component)

    def current(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, spins in self.spins.items():
            data = [spin.value() for spin in spins]
            out[key] = data if len(data) > 1 else data[0]
        return out
