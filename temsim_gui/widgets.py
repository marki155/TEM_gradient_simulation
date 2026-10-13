r"""Part V, Section 6: small building blocks of the panels.

Theory
======

None of the panels needs more than a handful of widget kinds: a spin box with a range, a
section that can be folded away, and a box with a form inside.  They are made here once,
so that the panels read as a list of *what* is shown, not *how*.

Implementation
==============
"""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6 import QtCore, QtWidgets


class CollapsibleSection(QtWidgets.QFrame):
    """A titled frame whose content can be folded away."""

    toggled = QtCore.Signal(bool)

    def __init__(self, title: str, expanded: bool = True, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.button = QtWidgets.QToolButton()
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.button.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.button.clicked.connect(self._toggle)
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(6, 6, 6, 6)
        header.addWidget(self.button)
        header.addStretch()
        self.content = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 4, 8, 8)
        self.content.setVisible(expanded)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addLayout(header)
        outer.addWidget(self.content)

    def _toggle(self, checked: bool) -> None:
        self.button.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self.content.setVisible(checked)
        self.toggled.emit(checked)


def double_spin(value: float = 0.0, low: float = 0.0, high: float = 1.0, step: float = 0.05, decimals: int = 2,
                suffix: str = "", prefix: str = "", tip: str = "") -> QtWidgets.QDoubleSpinBox:
    spin = QtWidgets.QDoubleSpinBox()
    spin.setDecimals(decimals)
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setValue(value)
    if suffix:
        spin.setSuffix(suffix)
    if prefix:
        spin.setPrefix(prefix)
    if tip:
        spin.setToolTip(tip)
    return spin


def int_spin(value: int = 0, low: int = 0, high: int = 1_000_000, step: int = 1, tip: str = "") -> QtWidgets.QSpinBox:
    spin = QtWidgets.QSpinBox()
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setValue(value)
    if tip:
        spin.setToolTip(tip)
    return spin


def set_quietly(widget: QtWidgets.QWidget, value) -> None:
    """Show ``value`` without emitting the widget's change signals."""

    widget.blockSignals(True)
    try:
        if isinstance(widget, QtWidgets.QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QtWidgets.QLineEdit):
            widget.setText(str(value))
        else:
            widget.setValue(value)
    finally:
        widget.blockSignals(False)


def row_of(*widgets: QtWidgets.QWidget, margins: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> QtWidgets.QWidget:
    holder = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(holder)
    layout.setContentsMargins(*margins)
    for widget in widgets:
        layout.addWidget(widget)
    return holder
