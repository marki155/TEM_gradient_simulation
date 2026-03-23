from __future__ import annotations

from dataclasses import fields, is_dataclass
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# -----------------------------
# Kleine Spoiler-/Collapsible-Section
# -----------------------------
class CollapsibleSection(QFrame):
    toggled = Signal(bool)

    def __init__(self, title: str, parent: QWidget | None = None, expanded: bool = True):
        super().__init__(parent)

        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)

        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

        self.toggle_button.clicked.connect(self._on_toggled)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(6, 6, 6, 6)
        header_layout.addWidget(self.toggle_button)
        header_layout.addStretch()

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 4, 8, 8)
        self.content.setVisible(expanded)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addLayout(header_layout)
        main_layout.addWidget(self.content)

    def _on_toggled(self, checked: bool):
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content.setVisible(checked)
        self.toggled.emit(checked)


# -----------------------------
# Widget für genau ein Tensor-/Parameter-Objekt
# -----------------------------
class TensorEditor(QFrame):
    """
    Editor für nn.Parameter / Tensor.
    Schreibt Änderungen direkt in den Tensor.
    Kann Skalar- und beliebige Tensorformen editieren.
    """

    def __init__(
        self,
        label: str,
        tensor: torch.Tensor,
        parent: QWidget | None = None,
        decimals: int = 6,
        step: float = 0.01,
    ):
        super().__init__(parent)
        self.label_text = label
        self.tensor = tensor
        self.spinboxes: list[QDoubleSpinBox] = []

        self._building = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QLabel(f"<b>{label}</b>")
        layout.addWidget(title)

        self.grid = QGridLayout()
        self.grid.setContentsMargins(12, 0, 0, 0)
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(4)
        layout.addLayout(self.grid)

        self._build_ui(decimals=decimals, step=step)
        self.refresh_from_tensor()

    def _build_ui(self, decimals: int, step: float):
        self._building = True

        cpu_tensor = self.tensor.detach().cpu()
        shape = tuple(cpu_tensor.shape)

        # Skalar
        if cpu_tensor.ndim == 0:
            sb = self._make_spinbox(decimals, step)
            sb.valueChanged.connect(partial(self._write_scalar))
            self.grid.addWidget(QLabel("value"), 0, 0)
            self.grid.addWidget(sb, 0, 1)
            self.spinboxes.append(sb)
            self._building = False
            return

        # Allgemein: flatten + Indexlabel
        flat_count = cpu_tensor.numel()
        for flat_idx in range(flat_count):
            multi_idx = np.unravel_index(flat_idx, shape)
            idx_text = "[" + ", ".join(str(i) for i in multi_idx) + "]"

            sb = self._make_spinbox(decimals, step)
            sb.valueChanged.connect(partial(self._write_at_index, multi_idx))

            row = flat_idx
            self.grid.addWidget(QLabel(idx_text), row, 0)
            self.grid.addWidget(sb, row, 1)
            self.spinboxes.append(sb)

        self._building = False

    def _make_spinbox(self, decimals: int, step: float) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setDecimals(decimals)
        sb.setRange(-1e12, 1e12)
        sb.setSingleStep(step)
        sb.setKeyboardTracking(False)
        sb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return sb

    def _write_scalar(self, value: float):
        if self._building:
            return
        with torch.no_grad():
            self.tensor.copy_(torch.tensor(value, dtype=self.tensor.dtype, device=self.tensor.device))

    def _write_at_index(self, index: tuple[int, ...], value: float):
        if self._building:
            return
        with torch.no_grad():
            self.tensor[index] = value

    def refresh_from_tensor(self):
        self._building = True
        cpu_tensor = self.tensor.detach().cpu()

        if cpu_tensor.ndim == 0:
            self.spinboxes[0].setValue(float(cpu_tensor.item()))
        else:
            flat = cpu_tensor.reshape(-1).tolist()
            for sb, val in zip(self.spinboxes, flat):
                sb.setValue(float(val))

        self._building = False


# -----------------------------
# Dynamisch generiertes Root-Frame
# -----------------------------
class ParametersFrame(QFrame):
    """
    Baut die UI dynamisch aus einer verschachtelten Dataclass-Struktur auf.
    Unterstützt:
      - verschachtelte Dataclasses
      - nn.Parameter
      - nn.ParameterList
      - nn.ParameterDict
    """

    def __init__(self, params_root: Any, parent: QWidget | None = None):
        super().__init__(parent)

        if not is_dataclass(params_root):
            raise TypeError("params_root muss eine Dataclass-Instanz sein.")

        self.params_root = params_root
        self._tensor_editors: list[TensorEditor] = []

        self.setFrameShape(QFrame.StyledPanel)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll)

        container = QWidget()
        self.main_layout = QVBoxLayout(container)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(8)

        scroll.setWidget(container)

        root_section = CollapsibleSection(type(params_root).__name__, expanded=True)
        self.main_layout.addWidget(root_section)

        self._build_dataclass_into_layout(root_section.content_layout, params_root, path=type(params_root).__name__)
        self.main_layout.addStretch()

    # -------------------------
    # Public API
    # -------------------------
    def refresh_from_params(self):
        for editor in self._tensor_editors:
            editor.refresh_from_tensor()

    # -------------------------
    # Internals
    # -------------------------
    def _build_dataclass_into_layout(self, layout: QVBoxLayout, obj: Any, path: str):
        for f in fields(obj):
            name = f.name
            value = getattr(obj, name)
            field_path = f"{path}.{name}"

            self._build_value(layout, name=name, value=value, path=field_path)

    def _build_value(self, layout: QVBoxLayout, name: str, value: Any, path: str):
        # Verschachtelte Dataclass -> Section
        if is_dataclass(value):
            section = CollapsibleSection(name, expanded=True)
            layout.addWidget(section)
            self._build_dataclass_into_layout(section.content_layout, value, path)
            return

        # Einzelner Parameter
        if isinstance(value, nn.Parameter):
            editor = TensorEditor(name, value)
            self._tensor_editors.append(editor)
            layout.addWidget(editor)
            return

        # ParameterList
        if isinstance(value, nn.ParameterList):
            section = CollapsibleSection(name, expanded=False)
            layout.addWidget(section)

            for i, param in enumerate(value):
                if not isinstance(param, nn.Parameter):
                    continue
                editor = TensorEditor(f"{name}[{i}]", param)
                self._tensor_editors.append(editor)
                section.content_layout.addWidget(editor)
            return

        # ParameterDict
        if isinstance(value, nn.ParameterDict):
            section = CollapsibleSection(name, expanded=False)
            layout.addWidget(section)

            for key, param in value.items():
                if not isinstance(param, nn.Parameter):
                    continue
                editor = TensorEditor(str(key), param)
                self._tensor_editors.append(editor)
                section.content_layout.addWidget(editor)
            return

        # Fallback: nur anzeigen
        row = QHBoxLayout()
        row.addWidget(QLabel(f"{name}:"))
        row.addWidget(QLabel(repr(value)))
        row.addStretch()
        wrapper = QWidget()
        wrapper.setLayout(row)
        layout.addWidget(wrapper)


# -----------------------------
# Demo
# -----------------------------
if __name__ == "__main__":
    from dataclasses import dataclass, field

    @dataclass
    class StrainTunableParameters:
        control: nn.Parameter = field(
            default_factory=lambda: nn.Parameter(torch.tensor([0.0, 1.0], dtype=torch.float32))
        )

    @dataclass
    class TunablePotentialParameters:
        positions: nn.ParameterList = field(
            default_factory=lambda: nn.ParameterList([
                nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32)),
                nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float32)),
            ])
        )
        intensities: nn.ParameterList = field(
            default_factory=lambda: nn.ParameterList([
                nn.Parameter(torch.tensor(0.5, dtype=torch.float32)),
                nn.Parameter(torch.tensor(1.5, dtype=torch.float32)),
            ])
        )
        occupancies: nn.ParameterList = field(
            default_factory=lambda: nn.ParameterList([
                nn.Parameter(torch.tensor(1.0, dtype=torch.float32)),
            ])
        )

        scale_Gr: nn.Parameter = field(
            default_factory=lambda: nn.Parameter(torch.tensor([1.12, 1.14], dtype=torch.float32))
        )
        rotation_Gr: nn.Parameter = field(
            default_factory=lambda: nn.Parameter(torch.tensor(0.025, dtype=torch.float32))
        )
        shift_Gr: nn.Parameter = field(
            default_factory=lambda: nn.Parameter(torch.tensor([-9.5, -11.0], dtype=torch.float32))
        )

        scale_WS2: nn.Parameter = field(
            default_factory=lambda: nn.Parameter(torch.tensor([1.05, 1.09], dtype=torch.float32))
        )
        rotation_WS2: nn.Parameter = field(
            default_factory=lambda: nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        )
        shift_WS2: nn.Parameter = field(
            default_factory=lambda: nn.Parameter(torch.tensor([15.9, -5.0], dtype=torch.float32))
        )

        blur_sigma: nn.Parameter = field(
            default_factory=lambda: nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        )

        ws2_displacement: StrainTunableParameters = field(default_factory=StrainTunableParameters)

    @dataclass
    class TunableCTFParameters:
        amp_contrast: nn.Parameter = field(
            default_factory=lambda: nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        )
        deltaE: nn.Parameter = field(
            default_factory=lambda: nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        )
        Cc_um: nn.Parameter = field(
            default_factory=lambda: nn.Parameter(torch.tensor(1.0e3, dtype=torch.float32))
        )
        aberrations: nn.ParameterDict = field(
            default_factory=lambda: nn.ParameterDict({
                "C10": nn.Parameter(torch.tensor(-13.0, dtype=torch.float32)),
                "C12": nn.Parameter(torch.tensor(1.0, dtype=torch.float32)),
                "phi12": nn.Parameter(torch.tensor(0.0, dtype=torch.float32)),
            })
        )

    @dataclass
    class TunableMultisliceParameters:
        sigma: nn.Parameter = field(
            default_factory=lambda: nn.Parameter(torch.tensor(0.0177, dtype=torch.float32))
        )

    @dataclass
    class ParticlesTunableParameters:
        potential_parameters: TunablePotentialParameters
        ctf_parameters: TunableCTFParameters
        multislice_parameters: TunableMultisliceParameters

    app = QApplication([])

    params = ParticlesTunableParameters(
        potential_parameters=TunablePotentialParameters(),
        ctf_parameters=TunableCTFParameters(),
        multislice_parameters=TunableMultisliceParameters(),
    )

    frame = ParametersFrame(params)
    frame.resize(700, 900)
    frame.show()

    app.exec()