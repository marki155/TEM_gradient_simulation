from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass, asdict
from functools import partial
from typing import Any, Type

from nn import ParticlesTunableParameters

import numpy as np
import torch
import torch.nn as nn

from PySide6.QtWidgets import QFileDialog

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
    value_changed = Signal()

    #: Ab dieser Groesse wird nicht mehr jedes Element als Spinbox angezeigt.
    #: Sonst baut die GUI fuer eine Schicht mit 6000 Atomen zehntausende
    #: Widgets - das ist der Grund, warum das Fenster so traege war.
    MAX_INLINE_ELEMENTS = 64

    def __init__(
        self,
        label: str,
        tensor: torch.Tensor,
        parent: QWidget | None = None,
        decimals: int = 6,
        step: float = 0.01,
        max_elements: int | None = None,
    ):
        super().__init__(parent)
        self.label_text = label
        self.tensor = tensor
        self.spinboxes: list[QDoubleSpinBox] = []
        self.max_elements = self.MAX_INLINE_ELEMENTS if max_elements is None else max_elements
        self.summary_label: QLabel | None = None

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

        # Zu gross fuer einzelne Spinboxen -> nur eine Zusammenfassung zeigen
        flat_count = cpu_tensor.numel()
        if flat_count > self.max_elements:
            self.summary_label = QLabel()
            self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.grid.addWidget(self.summary_label, 0, 0, 1, 2)
            self._building = False
            return

        # Allgemein: flatten + Indexlabel
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
        self.value_changed.emit()

    def _write_at_index(self, index: tuple[int, ...], value: float):
        if self._building:
            return
        with torch.no_grad():
            self.tensor[index] = value
        self.value_changed.emit()

    def _summary_text(self, cpu_tensor) -> str:
        values = cpu_tensor.reshape(-1).to(torch.float32)
        shape = "x".join(str(s) for s in tuple(cpu_tensor.shape))
        return (f"{shape} ({values.numel()} Werte)   "
                f"min {float(values.min()):.4g}   "
                f"mittel {float(values.mean()):.4g}   "
                f"max {float(values.max()):.4g}")

    def refresh_from_tensor(self):
        self._building = True
        cpu_tensor = self.tensor.detach().cpu()

        if self.summary_label is not None:
            self.summary_label.setText(self._summary_text(cpu_tensor))
            self._building = False
            return

        if cpu_tensor.ndim == 0:
            self.spinboxes[0].blockSignals(True)
            self.spinboxes[0].setValue(float(cpu_tensor.item()))
            self.spinboxes[0].blockSignals(False)
        else:
            flat = cpu_tensor.reshape(-1).tolist()
            for sb, val in zip(self.spinboxes, flat):
                sb.blockSignals(True)
                sb.setValue(float(val))
                sb.blockSignals(False)

        self._building = False

from ctf import (
    ABERRATION_TERMS,
    PARAMETRIZATION_CARTESIAN,
    PARAMETRIZATION_POLAR,
    cartesian_keys,
    migrate_serialized_ctf_parameters,
)
from potential import migrate_serialized_potential_parameters


class AberrationEditor(QWidget):
    """Aberrationen als Betrag + Winkel, gespeichert wird kartesisch (a, b).

    Intern optimiert das Modell a = C*cos(m*phi) und b = C*sin(m*phi); von Hand
    einstellen will man aber C und phi. Dieses Widget rechnet in beide
    Richtungen um.
    """

    value_changed = Signal()

    def __init__(self, params, parent: QWidget | None = None, decimals: int = 6):
        super().__init__(parent)
        self.label_text = "aberrations"
        self.params = params
        self.rows = {}
        self._building = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(QLabel("<b>aberrations</b>"))

        grid = QGridLayout()
        grid.setContentsMargins(12, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        layout.addLayout(grid)

        row = 0
        for magnitude, (_power, _pref, _scale, m) in ABERRATION_TERMS.items():
            a_key, b_key = cartesian_keys(magnitude)

            if m and a_key in params and b_key in params:
                spin_c = self._make_spinbox(decimals)
                spin_phi = self._make_spinbox(decimals)
                grid.addWidget(QLabel(magnitude), row, 0)
                grid.addWidget(spin_c, row, 1)
                grid.addWidget(QLabel(f"phi{magnitude[1:]} [rad]"), row, 2)
                grid.addWidget(spin_phi, row, 3)
                spin_c.valueChanged.connect(partial(self._write_polar, magnitude))
                spin_phi.valueChanged.connect(partial(self._write_polar, magnitude))
                self.rows[magnitude] = {"kind": "polar", "m": m,
                                        "C": spin_c, "phi": spin_phi,
                                        "a": params[a_key], "b": params[b_key]}
                row += 1

            elif m == 0 and magnitude in params:
                spin_c = self._make_spinbox(decimals)
                grid.addWidget(QLabel(magnitude), row, 0)
                grid.addWidget(spin_c, row, 1)
                spin_c.valueChanged.connect(partial(self._write_scalar, magnitude))
                self.rows[magnitude] = {"kind": "scalar", "C": spin_c,
                                        "param": params[magnitude]}
                row += 1

        self.refresh_from_tensor()

    @staticmethod
    def has_cartesian_aberrations(params) -> bool:
        return any(cartesian_keys(magnitude)[0] in params
                   for magnitude, term in ABERRATION_TERMS.items() if term[3])

    def _make_spinbox(self, decimals: int) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setDecimals(decimals)
        sb.setRange(-1e12, 1e12)
        sb.setSingleStep(0.01)
        sb.setKeyboardTracking(False)
        sb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return sb

    def _write_polar(self, magnitude, _value=None):
        if self._building:
            return
        row = self.rows[magnitude]
        c = float(row["C"].value())
        phi = float(row["phi"].value())
        m = row["m"]
        with torch.no_grad():
            row["a"].copy_(torch.tensor(c * math.cos(m * phi), dtype=row["a"].dtype))
            row["b"].copy_(torch.tensor(c * math.sin(m * phi), dtype=row["b"].dtype))
        self.value_changed.emit()

    def _write_scalar(self, magnitude, _value=None):
        if self._building:
            return
        row = self.rows[magnitude]
        with torch.no_grad():
            row["param"].copy_(torch.tensor(float(row["C"].value()), dtype=row["param"].dtype))
        self.value_changed.emit()

    def refresh_from_tensor(self):
        self._building = True
        for row in self.rows.values():
            if row["kind"] == "scalar":
                row["C"].blockSignals(True)
                row["C"].setValue(float(row["param"].detach().cpu()))
                row["C"].blockSignals(False)
                continue

            a = float(row["a"].detach().cpu())
            b = float(row["b"].detach().cpu())
            for widget, value in ((row["C"], math.hypot(a, b)),
                                  (row["phi"], math.atan2(b, a) / row["m"])):
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)
        self._building = False


def serialize(obj):
    # --- Tensor ---
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    # --- Parameter ---
    elif isinstance(obj, torch.nn.Parameter):
        return obj.detach().cpu().tolist()

    # --- ParameterList ---
    elif isinstance(obj, torch.nn.ParameterList):
        return [serialize(p) for p in obj]

    # --- ParameterDict ---
    elif isinstance(obj, torch.nn.ParameterDict):
        return {k: serialize(v) for k, v in obj.items()}

    # --- Module (optional, falls enthalten) ---
    elif isinstance(obj, torch.nn.Module):
        return {k: serialize(v) for k, v in obj.state_dict().items()}

    # --- Dataclass ---
    elif is_dataclass(obj):
        return serialize(asdict(obj))

    # --- Standard Container ---
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [serialize(v) for v in obj]

    elif isinstance(obj, tuple):
        return tuple(serialize(v) for v in obj)

    # --- Fallback ---
    else:
        return obj


from dataclasses import fields, is_dataclass
from typing import Any
import torch
import torch.nn as nn


def update_from_serialized(target: Any, source: Any):
    # Dataclass rekursiv durchlaufen
    if is_dataclass(target):
        if not isinstance(source, dict):
            raise TypeError(f"Für Dataclass {type(target).__name__} wurde dict erwartet, bekam {type(source).__name__}")
        for f in fields(target):
            name = f.name
            if name not in source:
                continue
            update_from_serialized(getattr(target, name), source[name])
        return

    # Einzelner Parameter
    if isinstance(target, nn.Parameter):
        tensor = torch.tensor(source, dtype=target.dtype, device=target.device)
        with torch.no_grad():
            target.copy_(tensor)
        return

    # Normaler Tensor
    if isinstance(target, torch.Tensor):
        tensor = torch.tensor(source, dtype=target.dtype, device=target.device)
        with torch.no_grad():
            target.copy_(tensor)
        return

    # ParameterList
    if isinstance(target, nn.ParameterList):
        if len(target) != len(source):
            raise ValueError(f"ParameterList-Länge passt nicht: {len(target)} != {len(source)}")
        for t, s in zip(target, source):
            update_from_serialized(t, s)
        return

    # ParameterDict
    if isinstance(target, nn.ParameterDict):
        for key, value in source.items():
            if key not in target:
                raise KeyError(f"Key {key!r} fehlt in ParameterDict")
            update_from_serialized(target[key], value)
        return

    # list
    if isinstance(target, list):
        if len(target) != len(source):
            raise ValueError(f"Listenlänge passt nicht: {len(target)} != {len(source)}")
        for t, s in zip(target, source):
            update_from_serialized(t, s)
        return

    # dict
    if isinstance(target, dict):
        for key, value in source.items():
            if key not in target:
                raise KeyError(f"Key {key!r} fehlt im dict")
            update_from_serialized(target[key], value)
        return

    # Fallback für einfache skalare Attribute:
    # hier kann nicht "in place" aktualisiert werden, also ignorieren oder separat behandeln
def migrate_parameters_dict(data, params_root, config=None):
    """Bringt eine gespeicherte Parameter-Datei auf die aktuelle Struktur.

    Alte Dateien haben Gr/WS2-Felder im Potential und polare Aberrationen -
    beides wird hier auf die Form umgerechnet, die ``params_root`` erwartet.
    """

    if not isinstance(data, dict):
        return data

    data = dict(data)

    potential = data.get("potential_parameters")
    if isinstance(potential, dict) and config is not None:
        data["potential_parameters"] = migrate_serialized_potential_parameters(
            potential, config)

    ctf_parameters = data.get("ctf_parameters")
    if isinstance(ctf_parameters, dict):
        target = params_root.ctf_parameters.aberrations
        parametrization = (PARAMETRIZATION_CARTESIAN
                           if AberrationEditor.has_cartesian_aberrations(target)
                           else PARAMETRIZATION_POLAR)
        data["ctf_parameters"] = migrate_serialized_ctf_parameters(
            ctf_parameters, parametrization)

    return data


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
    parameter_tuned = Signal()


    def __init__(self, params_root: Any, parent: QWidget | None = None, config: Any = None):
        super().__init__(parent)

        if not is_dataclass(params_root):
            raise TypeError("params_root muss eine Dataclass-Instanz sein.")

        self.params_root = params_root
        #: ParticlesConfig - wird zum Migrieren aelterer Parameter-Dateien gebraucht
        self.config = config
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

        self.button_save_parameters = QPushButton("Parameter speichern")
        self.button_load_parameters = QPushButton("Parameter laden...")
        self.button_save_as_parameters = QPushButton("Parameter speichern unter...")

        self.button_save_parameters.clicked.connect(self.save_parameters)
        self.button_save_as_parameters.clicked.connect(self.save_as_parameters)
        self.button_load_parameters.clicked.connect(self.load_parameters)

        self.main_layout.addWidget(self.button_save_parameters)
        self.main_layout.addWidget(self.button_save_as_parameters)
        self.main_layout.addWidget(self.button_load_parameters)

        self.step = 0
        self.loss = 10000

        scroll.setWidget(container)

        root_section = CollapsibleSection(type(params_root).__name__, expanded=True)
        self.main_layout.addWidget(root_section)

        self._build_dataclass_into_layout(root_section.content_layout, params_root, path=type(params_root).__name__)
        self.main_layout.addStretch()

    def trigger_parameter_changed(self):
        self.parameter_tuned.emit()

    def _reset_by_labels(self, prefix: str, layer_indices):
        """Setzt alle Editoren ``<prefix>[i]`` fuer die gegebenen Schicht-Indizes auf 0."""

        wanted = {f"{prefix}[{int(i)}]" for i in layer_indices}
        for editor in self._tensor_editors:
            if any(label in editor.label_text for label in wanted):
                with torch.no_grad():
                    editor.tensor.fill_(0.0)
                editor.refresh_from_tensor()
        self.trigger_parameter_changed()

    def reset_occupancies(self, layer_indices):
        self._reset_by_labels("occupancies", layer_indices)

    def reset_free_positions(self, layer_indices):
        self._reset_by_labels("position_shifts", layer_indices)

    def save_parameters(self):
        # Speichern
        p = self.params_root
        save_path = f"parameters/parameters_step={self.step}_loss={self.loss:.4f}.json"
        with open(save_path, "w") as f:
            param_dict = serialize(p)
            json.dump(param_dict, f)

    def save_as_parameters(self):
        save_path = f"parameters/parameters_step={self.step}_loss={self.loss:.4f}.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Parameter speichern unter",
            save_path,
            "JSON-Dateien (*.json)"
        )

        if not file_path:
            return

        p = self.params_root

        with open(file_path, "w") as f:
            param_dict = serialize(p)
            json.dump(param_dict, f)


    def load_parameters_no_dialog(self):
        with open("parameters.json", "r") as f:
            data = json.load(f)

        update_from_serialized(self.params_root, self.migrate_loaded_dict(data))
        print("parameters loaded:", f.name)
        self.refresh_from_params()

    def migrate_loaded_dict(self, data):
        return migrate_parameters_dict(data, self.params_root, self.config)

    def load_dataclass_json(self, path: str, cls):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return update_from_serialized(self.params_root, self.migrate_loaded_dict(data))

    def load_parameters(self, selected_fields: list[str] | None = None):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Parameter laden",
            "parameters",
            "JSON-Dateien (*.json)"
        )

        if not file_path:
            return

        self.load_dataclass_json(file_path, type(self.params_root))
        self.refresh_from_params()
        self.trigger_parameter_changed()
        return

        if selected_fields is None:
            self.params_root = loaded
            return

        for field_name in selected_fields:
            if hasattr(loaded, field_name) and hasattr(self.params_root, field_name):
                setattr(self.params_root, field_name, getattr(loaded, field_name))

    # -------------------------
    # Public API
    # -------------------------
    def refresh_from_params(self, step=None, loss=None):
        if step is not None:
            self.step = step
        if loss is not None:
            self.loss = loss
        save_path = "Parameter speichern " + f"parameters_step={self.step}_loss={self.loss:.4f}.json"
        self.button_save_parameters.setText(save_path)
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
            editor.value_changed.connect(self.trigger_parameter_changed)
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
                editor.value_changed.connect(self.trigger_parameter_changed)
                self._tensor_editors.append(editor)
                section.content_layout.addWidget(editor)
            return

        # Aberrationen: kartesisch gespeichert, als Betrag + Winkel anzeigen
        if (isinstance(value, nn.ParameterDict) and name == "aberrations"
                and AberrationEditor.has_cartesian_aberrations(value)):
            editor = AberrationEditor(value)
            editor.value_changed.connect(self.trigger_parameter_changed)
            self._tensor_editors.append(editor)
            layout.addWidget(editor)
            return

        # ParameterDict
        if isinstance(value, nn.ParameterDict):
            section = CollapsibleSection(name, expanded=False)
            layout.addWidget(section)

            for key, param in value.items():
                if not isinstance(param, nn.Parameter):
                    continue
                editor = TensorEditor(str(key), param)
                editor.value_changed.connect(self.trigger_parameter_changed)
                self._tensor_editors.append(editor)
                section.content_layout.addWidget(editor)
            return

        # dict / Liste von Dataclasses (z.B. strain_fields je Gruppe)
        if isinstance(value, (dict, list, tuple)) and len(value) > 0:
            items = value.items() if isinstance(value, dict) else enumerate(value)
            items = list(items)
            if all(is_dataclass(v) or isinstance(v, nn.Parameter) for _, v in items):
                section = CollapsibleSection(name, expanded=False)
                layout.addWidget(section)
                for key, item in items:
                    self._build_value(section.content_layout, name=str(key), value=item,
                                      path=f"{path}[{key}]")
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