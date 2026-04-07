from dataclasses import fields, is_dataclass
from PySide6 import QtCore, QtWidgets


class TrainMethodEditor(QtWidgets.QGroupBox):
    changed = QtCore.Signal()

    def __init__(self, train_method, parent=None):
        super().__init__("TrainMethod", parent)

        if not is_dataclass(train_method):
            raise TypeError("train_method muss eine Dataclass-Instanz sein")

        self.train_method = train_method
        self.widgets = {}

        self.form = QtWidgets.QFormLayout(self)
        self._build_from_dataclass()

    def _is_train_param(self, value):
        return is_dataclass(value) and hasattr(value, "enabled") and hasattr(value, "lr")

    def _build_from_dataclass(self):
        for f in fields(self.train_method):
            name = f.name
            value = getattr(self.train_method, name)

            if self._is_train_param(value):
                row_widget = QtWidgets.QWidget()
                row_layout = QtWidgets.QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)

                chk = QtWidgets.QCheckBox()
                chk.setChecked(bool(value.enabled))

                spin = QtWidgets.QDoubleSpinBox()
                spin.setDecimals(10)
                spin.setRange(0.0, 1e12)
                spin.setSingleStep(1e-5)
                spin.setValue(float(value.lr))
                spin.setMinimumWidth(120)

                chk.stateChanged.connect(
                    lambda _=None, n=name, widget=chk: self._update_trainparam_enabled(
                        n, widget.isChecked()
                    )
                )
                spin.valueChanged.connect(
                    lambda _=None, n=name, widget=spin: self._update_trainparam_lr(
                        n, float(widget.value())
                    )
                )

                row_layout.addWidget(chk, 0)
                row_layout.addWidget(spin, 1)

                self.form.addRow(name, row_widget)
                self.widgets[name] = {
                    "kind": "trainparam",
                    "enabled": chk,
                    "lr": spin,
                }

            elif isinstance(value, bool):
                w = QtWidgets.QCheckBox()
                w.setChecked(value)
                w.stateChanged.connect(
                    lambda _=None, n=name, widget=w: self._update_attr(n, widget.isChecked())
                )
                self.form.addRow(name, w)
                self.widgets[name] = {
                    "kind": "checkbox",
                    "widget": w,
                }

            elif isinstance(value, int):
                w = QtWidgets.QSpinBox()
                w.setRange(-10_000_000, 10_000_000)
                w.setValue(value)
                w.valueChanged.connect(
                    lambda _=None, n=name, widget=w: self._update_attr(n, int(widget.value()))
                )
                self.form.addRow(name, w)
                self.widgets[name] = {
                    "kind": "spinbox",
                    "widget": w,
                }

            elif isinstance(value, float):
                w = QtWidgets.QDoubleSpinBox()
                w.setDecimals(10)
                w.setRange(-1e12, 1e12)
                w.setSingleStep(1e-5)
                w.setValue(value)
                w.valueChanged.connect(
                    lambda _=None, n=name, widget=w: self._update_attr(n, float(widget.value()))
                )
                self.form.addRow(name, w)
                self.widgets[name] = {
                    "kind": "doublespinbox",
                    "widget": w,
                }

            elif isinstance(value, str):
                if name == "loss_fkt":
                    w = QtWidgets.QComboBox()
                    w.addItems(["MSE", "NCC", "SSIM"])
                    idx = w.findText(value)
                    if idx >= 0:
                        w.setCurrentIndex(idx)
                    w.currentTextChanged.connect(
                        lambda text, n=name: self._update_attr(n, text)
                    )
                    self.form.addRow(name, w)
                    self.widgets[name] = {
                        "kind": "combobox",
                        "widget": w,
                    }
                else:
                    w = QtWidgets.QLineEdit(value)
                    w.editingFinished.connect(
                        lambda n=name, widget=w: self._update_attr(n, widget.text())
                    )
                    self.form.addRow(name, w)
                    self.widgets[name] = {
                        "kind": "lineedit",
                        "widget": w,
                    }

            else:
                w = QtWidgets.QLineEdit(str(value))
                w.editingFinished.connect(
                    lambda n=name, widget=w: self._update_attr(n, widget.text())
                )
                self.form.addRow(name, w)
                self.widgets[name] = {
                    "kind": "lineedit",
                    "widget": w,
                }

    def _update_attr(self, name, value):
        setattr(self.train_method, name, value)
        self.changed.emit()

    def _update_trainparam_enabled(self, name, value):
        getattr(self.train_method, name).enabled = bool(value)
        self.changed.emit()

    def _update_trainparam_lr(self, name, value):
        getattr(self.train_method, name).lr = float(value)
        self.changed.emit()

    def refresh_from_object(self):
        for f in fields(self.train_method):
            name = f.name
            if name not in self.widgets:
                continue

            info = self.widgets[name]
            value = getattr(self.train_method, name)

            if info["kind"] == "trainparam":
                info["enabled"].blockSignals(True)
                info["enabled"].setChecked(bool(value.enabled))
                info["enabled"].blockSignals(False)

                info["lr"].blockSignals(True)
                info["lr"].setValue(float(value.lr))
                info["lr"].blockSignals(False)

            elif info["kind"] == "checkbox":
                w = info["widget"]
                w.blockSignals(True)
                w.setChecked(bool(value))
                w.blockSignals(False)

            elif info["kind"] == "spinbox":
                w = info["widget"]
                w.blockSignals(True)
                w.setValue(int(value))
                w.blockSignals(False)

            elif info["kind"] == "doublespinbox":
                w = info["widget"]
                w.blockSignals(True)
                w.setValue(float(value))
                w.blockSignals(False)

            elif info["kind"] == "combobox":
                w = info["widget"]
                w.blockSignals(True)
                idx = w.findText(str(value))
                if idx >= 0:
                    w.setCurrentIndex(idx)
                w.blockSignals(False)

            elif info["kind"] == "lineedit":
                w = info["widget"]
                w.blockSignals(True)
                w.setText(str(value))
                w.blockSignals(False)

    def graphen_training(self):
        tm = self.train_method

        tm.train_graphen_intensities.enabled = False
        tm.train_S1_intensities.enabled = False
        tm.train_W_intensities.enabled = False
        tm.train_S2_intensities.enabled = False

        tm.train_graphen_occupancies.enabled = False
        tm.train_S1_occupancies.enabled = False
        tm.train_W_occupancies.enabled = False
        tm.train_S2_occupancies.enabled = False

        tm.graphen_shift.enabled = True
        tm.graphen_rotation.enabled = True
        tm.graphen_free_positions.enabled = False
        tm.graphen_scale.enabled = True

        tm.WS2_shift.enabled = False
        tm.WS2_rotation.enabled = False
        tm.WS2_free_positions.enabled = False
        tm.WS2_scale.enabled = False

        tm.train_ctf_aberrations.enabled = False

        if hasattr(tm, "only_train_small_area"):
            tm.only_train_small_area = 0

        self.refresh_from_object()
        self.changed.emit()

    def WS2_training(self):
        tm = self.train_method

        if hasattr(tm, "train_blur"):
            tm.train_blur.enabled = True
        if hasattr(tm, "train_amplitude_contrast"):
            tm.train_amplitude_contrast.enabled = True

        tm.train_graphen_intensities.enabled = False
        tm.train_S1_intensities.enabled = False
        tm.train_W_intensities.enabled = False
        tm.train_S2_intensities.enabled = False

        tm.train_graphen_occupancies.enabled = False
        tm.train_S1_occupancies.enabled = False
        tm.train_W_occupancies.enabled = False
        tm.train_S2_occupancies.enabled = False

        tm.graphen_shift.enabled = False
        tm.graphen_rotation.enabled = False
        tm.graphen_free_positions.enabled = False
        tm.graphen_scale.enabled = False

        tm.WS2_shift.enabled = True
        tm.WS2_rotation.enabled = True
        tm.WS2_free_positions.enabled = False
        tm.WS2_scale.enabled = True

        tm.train_ctf_aberrations.enabled = True

        if hasattr(tm, "train_env_parameters"):
            tm.train_env_parameters.enabled = True

        if hasattr(tm, "only_train_small_area"):
            tm.only_train_small_area = 1

        self.refresh_from_object()
        self.changed.emit()

    def main_training(self):
        tm = self.train_method

        if hasattr(tm, "train_blur"):
            tm.train_blur.enabled = False
        if hasattr(tm, "train_amplitude_contrast"):
            tm.train_amplitude_contrast.enabled = False

        tm.train_graphen_intensities.enabled = False
        tm.train_S1_intensities.enabled = False
        tm.train_W_intensities.enabled = False
        tm.train_S2_intensities.enabled = False

        tm.train_graphen_occupancies.enabled = False
        tm.train_S1_occupancies.enabled = False
        tm.train_W_occupancies.enabled = False
        tm.train_S2_occupancies.enabled = False

        tm.graphen_shift.enabled = False
        tm.graphen_rotation.enabled = False
        tm.graphen_free_positions.enabled = False
        tm.graphen_scale.enabled = False

        tm.WS2_shift.enabled = False
        tm.WS2_rotation.enabled = False
        tm.WS2_free_positions.enabled = True
        tm.WS2_scale.enabled = False

        tm.train_ctf_aberrations.enabled = False

        if hasattr(tm, "only_train_small_area"):
            tm.only_train_small_area = -1

        self.refresh_from_object()
        self.changed.emit()


class TrainMethodEditorOLD(QtWidgets.QGroupBox):
    changed = QtCore.Signal()

    def __init__(self, train_method, parent=None):
        super().__init__("TrainMethod", parent)

        if not is_dataclass(train_method):
            raise TypeError("train_method muss eine Dataclass-Instanz sein")

        self.train_method = train_method
        self.widgets = {}
        self.form = QtWidgets.QFormLayout(self)

        self._build_from_dataclass()

    def _build_from_dataclass(self):
        for f in fields(self.train_method):
            name = f.name
            value = getattr(self.train_method, name)

            if is_dataclass(value) and hasattr(value, "enabled") and hasattr(value, "lr"):
                row_widget = QtWidgets.QWidget()
                row_layout = QtWidgets.QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)

                chk = QtWidgets.QCheckBox()
                chk.setChecked(value.enabled)

                spin = QtWidgets.QDoubleSpinBox()
                spin.setDecimals(10)
                spin.setRange(0.0, 1e12)
                spin.setSingleStep(1e-5)
                spin.setValue(value.lr)

                chk.stateChanged.connect(
                    lambda _=None, n=name, widget=chk: self._update_trainparam_enabled(n, widget.isChecked())
                )
                spin.valueChanged.connect(
                    lambda _=None, n=name, widget=spin: self._update_trainparam_lr(n, float(widget.value()))
                )

                row_layout.addWidget(chk)
                row_layout.addWidget(spin)

                self.form.addRow(name, row_widget)
                self.widgets[name] = {"enabled": chk, "lr": spin}

            elif isinstance(value, bool):
                w = QtWidgets.QCheckBox()
                w.setChecked(value)
                w.stateChanged.connect(
                    lambda _=None, n=name, widget=w: self._update_attr(n, widget.isChecked())
                )
                self.form.addRow(name, w)
                self.widgets[name] = w

            elif isinstance(value, int):
                w = QtWidgets.QSpinBox()
                w.setRange(-10_000_000, 10_000_000)
                w.setValue(value)
                w.valueChanged.connect(
                    lambda _=None, n=name, widget=w: self._update_attr(n, int(widget.value()))
                )
                self.form.addRow(name, w)
                self.widgets[name] = w

            elif isinstance(value, float):
                w = QtWidgets.QDoubleSpinBox()
                w.setDecimals(10)
                w.setRange(-1e12, 1e12)
                w.setSingleStep(1e-5)
                w.setValue(value)
                w.valueChanged.connect(
                    lambda _=None, n=name, widget=w: self._update_attr(n, float(widget.value()))
                )
                self.form.addRow(name, w)
                self.widgets[name] = w

            elif isinstance(value, str):
                w = QtWidgets.QLineEdit(value)
                w.editingFinished.connect(
                    lambda n=name, widget=w: self._update_attr(n, widget.text())
                )
                self.form.addRow(name, w)
                self.widgets[name] = w

    def _update_attr(self, name, value):
        setattr(self.train_method, name, value)
        self.changed.emit()

    def _update_trainparam_enabled(self, name, value):
        getattr(self.train_method, name).enabled = value
        self.changed.emit()

    def _update_trainparam_lr(self, name, value):
        getattr(self.train_method, name).lr = value
        self.changed.emit()

    def refresh_from_object(self):
        for f in fields(self.train_method):
            name = f.name
            if name not in self.widgets:
                continue

            value = getattr(self.train_method, name)
            widget = self.widgets[name]

            if isinstance(widget, dict):
                widget["enabled"].blockSignals(True)
                widget["enabled"].setChecked(value.enabled)
                widget["enabled"].blockSignals(False)

                widget["lr"].blockSignals(True)
                widget["lr"].setValue(value.lr)
                widget["lr"].blockSignals(False)