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

    def _build_from_dataclass(self):
        for f in fields(self.train_method):
            name = f.name
            value = getattr(self.train_method, name)

            if isinstance(value, bool):
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
                    self.widgets[name] = w
                else:
                    w = QtWidgets.QLineEdit(value)
                    w.editingFinished.connect(
                        lambda n=name, widget=w: self._update_attr(n, widget.text())
                    )
                    self.form.addRow(name, w)
                    self.widgets[name] = w

            else:
                w = QtWidgets.QLineEdit(str(value))
                w.editingFinished.connect(
                    lambda n=name, widget=w: self._update_attr(n, widget.text())
                )
                self.form.addRow(name, w)
                self.widgets[name] = w

    def _set_learning_rate(self, value: float):
        if hasattr(self.train_method, "learning_rate"):
            self.train_method.learning_rate = value
        if hasattr(self.train_method, "learing_rate"):
            self.train_method.learing_rate = value

    def _update_attr(self, name, value):
        setattr(self.train_method, name, value)
        self.changed.emit()

    def refresh_from_object(self):
        for f in fields(self.train_method):
            name = f.name
            if name not in self.widgets:
                continue

            widget = self.widgets[name]
            value = getattr(self.train_method, name)

            if isinstance(widget, QtWidgets.QCheckBox):
                widget.blockSignals(True)
                widget.setChecked(bool(value))
                widget.blockSignals(False)

            elif isinstance(widget, QtWidgets.QSpinBox):
                widget.blockSignals(True)
                widget.setValue(int(value))
                widget.blockSignals(False)

            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                widget.blockSignals(True)
                widget.setValue(float(value))
                widget.blockSignals(False)

            elif isinstance(widget, QtWidgets.QComboBox):
                widget.blockSignals(True)
                idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                widget.blockSignals(False)

            elif isinstance(widget, QtWidgets.QLineEdit):
                widget.blockSignals(True)
                widget.setText(str(value))
                widget.blockSignals(False)

    def graphen_training(self):
        tm = self.train_method
        self._set_learning_rate(1e-9)

        tm.train_graphen_intensities = False
        tm.train_S1_intensities = False
        tm.train_W_intensities = False
        tm.train_S2_intensities = False

        tm.train_graphen_occupancies = False
        tm.train_S1_occupancies = False
        tm.train_W_occupancies = False
        tm.train_S2_occupancies = False

        tm.graphen_shift = True
        tm.graphen_rotation = True
        tm.graphen_free_positions = False
        tm.graphen_scale = True

        tm.WS2_shift = False
        tm.WS2_rotation = False
        tm.WS2_free_positions = False
        tm.WS2_scale = False

        tm.train_ctf_aberrations = False
        tm.train_multislice_sigma = False

        if hasattr(tm, "only_train_small_area"):
            tm.only_train_small_area = 0

        self.refresh_from_object()
        self.changed.emit()

    def WS2_training(self):
        tm = self.train_method

        if hasattr(tm, "train_blur"):
            tm.train_blur = True
        if hasattr(tm, "train_amplitude_contrast"):
            tm.train_amplitude_contrast = True

        self._set_learning_rate(1e-5)

        tm.train_graphen_intensities = False
        tm.train_S1_intensities = False
        tm.train_W_intensities = False
        tm.train_S2_intensities = False

        tm.train_graphen_occupancies = False
        tm.train_S1_occupancies = False
        tm.train_W_occupancies = False
        tm.train_S2_occupancies = False

        tm.graphen_shift = False
        tm.graphen_rotation = False
        tm.graphen_free_positions = False
        tm.graphen_scale = False

        tm.WS2_shift = True
        tm.WS2_rotation = True
        tm.WS2_free_positions = False
        tm.WS2_scale = True

        tm.train_ctf_aberrations = True
        tm.train_multislice_sigma = True

        if hasattr(tm, "train_env_parameters"):
            tm.train_env_parameters = True

        if hasattr(tm, "only_train_small_area"):
            tm.only_train_small_area = 1

        self.refresh_from_object()
        self.changed.emit()

    def main_training(self):
        tm = self.train_method
        self._set_learning_rate(1e-4)

        if hasattr(tm, "train_blur"):
            tm.train_blur = False
        if hasattr(tm, "train_amplitude_contrast"):
            tm.train_amplitude_contrast = False

        tm.train_graphen_intensities = False
        tm.train_S1_intensities = False
        tm.train_W_intensities = False
        tm.train_S2_intensities = False

        tm.train_graphen_occupancies = False
        tm.train_S1_occupancies = False
        tm.train_W_occupancies = False
        tm.train_S2_occupancies = False


        tm.graphen_shift = False
        tm.graphen_rotation = False
        tm.graphen_free_positions = False
        tm.graphen_scale = False

        tm.WS2_shift = False
        tm.WS2_rotation = False
        tm.WS2_free_positions = True
        tm.WS2_scale = False

        tm.train_ctf_aberrations = False
        tm.train_multislice_sigma = False

        if hasattr(tm, "only_train_small_area"):
            tm.only_train_small_area = -1

        self.refresh_from_object()
        self.changed.emit()