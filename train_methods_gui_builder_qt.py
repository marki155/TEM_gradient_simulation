"""GUI-Editor fuer die TrainMethod.

Der Editor baut sich komplett aus dem Objekt auf: globale Felder aus den
Dataclass-Feldern, dazu je eine Sektion fuer jede Struktur-Gruppe und jede
Schicht aus ``train_method.groups`` / ``train_method.layers``.  Neue Schichten
in der Config tauchen damit automatisch in der GUI auf.
"""

from dataclasses import fields, is_dataclass

from PySide6 import QtCore, QtWidgets

from training_parameter import GROUP_KEY, LAYER_KEY, is_train_param

MAIN_ENTRY = "Main (alle Gruppen)"


class TrainMethodEditor(QtWidgets.QGroupBox):
    changed = QtCore.Signal()

    def __init__(self, train_method, config=None, parent=None):
        super().__init__("TrainMethod", parent)

        if not is_dataclass(train_method):
            raise TypeError("train_method muss eine Dataclass-Instanz sein")

        self.train_method = train_method
        self.config = config
        self.widgets = {}

        self.root = QtWidgets.QVBoxLayout(self)
        self._build_active_group_selector()
        self._build_group_sections()
        self._build_layer_sections()
        self._build_global_section()

    # ------------------------------------------------------------------
    # Aufbau
    # ------------------------------------------------------------------
    def _build_active_group_selector(self):
        box = QtWidgets.QGroupBox("Aktive Gruppe")
        layout = QtWidgets.QFormLayout(box)

        self.active_group_combo = QtWidgets.QComboBox()
        self.active_group_combo.addItem(MAIN_ENTRY, None)
        for name in self._group_names():
            self.active_group_combo.addItem(name, name)
        self._select_active_group_in_combo()
        self.active_group_combo.currentIndexChanged.connect(self._on_active_group_changed)
        layout.addRow("Gruppe", self.active_group_combo)

        self.isolate_checkbox = QtWidgets.QCheckBox(
            "nur diese Gruppe simulieren (andere Schichten auf Null)")
        self.isolate_checkbox.toggled.connect(self._on_isolate_changed)
        layout.addRow("", self.isolate_checkbox)
        self._refresh_isolate_checkbox()

        self.preset_buttons_box = QtWidgets.QWidget()
        preset_layout = QtWidgets.QHBoxLayout(self.preset_buttons_box)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        for name in self._group_names():
            btn = QtWidgets.QPushButton(f"nur {name}")
            btn.setToolTip(f"Nur die Geometrie der Gruppe {name} trainieren")
            btn.clicked.connect(lambda _=False, n=name: self.train_only_group(n))
            preset_layout.addWidget(btn)
        btn_free = QtWidgets.QPushButton("freie Positionen")
        btn_free.setToolTip("Alle Gruppen fix, dafuer freie Atompositionen trainieren")
        btn_free.clicked.connect(self.free_positions_training)
        preset_layout.addWidget(btn_free)
        layout.addRow("Presets", self.preset_buttons_box)

        self.root.addWidget(box)

    def _build_group_sections(self):
        if not self.train_method.groups:
            return
        box = QtWidgets.QGroupBox("Gruppen (Shift / Rotation / Scale / Strain)")
        outer = QtWidgets.QVBoxLayout(box)
        for gname, gparams in self.train_method.groups.items():
            sub = QtWidgets.QGroupBox(gname)
            form = QtWidgets.QFormLayout(sub)
            for f in fields(gparams):
                key = f"{GROUP_KEY}.{gname}.{f.name}"
                form.addRow(f.name, self._make_trainparam_row(key))
            outer.addWidget(sub)
        self.root.addWidget(box)

    def _build_layer_sections(self):
        if not self.train_method.layers:
            return
        box = QtWidgets.QGroupBox("Schichten (Positionen / Intensitaeten / Occupancies)")
        outer = QtWidgets.QVBoxLayout(box)
        for lname, lparams in self.train_method.layers.items():
            sub = QtWidgets.QGroupBox(self._layer_title(lname))
            form = QtWidgets.QFormLayout(sub)
            for f in fields(lparams):
                key = f"{LAYER_KEY}.{lname}.{f.name}"
                form.addRow(f.name, self._make_trainparam_row(key))
            outer.addWidget(sub)
        self.root.addWidget(box)

    def _build_global_section(self):
        box = QtWidgets.QGroupBox("Global")
        self.form = QtWidgets.QFormLayout(box)
        skip = set(getattr(self.train_method, "NON_GUI_FIELDS", ()))

        for f in fields(self.train_method):
            name = f.name
            if name in skip:
                continue
            value = getattr(self.train_method, name)

            if is_train_param(value):
                self.form.addRow(name, self._make_trainparam_row(name))

            elif isinstance(value, bool):
                w = QtWidgets.QCheckBox()
                w.setChecked(value)
                w.stateChanged.connect(
                    lambda _=None, n=name, widget=w: self._update_attr(n, widget.isChecked())
                )
                self.form.addRow(name, w)
                self.widgets[name] = {"kind": "checkbox", "widget": w}

            elif isinstance(value, int):
                w = QtWidgets.QSpinBox()
                w.setRange(-10_000_000, 10_000_000)
                w.setValue(value)
                w.valueChanged.connect(
                    lambda _=None, n=name, widget=w: self._update_attr(n, int(widget.value()))
                )
                self.form.addRow(name, w)
                self.widgets[name] = {"kind": "spinbox", "widget": w}

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
                self.widgets[name] = {"kind": "doublespinbox", "widget": w}

            elif isinstance(value, str):
                w = QtWidgets.QLineEdit(value)
                w.editingFinished.connect(
                    lambda n=name, widget=w: self._update_attr(n, widget.text())
                )
                self.form.addRow(name, w)
                self.widgets[name] = {"kind": "lineedit", "widget": w}

        self.root.addWidget(box)

    def _make_trainparam_row(self, key: str) -> QtWidgets.QWidget:
        train_param = self.train_method.get_train_param(key)

        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        chk = QtWidgets.QCheckBox()
        chk.setChecked(bool(train_param.enabled))

        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(10)
        spin.setRange(0.0, 1e12)
        spin.setSingleStep(1e-5)
        spin.setValue(float(train_param.lr))
        spin.setMinimumWidth(120)

        chk.stateChanged.connect(
            lambda _=None, k=key, widget=chk: self._update_trainparam_enabled(k, widget.isChecked())
        )
        spin.valueChanged.connect(
            lambda _=None, k=key, widget=spin: self._update_trainparam_lr(k, float(widget.value()))
        )

        row_layout.addWidget(chk, 0)
        row_layout.addWidget(spin, 1)

        self.widgets[key] = {"kind": "trainparam", "enabled": chk, "lr": spin}
        return row_widget

    # ------------------------------------------------------------------
    # Hilfen
    # ------------------------------------------------------------------
    def _group_names(self):
        if self.config is not None:
            return list(self.config.group_names)
        return list(self.train_method.groups.keys())

    def _layer_names(self):
        if self.config is not None:
            return list(self.config.layer_names)
        return list(self.train_method.layers.keys())

    def _layer_title(self, layer_name):
        if self.config is None:
            return layer_name
        for layer in self.config.layers:
            if layer.name == layer_name:
                return f"{layer.name}  ({layer.element}, z={layer.z_nm:g} nm, Gruppe {layer.group})"
        return layer_name

    def _active_group_config(self):
        if self.config is None or self.train_method.active_group is None:
            return None
        try:
            return self.config.group(self.train_method.active_group)
        except KeyError:
            return None

    def _select_active_group_in_combo(self):
        idx = self.active_group_combo.findData(self.train_method.active_group)
        self.active_group_combo.blockSignals(True)
        self.active_group_combo.setCurrentIndex(max(idx, 0))
        self.active_group_combo.blockSignals(False)

    def _refresh_isolate_checkbox(self):
        group = self._active_group_config()
        self.isolate_checkbox.blockSignals(True)
        self.isolate_checkbox.setEnabled(group is not None)
        self.isolate_checkbox.setChecked(bool(group.isolate_when_active) if group else False)
        self.isolate_checkbox.blockSignals(False)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_active_group_changed(self, _index):
        self.train_method.set_active_group(self.active_group_combo.currentData())
        self._refresh_isolate_checkbox()
        self.changed.emit()

    def _on_isolate_changed(self, checked):
        group = self._active_group_config()
        if group is None:
            return
        group.isolate_when_active = bool(checked)
        self.changed.emit()

    def _update_attr(self, name, value):
        setattr(self.train_method, name, value)
        self.changed.emit()

    def _update_trainparam_enabled(self, key, value):
        train_param = self.train_method.get_train_param(key)
        if train_param is not None:
            train_param.enabled = bool(value)
        self.changed.emit()

    def _update_trainparam_lr(self, key, value):
        train_param = self.train_method.get_train_param(key)
        if train_param is not None:
            train_param.lr = float(value)
        self.changed.emit()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------
    def refresh_from_object(self):
        self._select_active_group_in_combo()
        self._refresh_isolate_checkbox()

        for key, info in self.widgets.items():
            if info["kind"] == "trainparam":
                value = self.train_method.get_train_param(key)
                if value is None:
                    continue
                info["enabled"].blockSignals(True)
                info["enabled"].setChecked(bool(value.enabled))
                info["enabled"].blockSignals(False)

                info["lr"].blockSignals(True)
                info["lr"].setValue(float(value.lr))
                info["lr"].blockSignals(False)
                continue

            value = getattr(self.train_method, key, None)
            if value is None:
                continue
            w = info["widget"]
            w.blockSignals(True)
            if info["kind"] == "checkbox":
                w.setChecked(bool(value))
            elif info["kind"] == "spinbox":
                w.setValue(int(value))
            elif info["kind"] == "doublespinbox":
                w.setValue(float(value))
            elif info["kind"] == "lineedit":
                w.setText(str(value))
            w.blockSignals(False)

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    def _disable_all(self):
        for _key, train_param in self.train_method.iter_train_params():
            train_param.enabled = False

    def train_only_group(self, group_name: str):
        """Nur die starre Geometrie einer Gruppe trainieren.

        Entspricht den frueheren Presets "Graphen area" / "WS2 area": alles
        andere wird eingefroren, die Gruppe wird aktiv gesetzt.  CTF/Envelope
        bleiben nur an, wenn die Gruppe nicht isoliert simuliert wird (bei einer
        isolierten Gruppe fehlt der Rest der Struktur, dann waeren die
        CTF-Parameter nicht sinnvoll bestimmbar).
        """

        tm = self.train_method
        self._disable_all()

        group = tm.group(group_name)
        group.shift.enabled = True
        group.rotation.enabled = True
        group.scale.enabled = True

        tm.set_active_group(group_name)

        isolated = False
        if self.config is not None:
            try:
                isolated = bool(self.config.group(group_name).isolate_when_active)
            except KeyError:
                isolated = False

        tm.train_ctf_aberrations.enabled = not isolated
        tm.train_env_parameters.enabled = not isolated
        tm.train_normalisation_parameters.enabled = True

        self.refresh_from_object()
        self.changed.emit()

    def free_positions_training(self):
        """Alle Gruppen fix, dafuer die freien Atompositionen trainieren."""

        tm = self.train_method
        self._disable_all()

        for lname in self._layer_names():
            tm.layer(lname).free_positions.enabled = True

        tm.set_active_group(None)
        tm.train_normalisation_parameters.enabled = True

        self.refresh_from_object()
        self.changed.emit()
