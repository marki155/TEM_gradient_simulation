"""Testset-Builder: Atome per Klick setzen, simulieren und als Datenordner fuer das Seeding speichern.

Start::

    python3 testset_builder_gui.py [ordner]

Der gespeicherte Ordner enthaelt alles, was ``mains/main_random_atoms.py`` (und damit die
Seeding-GUI) braucht: ``target.tif``, ``mask_<Gruppe>.tif`` und ``init_random.json``.
Dazu kommen die wahren Positionen (``truth_positions_px.npy``, ``truth.xyz``) und
``testset_builder.json`` zum spaeteren Weiterbearbeiten.

Positionen sind (Zeile, Spalte) in Pixeln, wie im Rest des Projekts.
"""

import copy
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multislice  # noqa: E402

multislice.CHECK_PHASE = False

BUILDER_FILE = "testset_builder.json"
TRUTH_NPY = "truth_positions_px.npy"
TRUTH_XYZ = "truth.xyz"
BOND_A = 1.42
GRAPHENE_A = 2.46
UNDO_LIMIT = 200

ENVELOPE_DEFAULT = {"source_size": 0.1e-3, "Cc_um": 1.0e3, "deltaE": 0.35,
                    "instab_defocus_nm": 0.01, "use_detector": False}


def default_params():
    from ctf import DEFAULT_ABERRATIONS, DEFAULT_PROBE_ABERRATIONS
    return {
        "name": "Gr", "element": "C", "debye_waller_u": None,
        "height": 128, "width": 128, "pixel_A": 0.2,
        "mode": "stem", "voltage_kV": 80.0, "padding": 50,
        "convergence_mrad": 24.81, "detector_inner_mrad": 60.0, "detector_outer_mrad": 200.0,
        "source_size_A": 0.3, "probe_aberrations": dict(DEFAULT_PROBE_ABERRATIONS),
        "aberrations": dict(DEFAULT_ABERRATIONS), "amp_contrast": 0.1, "alpha_max_mrad": 50.0,
        "envelope": dict(ENVELOPE_DEFAULT),
        "noise_sigma": 0.0, "dose": 0.0, "noise_seed": 0,
        "mask_mode": "all", "margin_px": 10, "mask_radius_A": 3.0,
    }


def hexagon(center, angle_deg, pixel_A, bond_A=BOND_A):
    """6 Atome eines Rings um ``center`` (row, col)."""

    radius = bond_A / pixel_A
    angles = np.deg2rad(angle_deg) + np.arange(6) * np.pi / 3
    return np.asarray(center, dtype=float) + radius * np.stack([np.sin(angles), np.cos(angles)], axis=1)


def graphene_points(shape, pixel_A, angle_deg=0.0, margin_px=0.0, a_A=GRAPHENE_A):
    """Graphen-Gitter (zwei Atome je Zelle), um die Bildmitte um ``angle_deg`` gedreht."""

    a = a_A / pixel_A
    a1 = np.array([a, 0.0])
    a2 = np.array([a / 2, a * np.sqrt(3) / 2])
    basis = np.array([a / 2, a / (2 * np.sqrt(3))])
    n = int(np.ceil(np.hypot(*shape) / a)) + 2
    i, j = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
    cells = i.reshape(-1, 1) * a1 + j.reshape(-1, 1) * a2
    points = np.concatenate([cells, cells + basis])
    center = (np.array(shape) - 1) / 2
    t = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    points = (points - basis) @ rot.T + center
    keep = ((points[:, 0] >= margin_px) & (points[:, 0] <= shape[0] - 1 - margin_px)
            & (points[:, 1] >= margin_px) & (points[:, 1] <= shape[1] - 1 - margin_px))
    return points[keep]


class TestsetModel:
    """Atome, Parameter, Simulation und Speichern - ohne GUI."""

    def __init__(self, params=None):
        self.params = default_params()
        if params:
            self.params.update(copy.deepcopy(params))
        self.positions = np.zeros((0, 2))
        self.weights = np.zeros(0)
        self._sim = None
        self._sim_key = None

    # -- Parameter ---------------------------------------------------------------
    @property
    def shape(self):
        return int(self.params["height"]), int(self.params["width"])

    @property
    def pixel_nm(self):
        return float(self.params["pixel_A"]) / 10.0

    def _optics_key(self):
        p = self.params
        keys = ["height", "width", "pixel_A", "mode", "voltage_kV", "padding", "element",
                "debye_waller_u", "name", "convergence_mrad", "detector_inner_mrad",
                "detector_outer_mrad", "source_size_A", "probe_aberrations", "aberrations",
                "amp_contrast", "alpha_max_mrad", "envelope"]
        return json.dumps({k: p[k] for k in keys}, sort_keys=True)

    # -- Atome -------------------------------------------------------------------
    def set_atoms(self, positions, weights=None):
        self.positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        n = len(self.positions)
        self.weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float).reshape(n)

    def add(self, positions, weight=1.0):
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        self.positions = np.concatenate([self.positions, positions])
        self.weights = np.concatenate([self.weights, np.full(len(positions), float(weight))])

    def remove(self, indices):
        keep = np.ones(len(self.positions), dtype=bool)
        keep[np.asarray(indices, dtype=int)] = False
        self.positions, self.weights = self.positions[keep], self.weights[keep]

    def nearest(self, point, max_px=None):
        """Index des naechsten Atoms (oder None, wenn weiter als ``max_px`` weg)."""

        if len(self.positions) == 0:
            return None
        d = np.hypot(*(self.positions - np.asarray(point)).T)
        i = int(np.argmin(d))
        return i if max_px is None or d[i] <= max_px else None

    def jitter(self, fraction, sigma_A, rng=None):
        rng = rng or np.random.default_rng()
        chosen = rng.random(len(self.positions)) < fraction
        self.positions = self.positions.copy()
        self.positions[chosen] += rng.normal(0.0, sigma_A / self.params["pixel_A"], (int(chosen.sum()), 2))

    def make_vacancies(self, fraction, rng=None):
        rng = rng or np.random.default_rng()
        self.remove(np.flatnonzero(rng.random(len(self.positions)) < fraction))

    def fill_graphene(self, angle_deg, margin_px):
        points = graphene_points(self.shape, self.params["pixel_A"], angle_deg, margin_px)
        self.set_atoms(points)

    # -- Simulation --------------------------------------------------------------
    def _simulator(self):
        key = self._optics_key()
        if self._sim is not None and self._sim_key == key:
            return self._sim
        from atom_seeding import AtomStore
        from mains.main_random_atoms import build_seeding_model
        p = self.params
        dummy = np.random.default_rng(0).random(self.shape)
        model, _target, _tm = build_seeding_model(
            dummy, self.pixel_nm, p["mode"], voltage_kV=float(p["voltage_kV"]), border_px=0,
            convergence_mrad=float(p["convergence_mrad"]),
            detector_mrad=(float(p["detector_inner_mrad"]), float(p["detector_outer_mrad"])),
            source_size_A=float(p["source_size_A"]),
            aberrations=dict(p["aberrations"]) if p["mode"] == "tem" else None,
            probe_aberrations=dict(p["probe_aberrations"]) if p["mode"] == "stem" else None,
            env_params=dict(p["envelope"]), amp_contrast=float(p["amp_contrast"]),
            alpha_max_mrad=float(p["alpha_max_mrad"]), padding=int(p["padding"]),
            group_name=p["name"], element=p["element"], debye_waller_u=p["debye_waller_u"])
        self._sim = (model, AtomStore(model.potential_layer))
        self._sim_key = key
        return self._sim

    def render_clean(self):
        """Simulation ohne Rauschen, auf [0, 1] normiert."""

        import torch
        model, store = self._simulator()
        weights = self.weights * store.new_weight
        store.apply(np.arange(store.n), self.positions, weights if len(weights) else None)
        with torch.no_grad():
            image = model.model(None)[0].detach().cpu().numpy().astype(np.float64)
        span = image.max() - image.min()
        return ((image - image.min()) / span if span > 0 else np.zeros_like(image)).astype(np.float32)

    def add_noise(self, image):
        p = self.params
        rng = np.random.default_rng(int(p["noise_seed"]))
        image = image.astype(np.float64)
        if p["dose"] > 0:
            image = rng.poisson(np.clip(image, 0, None) * p["dose"]) / p["dose"]
        if p["noise_sigma"] > 0:
            image = image + rng.normal(0.0, p["noise_sigma"], image.shape)
        return image.astype(np.float32)

    def render(self, noise=True):
        image = self.render_clean()
        return self.add_noise(image) if noise else image

    # -- Maske -------------------------------------------------------------------
    def mask(self):
        p = self.params
        shape = self.shape
        if p["mask_mode"] == "margin":
            m = int(p["margin_px"])
            mask = np.zeros(shape, dtype=bool)
            mask[m:shape[0] - m, m:shape[1] - m] = True
            return mask
        if p["mask_mode"] == "atoms" and len(self.positions):
            from scipy.ndimage import distance_transform_edt
            hit = np.ones(shape, dtype=bool)
            r = np.clip(np.rint(self.positions).astype(int), 0, np.array(shape) - 1)
            hit[r[:, 0], r[:, 1]] = False
            return distance_transform_edt(hit) <= p["mask_radius_A"] / p["pixel_A"]
        return np.ones(shape, dtype=bool)

    # -- Speichern ---------------------------------------------------------------
    def init_dict(self, mask_file):
        p = self.params
        imaging = {"mode": "tem" if p["mode"] == "tem" else "stem_haadf",
                   "amp_contrast": p["amp_contrast"], "alpha_max_mrad": p["alpha_max_mrad"]}
        if p["mode"] == "tem":
            imaging["aberrations"] = dict(p["aberrations"])
            imaging["envelope"] = dict(p["envelope"])
        else:
            imaging["stem"] = {"aberrations": dict(p["probe_aberrations"]),
                               "convergence_mrad": p["convergence_mrad"],
                               "detector_mrad": [p["detector_inner_mrad"], p["detector_outer_mrad"]],
                               "source_size_A": p["source_size_A"]}
        layer = {"element": p["element"]}
        if p["debye_waller_u"] is not None:
            layer["debye_waller_u"] = p["debye_waller_u"]
        return {"pixel_size_nm": self.pixel_nm, "voltage_kV": p["voltage_kV"],
                "shape": list(self.shape), "padding": int(p["padding"]),
                "imaging": imaging, "layers": [layer],
                "groups": [{"name": p["name"], "mask_file": mask_file}]}

    def save(self, folder, image=None):
        """Datenordner schreiben; ``image`` (fertiges Zielbild) wird sonst neu simuliert."""

        import tifffile
        os.makedirs(folder, exist_ok=True)
        if image is None:
            image = self.render(noise=True)
        p = self.params
        mask_file = f"mask_{p['name']}.tif"
        tifffile.imwrite(os.path.join(folder, "target.tif"), np.asarray(image, dtype=np.float32))
        tifffile.imwrite(os.path.join(folder, mask_file), self.mask().astype(np.uint8) * 255)
        with open(os.path.join(folder, "init_random.json"), "w") as f:
            json.dump(self.init_dict(mask_file), f, indent=2)
        np.save(os.path.join(folder, TRUTH_NPY), self.positions)
        with open(os.path.join(folder, TRUTH_XYZ), "w") as f:
            f.write(f"{len(self.positions)}\ntruth positions, x=col*pixel_A, y=row*pixel_A\n")
            for (row, col) in self.positions:
                f.write(f"{p['element']} {col * p['pixel_A']:.5f} {row * p['pixel_A']:.5f} 0.0\n")
        with open(os.path.join(folder, BUILDER_FILE), "w") as f:
            json.dump({"params": p, "positions": self.positions.tolist(),
                       "weights": self.weights.tolist()}, f, indent=1)

    @classmethod
    def load(cls, folder):
        with open(os.path.join(folder, BUILDER_FILE)) as f:
            state = json.load(f)
        model = cls(state["params"])
        model.set_atoms(state["positions"], state["weights"])
        return model


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def _qt():
    from PySide6 import QtCore, QtWidgets
    return QtCore, QtWidgets


def build_window_class():
    QtCore, QtWidgets = _qt()
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure

    class AberrationTable(QtWidgets.QTableWidget):
        def __init__(self, values, changed):
            super().__init__(len(values), 2)
            self.setHorizontalHeaderLabels(["Aberration", "Wert"])
            self.verticalHeader().hide()
            self.horizontalHeader().setStretchLastSection(True)
            self.set_values(values)
            self.itemChanged.connect(lambda _item: changed())
            self.setMaximumHeight(220)

        def set_values(self, values):
            self.blockSignals(True)
            self.setRowCount(len(values))
            for row, (key, value) in enumerate(values.items()):
                name = QtWidgets.QTableWidgetItem(key)
                name.setFlags(QtCore.Qt.ItemIsEnabled)
                self.setItem(row, 0, name)
                self.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{float(value):g}"))
            self.blockSignals(False)

        def values(self):
            result = {}
            for row in range(self.rowCount()):
                try:
                    result[self.item(row, 0).text()] = float(self.item(row, 1).text().replace(",", "."))
                except ValueError:
                    result[self.item(row, 0).text()] = 0.0
            return result

    class TestsetBuilderWindow(QtWidgets.QMainWindow):
        def __init__(self, folder=None, model=None):
            super().__init__()
            self.setWindowTitle("Testset-Builder")
            self.model = model or TestsetModel()
            self._undo = []
            self._clean = None
            self._grab = None
            self._stale = True
            self._widgets = {}
            self._loading = True
            self.render_timer = QtCore.QTimer(self)
            self.render_timer.setSingleShot(True)
            self.render_timer.setInterval(350)
            self.render_timer.timeout.connect(self.render)

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            layout = QtWidgets.QHBoxLayout(central)

            left = QtWidgets.QVBoxLayout()
            layout.addLayout(left, 1)
            self.figure = Figure(figsize=(7, 7))
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            left.addWidget(self.toolbar)
            left.addWidget(self.canvas, 1)
            self.info = QtWidgets.QLabel()
            left.addWidget(self.info)
            self.ax = self.figure.add_axes([0.04, 0.04, 0.94, 0.94])
            self.image_artist = None
            self.scatter = None
            self.mask_artist = None
            self.canvas.mpl_connect("button_press_event", self._on_press)
            self.canvas.mpl_connect("motion_notify_event", self._on_motion)
            self.canvas.mpl_connect("button_release_event", self._on_release)

            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFixedWidth(400)
            panel = QtWidgets.QWidget()
            scroll.setWidget(panel)
            self.panel_layout = QtWidgets.QVBoxLayout(panel)
            layout.addWidget(scroll)
            self._build_panel()

            self.set_params_to_widgets()
            if folder and os.path.exists(os.path.join(folder, BUILDER_FILE)):
                self.folder_edit.setText(folder)
                self.load_folder()
            elif folder:
                self.folder_edit.setText(folder)
            self.rebuild_view()
            self.statusBar().showMessage("Linksklick: Atom setzen / ziehen, Rechtsklick: loeschen")

        # -- Aufbau --------------------------------------------------------------
        def _group(self, title):
            box = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(box)
            self.panel_layout.addWidget(box)
            return form

        def _spin(self, form, key, label, lo, hi, decimals=3, step=None, special=None, integer=False):
            if integer:
                w = QtWidgets.QSpinBox()
                w.setRange(int(lo), int(hi))
            else:
                w = QtWidgets.QDoubleSpinBox()
                w.setDecimals(decimals)
                w.setRange(lo, hi)
                w.setSingleStep(step or 10 ** -decimals)
            if special:
                w.setSpecialValueText(special)
            w.valueChanged.connect(lambda _v, k=key: self._param_changed(k))
            form.addRow(label, w)
            self._widgets[key] = w
            return w

        def _build_panel(self):
            L = self.panel_layout

            form = self._group("Ordner")
            row = QtWidgets.QHBoxLayout()
            self.folder_edit = QtWidgets.QLineEdit()
            browse = QtWidgets.QPushButton("...")
            browse.clicked.connect(self.browse_folder)
            row.addWidget(self.folder_edit)
            row.addWidget(browse)
            form.addRow(row)
            buttons = QtWidgets.QHBoxLayout()
            for text, slot in (("Speichern", self.save_clicked), ("Laden", self.load_clicked),
                               ("Seeding-GUI", self.start_seeding_gui)):
                b = QtWidgets.QPushButton(text)
                b.clicked.connect(slot)
                buttons.addWidget(b)
            form.addRow(buttons)

            form = self._group("Werkzeug")
            self.tool = QtWidgets.QComboBox()
            self.tool.addItems(["Atom setzen / ziehen", "6-Ring setzen", "Atom loeschen"])
            form.addRow("Linksklick", self.tool)
            self._spin(form, "new_weight", "Gewicht neuer Atome", 0.05, 5.0, 2, 0.05).setValue(1.0)
            self.ring_angle = QtWidgets.QDoubleSpinBox()
            self.ring_angle.setRange(-180, 180)
            self.ring_angle.setSuffix(" deg")
            form.addRow("Ring-/Gitterwinkel", self.ring_angle)
            row = QtWidgets.QHBoxLayout()
            for text, slot in (("Rueckgaengig", self.undo), ("Alle loeschen", self.clear_atoms)):
                b = QtWidgets.QPushButton(text)
                b.clicked.connect(slot)
                row.addWidget(b)
            form.addRow(row)

            form = self._group("Erzeugen / Stoeren")
            self.fill_margin = QtWidgets.QSpinBox()
            self.fill_margin.setRange(0, 500)
            self.fill_margin.setValue(8)
            form.addRow("Gitter-Rand [px]", self.fill_margin)
            b = QtWidgets.QPushButton("Graphen-Gitter fuellen (ersetzt alle)")
            b.clicked.connect(self.fill_graphene)
            form.addRow(b)
            self.jitter_fraction = QtWidgets.QDoubleSpinBox()
            self.jitter_fraction.setRange(0, 1)
            self.jitter_fraction.setSingleStep(0.1)
            self.jitter_fraction.setValue(0.3)
            self.jitter_sigma = QtWidgets.QDoubleSpinBox()
            self.jitter_sigma.setRange(0, 5)
            self.jitter_sigma.setSingleStep(0.05)
            self.jitter_sigma.setValue(0.2)
            form.addRow("Anteil", self.jitter_fraction)
            form.addRow("Sigma [A]", self.jitter_sigma)
            row = QtWidgets.QHBoxLayout()
            for text, slot in (("Verschieben", self.jitter_atoms), ("Fehlstellen", self.vacancies)):
                b = QtWidgets.QPushButton(text)
                b.clicked.connect(slot)
                row.addWidget(b)
            form.addRow(row)

            form = self._group("Bild")
            self._spin(form, "height", "Hoehe [px]", 16, 4096, integer=True)
            self._spin(form, "width", "Breite [px]", 16, 4096, integer=True)
            self._spin(form, "pixel_A", "Pixelgroesse [A/px]", 0.005, 5.0, 4, 0.01)
            self.fov_label = QtWidgets.QLabel()
            form.addRow("Bildfeld", self.fov_label)
            self.mode = QtWidgets.QComboBox()
            self.mode.addItems(["stem", "tem"])
            self.mode.currentIndexChanged.connect(lambda _i: self._param_changed("mode"))
            form.addRow("Modus", self.mode)
            self._spin(form, "voltage_kV", "Spannung [kV]", 20, 400, 1, 10)
            self._spin(form, "padding", "Padding [px]", 0, 1000, integer=True)
            self.element = QtWidgets.QLineEdit()
            self.element.editingFinished.connect(lambda: self._param_changed("element"))
            form.addRow("Element", self.element)
            self._spin(form, "debye_waller_u", "Debye-Waller u [A^2]", 0.0, 1.0, 4, 0.001, special="Standard")
            self.group_name = QtWidgets.QLineEdit()
            self.group_name.editingFinished.connect(lambda: self._param_changed("name"))
            form.addRow("Gruppe", self.group_name)

            form = self._group("STEM")
            self._spin(form, "convergence_mrad", "Konvergenz [mrad]", 1, 100, 2, 0.5)
            self._spin(form, "detector_inner_mrad", "Detektor innen [mrad]", 0, 500, 1, 5)
            self._spin(form, "detector_outer_mrad", "Detektor aussen [mrad]", 1, 1000, 1, 5)
            self._spin(form, "source_size_A", "Quellgroesse [A]", 0, 5, 3, 0.05)
            self.probe_table = AberrationTable(self.model.params["probe_aberrations"],
                                               lambda: self._param_changed("probe_aberrations"))
            form.addRow(self.probe_table)

            form = self._group("TEM")
            self._spin(form, "amp_contrast", "Amplitudenkontrast", 0, 1, 3, 0.05)
            self._spin(form, "alpha_max_mrad", "alpha max [mrad]", 1, 500, 1, 5)
            self.env_widgets = {}
            for key, label in (("source_size", "Quellgroesse [rad]"), ("Cc_um", "Cc [um]"),
                               ("deltaE", "deltaE [eV]"), ("instab_defocus_nm", "Defokus-Instab. [nm]")):
                w = QtWidgets.QDoubleSpinBox()
                w.setDecimals(6)
                w.setRange(0, 1e6)
                w.valueChanged.connect(lambda _v: self._param_changed("envelope"))
                form.addRow(label, w)
                self.env_widgets[key] = w
            self.tem_table = AberrationTable(self.model.params["aberrations"],
                                             lambda: self._param_changed("aberrations"))
            form.addRow(self.tem_table)

            form = self._group("Rauschen")
            self._spin(form, "noise_sigma", "Gauss-Sigma (Bild 0..1)", 0, 2, 3, 0.01)
            self._spin(form, "dose", "Dosis [e/px] (0 = aus)", 0, 1e7, 1, 10)
            self._spin(form, "noise_seed", "Seed", 0, 10 ** 9, integer=True)

            form = self._group("Maske")
            self.mask_mode = QtWidgets.QComboBox()
            self.mask_mode.addItems(["all", "margin", "atoms"])
            self.mask_mode.currentIndexChanged.connect(lambda _i: self._param_changed("mask_mode"))
            form.addRow("Bereich", self.mask_mode)
            self._spin(form, "margin_px", "Rand [px]", 0, 1000, integer=True)
            self._spin(form, "mask_radius_A", "Radius um Atome [A]", 0.1, 50, 2, 0.5)

            form = self._group("Ansicht")
            self.show_noise = QtWidgets.QCheckBox("Rauschen zeigen")
            self.show_noise.setChecked(True)
            self.show_atoms = QtWidgets.QCheckBox("Atome zeigen")
            self.show_atoms.setChecked(True)
            self.show_mask = QtWidgets.QCheckBox("Maske zeigen")
            self.auto_render = QtWidgets.QCheckBox("automatisch rendern")
            self.auto_render.setChecked(True)
            for w in (self.show_noise, self.show_atoms, self.show_mask):
                w.toggled.connect(lambda _c: self.rebuild_view())
            self.auto_render.toggled.connect(lambda _c: self._schedule_render())
            for w in (self.show_noise, self.show_atoms, self.show_mask, self.auto_render):
                form.addRow(w)
            b = QtWidgets.QPushButton("Jetzt rendern")
            b.clicked.connect(self.render)
            form.addRow(b)
            L.addStretch(1)

        # -- Parameter <-> Widgets ---------------------------------------------------
        def set_params_to_widgets(self):
            self._loading = True
            p = self.model.params
            for key, w in self._widgets.items():
                if key == "new_weight":
                    continue
                value = p[key]
                if key == "debye_waller_u":
                    value = 0.0 if value is None else value
                w.setValue(value)
            self.mode.setCurrentText(p["mode"])
            self.element.setText(p["element"])
            self.group_name.setText(p["name"])
            self.mask_mode.setCurrentText(p["mask_mode"])
            self.probe_table.set_values(p["probe_aberrations"])
            self.tem_table.set_values(p["aberrations"])
            for key, w in self.env_widgets.items():
                w.setValue(float(p["envelope"].get(key, ENVELOPE_DEFAULT[key])))
            self._loading = False
            self._update_fov()

        def _read_widgets(self):
            p = self.model.params
            for key, w in self._widgets.items():
                if key == "new_weight":
                    continue
                value = w.value()
                if key == "debye_waller_u":
                    value = None if value <= 0 else value
                p[key] = value
            p["mode"] = self.mode.currentText()
            p["mask_mode"] = self.mask_mode.currentText()
            p["name"] = self.group_name.text().strip() or "Gr"
            element = self.element.text().strip() or "C"
            from ase.data import atomic_numbers
            if element not in atomic_numbers:
                self.statusBar().showMessage(f"Unbekanntes Element '{element}' - bleibt {p['element']}")
                self.element.setText(p["element"])
            else:
                p["element"] = element
            p["probe_aberrations"] = self.probe_table.values()
            p["aberrations"] = self.tem_table.values()
            env = dict(p["envelope"])
            env.update({k: w.value() for k, w in self.env_widgets.items()})
            p["envelope"] = env

        def _param_changed(self, _key=None):
            if self._loading or _key == "new_weight":
                return
            self._read_widgets()
            self._update_fov()
            self.rebuild_view()
            self._mark_stale()

        def _update_fov(self):
            p = self.model.params
            self.fov_label.setText(f"{p['width'] * p['pixel_A'] / 10:.3f} x "
                                   f"{p['height'] * p['pixel_A'] / 10:.3f} nm")

        # -- Anzeige ---------------------------------------------------------------
        def _mark_stale(self):
            self._stale = True
            self._schedule_render()
            self._update_info()

        def _schedule_render(self):
            if self.auto_render.isChecked() and self._stale:
                self.render_timer.start()

        def _update_info(self):
            state = "Bild veraltet" if self._stale else "Bild aktuell"
            self.info.setText(f"{len(self.model.positions)} Atome, {self.model.shape[1]}x"
                              f"{self.model.shape[0]} px, {self.model.params['mode'].upper()} - {state}")

        def render(self):
            self.render_timer.stop()
            self.setCursor(QtCore.Qt.WaitCursor)
            QtWidgets.QApplication.processEvents()
            try:
                self._clean = self.model.render_clean()
                self._stale = False
            except Exception as exc:
                self.statusBar().showMessage(f"Rendern fehlgeschlagen: {exc}")
                raise
            finally:
                self.unsetCursor()
            self.rebuild_view()

        def displayed_image(self):
            if self._clean is None or self._clean.shape != self.model.shape:
                return np.zeros(self.model.shape, dtype=np.float32)
            return self.model.add_noise(self._clean) if self.show_noise.isChecked() else self._clean

        def rebuild_view(self):
            """Bild, Atome und Maske neu zeichnen (Zoom bleibt, solange die Bildgroesse gleich ist)."""

            ax = self.ax
            shape = self.model.shape
            same = (self.image_artist is not None and self.image_artist.get_array().shape == shape)
            limits = (ax.get_xlim(), ax.get_ylim()) if same else None
            ax.clear()
            image = self.displayed_image()
            self.image_artist = ax.imshow(image, cmap="gray", interpolation="nearest")
            if self.show_mask.isChecked():
                mask = self.model.mask()
                if not mask.all():
                    ax.contour(mask.astype(float), levels=[0.5], colors="yellow", linewidths=1)
            pos = self.model.positions
            if self.show_atoms.isChecked() and len(pos):
                self.scatter = ax.scatter(pos[:, 1], pos[:, 0], s=22, facecolors="none",
                                          edgecolors="red", linewidths=1.0)
            else:
                self.scatter = None
            ax.set_xlim(-0.5, shape[1] - 0.5)
            ax.set_ylim(shape[0] - 0.5, -0.5)
            if limits:
                ax.set_xlim(limits[0])
                ax.set_ylim(limits[1])
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw_idle()
            self._update_info()

        # -- Atome bearbeiten --------------------------------------------------------
        def _snapshot(self):
            self._undo.append((self.model.positions.copy(), self.model.weights.copy()))
            del self._undo[:-UNDO_LIMIT]

        def undo(self):
            if not self._undo:
                return
            self.model.set_atoms(*self._undo.pop())
            self._atoms_changed()

        def _atoms_changed(self):
            self.rebuild_view()
            self._mark_stale()

        def clear_atoms(self):
            self._snapshot()
            self.model.set_atoms(np.zeros((0, 2)))
            self._atoms_changed()

        def fill_graphene(self):
            self._snapshot()
            self.model.fill_graphene(self.ring_angle.value(), self.fill_margin.value())
            self._atoms_changed()

        def jitter_atoms(self):
            self._snapshot()
            self.model.jitter(self.jitter_fraction.value(), self.jitter_sigma.value())
            self._atoms_changed()

        def vacancies(self):
            self._snapshot()
            self.model.make_vacancies(self.jitter_fraction.value())
            self._atoms_changed()

        def _pick_radius(self):
            return 0.4 / self.model.params["pixel_A"] + 1.0

        def click(self, row, col, button=1):
            """Klick auf Bildkoordinaten (auch fuer Tests); liefert True, wenn sich etwas aenderte."""

            m = self.model
            if button == 3 or self.tool.currentIndex() == 2:
                index = m.nearest((row, col), self._pick_radius() * 2)
                if index is None:
                    return False
                self._snapshot()
                m.remove([index])
            elif self.tool.currentIndex() == 1:
                self._snapshot()
                m.add(hexagon((row, col), self.ring_angle.value(), m.params["pixel_A"]),
                      self._widgets["new_weight"].value())
            else:
                index = m.nearest((row, col), self._pick_radius())
                if index is not None:
                    self._snapshot()
                    self._grab = index
                    return False
                self._snapshot()
                m.add([[row, col]], self._widgets["new_weight"].value())
            self._atoms_changed()
            return True

        def _event_point(self, event):
            if event.inaxes is not self.ax or event.xdata is None or self.toolbar.mode:
                return None
            return float(event.ydata), float(event.xdata)

        def _on_press(self, event):
            point = self._event_point(event)
            if point is not None:
                self.click(*point, button=event.button)

        def _on_motion(self, event):
            point = self._event_point(event)
            if point is None:
                return
            if self._grab is not None:
                self.model.positions[self._grab] = point
                self.rebuild_view_atoms_only()

        def rebuild_view_atoms_only(self):
            if self.scatter is not None:
                self.scatter.set_offsets(self.model.positions[:, ::-1])
                self.canvas.draw_idle()
            else:
                self.rebuild_view()

        def _on_release(self, _event):
            if self._grab is not None:
                self._grab = None
                self._atoms_changed()

        # -- Datei -------------------------------------------------------------------
        def browse_folder(self):
            start = self.folder_edit.text() or os.getcwd()
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Datenordner", start)
            if folder:
                self.folder_edit.setText(folder)

        def folder(self):
            folder = self.folder_edit.text().strip()
            if not folder:
                raise ValueError("Kein Ordner angegeben")
            return os.path.abspath(folder)

        def save_folder(self):
            self._read_widgets()
            folder = self.folder()
            image = self.model.add_noise(self.model.render_clean())
            self.model.save(folder, image=image)
            self.statusBar().showMessage(f"{len(self.model.positions)} Atome gespeichert in {folder}")
            return folder

        def load_folder(self):
            self.model = TestsetModel.load(self.folder())
            self._undo.clear()
            self._clean = None
            self.set_params_to_widgets()
            self.rebuild_view()
            self._mark_stale()

        def _guarded(self, function):
            try:
                return function()
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Fehler", str(exc))

        def save_clicked(self):
            self._guarded(self.save_folder)

        def load_clicked(self):
            self._guarded(self.load_folder)

        def seeding_command(self, folder):
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mains", "main_random_atoms.py")
            return [sys.executable, script, folder, "--gui", "--fresh"]

        def start_seeding_gui(self):
            def run():
                folder = self.save_folder()
                subprocess.Popen(self.seeding_command(folder), cwd=os.path.dirname(os.path.abspath(__file__)))
                self.statusBar().showMessage(f"Seeding-GUI gestartet fuer {folder}")
            self._guarded(run)

    return TestsetBuilderWindow


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = build_window_class()(argv[0] if argv else None)
    window.resize(1400, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
