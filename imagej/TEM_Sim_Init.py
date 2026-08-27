#@ String (label="Parameterdatei (leer = Dialog)", value="", persist=false, required=false) params_path

"""TEM/STEM-Simulation: Initialisierung aus dem aktiven Bild.

Sammelt alles, was eine Simulation braucht - Pixelgroesse, Gitter, Schichten,
Gruppen, Masken, Abbildung - und schreibt es als init.json neben das Bild.
Danach baut mains/imagej_main.py daraus Config, TrainMethod und Modell.

Das Plugin arbeitet auf dem aktiven Bild. ROIs aus dem ROI-Manager werden zu
Gruppenmasken (Name des ROI = Name der Gruppe).

Mit einer Parameterdatei laeuft dasselbe ohne Dialog - so ist es testbar und
skriptbar:  ImageJ-linux64 --headless --run TEM_Sim_Init.py 'params_path="..."'
"""

import json
import math
import os
import sys

from ij import IJ, WindowManager
from ij.gui import Overlay, PointRoi
# NonBlockingGenericDialog erst im Dialog importieren: im Headless-Betrieb
# patcht ImageJ die GenericDialog so um, dass die Unterklasse nicht mehr laedt
from ij.io import FileSaver
from ij.plugin.frame import RoiManager
from ij.process import FHT, ByteProcessor
from java.awt import Color

# ---------------------------------------------------------------------------
# Modul mit den Gitterbasen daneben liegen lassen
# ---------------------------------------------------------------------------
try:
    from tem_sim_lattice import basis_for, layer_points, MX2_TABLE
except ImportError:
    _here = os.path.dirname(os.path.abspath(__file__ if "__file__" in dir() else "."))
    if _here not in sys.path:
        sys.path.append(_here)
    from tem_sim_lattice import basis_for, layer_points, MX2_TABLE

INIT_VERSION = 2

LATTICE_KINDS = ["Graphen", "hBN", "2H-MX2 (WS2, MoS2 ...)", "eigene Zelle", "aus Datei (.cif/.xyz)"]
IMAGING_MODES = ["TEM (Multislice + CTF)", "STEM HAADF (Sonde x Objekt)"]

DEFAULT_ABERRATIONS = {
    "C10": -13.07, "C12": 0.35, "phi12": 0.0, "C23": 0.17, "phi23": 0.0,
    "C21": 3.71, "phi21": 0.0, "C30": 4.16, "C34": -0.2, "phi34": 0.0,
    "C45": 0.25, "phi45": 0.0,
}

TYPICAL_DEBYE_WALLER = {"B": 0.070, "C": 0.060, "N": 0.060, "O": 0.070, "S": 0.070, "W": 0.055}


# ---------------------------------------------------------------------------
# FFT: Orientierung und Gitterkonstante schaetzen
# ---------------------------------------------------------------------------

def power_spectrum(imp):
    """Leistungsspektrum eines zentrierten, quadratischen Zweierpotenz-Ausschnitts."""

    ip = imp.getProcessor()
    size = 1
    while size * 2 <= min(ip.getWidth(), ip.getHeight()):
        size *= 2

    x0 = (ip.getWidth() - size) // 2
    y0 = (ip.getHeight() - size) // 2
    ip = ip.duplicate()
    ip.setRoi(x0, y0, size, size)
    ip = ip.crop().convertToFloat()

    fht = FHT(ip)
    fht.transform()
    return fht.getPowerSpectrum(), size


def find_lattice_peak(spectrum, size, min_radius_px=6, max_radius_px=None,
                      max_radius_frac=0.45):
    """Staerkster Reflex ausserhalb des Zentrums, subpixelgenau ueber den Schwerpunkt."""

    centre = size // 2
    best, best_value = None, -1.0
    limit = max_radius_px if max_radius_px else max_radius_frac * size

    for y in range(size):
        dy = y - centre
        for x in range(size):
            dx = x - centre
            r = math.hypot(dx, dy)
            if r < min_radius_px or r > limit:
                continue
            value = spectrum.getf(x, y)
            if value > best_value:
                best_value, best = value, (x, y)

    if best is None:
        return None

    # Schwerpunkt in einem kleinen Fenster fuer Subpixelgenauigkeit
    window = 3
    sx = sy = sw = 0.0
    for y in range(max(0, best[1] - window), min(size, best[1] + window + 1)):
        for x in range(max(0, best[0] - window), min(size, best[0] + window + 1)):
            w = spectrum.getf(x, y)
            sx += w * x
            sy += w * y
            sw += w
    if sw > 0:
        best = (sx / sw, sy / sw)

    dx = best[0] - centre
    dy = best[1] - centre
    return {"dx": dx, "dy": dy, "radius_px": math.hypot(dx, dy),
            "angle_deg": math.degrees(math.atan2(dy, dx)), "value": best_value}


def estimate_from_fft(imp, lattice_a_A, gamma_deg=120.0, min_radius_px=6.0, max_radius_px=None):
    """Pixelgroesse und Orientierung aus dem staerksten Gitterreflex.

    Bei hexagonalen Gittern liegt der erste Reflex bei d = a * sin(gamma), also
    2.13 A fuer Graphen - und seine Richtung ist gegen die Gitterachse um
    90 - gamma/2 verdreht (30 Grad bei gamma = 120).
    """

    spectrum, size = power_spectrum(imp)
    peak = find_lattice_peak(spectrum, size, min_radius_px, max_radius_px)
    if peak is None:
        return None

    d_theory_A = lattice_a_A * math.sin(math.radians(gamma_deg))
    d_px = size / peak["radius_px"]
    pixel_size_nm = d_theory_A / d_px / 10.0

    # Der Reflex steht senkrecht auf der zweiten Gitterachse, also um
    # 90 - gamma/2 gegen die erste verdreht (30 Grad bei gamma = 120). Dazu
    # kommt, dass die FFT-Winkel in Bildkoordinaten (x = Spalte, y = Zeile)
    # gemessen werden, die Gitterdrehung aber in (Zeile, Spalte): das dreht den
    # Winkel um (90 - psi). Zusammen: theta = gamma/2 - Peakwinkel.
    rotation = gamma_deg / 2.0 - peak["angle_deg"]
    period = 360.0 / round(360.0 / (180.0 - gamma_deg)) if gamma_deg != 180.0 else 180.0
    rotation = ((rotation + period / 2.0) % period) - period / 2.0

    return {"pixel_size_nm": pixel_size_nm, "rotation_deg": rotation,
            "d_px": d_px, "d_theory_A": d_theory_A, "peak": peak, "fft_size": size}


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def resolve_lattice(entry):
    """(Gitterkonstante, Basis) fuer einen Gitter-Eintrag."""

    if entry["kind"] == "aus Datei (.cif/.xyz)":
        return entry.get("a") or 0.0, []
    a, basis = basis_for(entry["kind"], entry.get("formula", "WS2"),
                         {"a": entry.get("a", 3.0), "basis": entry.get("custom_basis", [])})
    if entry.get("a"):
        a = entry["a"]
    return a, basis


def lattice_layers_for(entry, params, shape, margin_px=0.0, limit=None):
    """Schichten eines einzelnen Gitters in Bildkoordinaten."""

    a, basis = resolve_lattice(entry)
    if not basis:
        return []
    return layer_points(a, a, entry.get("gamma_deg", 120.0), basis,
                        params["pixel_size_nm"], shape, entry["rotation_deg"],
                        entry["offset_px"], margin_px=margin_px, limit=limit)


def update_overlay(imp, params, limit=4000):
    """Gitterpunkte aller Gitter als Overlay - eine Farbe je Gitter."""

    shape = (imp.getHeight(), imp.getWidth())
    overlay = Overlay()
    colors = [Color.yellow, Color.cyan, Color.magenta, Color.green, Color.orange]

    total = 0
    for index, entry in enumerate(params["lattices"]):
        layers = lattice_layers_for(entry, params, shape, limit=limit)
        for layer in layers:
            xs = [p[1] for p in layer["points"]]     # Spalte -> x
            ys = [p[0] for p in layer["points"]]     # Zeile  -> y
            if not xs:
                continue
            roi = PointRoi(xs, ys, len(xs))
            roi.setStrokeColor(colors[index % len(colors)])
            roi.setPointType(0)
            roi.setSize(0)
            overlay.add(roi)
            total += len(xs)

    imp.setOverlay(overlay)
    return total


# ---------------------------------------------------------------------------
# Masken aus dem ROI-Manager
# ---------------------------------------------------------------------------

def masks_from_params(imp, params, group_names):
    """Masken aus explizit angegebenen Rechtecken (fuer Skripte und Tests)."""

    from ij.gui import Roi

    entries = params.get("rois") or []
    masks = {}
    for index, entry in enumerate(entries):
        name = entry.get("name") or (group_names[index] if index < len(group_names) else None)
        if name is None:
            continue
        x, y, w, h = entry["rect"]
        imp.setRoi(Roi(int(x), int(y), int(w), int(h)))
        masks[name] = imp.createRoiMask()
    imp.killRoi()
    return masks


def masks_from_roi_manager(imp, group_names):
    """{Gruppenname: ByteProcessor}. ROI-Name entscheidet, sonst der Reihe nach."""

    manager = RoiManager.getInstance()
    if manager is None or manager.getCount() == 0:
        return {}

    rois = list(manager.getRoisAsArray())
    names = [manager.getName(i) for i in range(manager.getCount())]

    masks = {}
    for index, group in enumerate(group_names):
        roi = None
        for name, candidate in zip(names, rois):
            if name == group:
                roi = candidate
                break
        if roi is None and len(rois) == len(group_names):
            roi = rois[index]
        if roi is None and len(rois) == 1 and len(group_names) == 1:
            roi = rois[0]
        if roi is None:
            continue

        imp.setRoi(roi)
        mask = imp.createRoiMask()
        masks[group] = mask

    imp.killRoi()
    return masks


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def build_init(imp, params, layers_info):
    """Das Dictionary, das als init.json geschrieben wird."""

    shape = [imp.getHeight(), imp.getWidth()]
    stem_mode = params["imaging_mode"].startswith("STEM")

    lattices = []
    for entry in params["lattices"]:
        item = {"name": entry["name"],
                "type": "file" if entry["kind"] == "aus Datei (.cif/.xyz)" else "custom",
                "a": entry.get("a"),
                "gamma_deg": entry.get("gamma_deg", 120.0),
                "rotation_deg": entry["rotation_deg"],
                "offset_px": list(entry["offset_px"]),
                "z_offset_nm": entry.get("z_offset_nm", 0.0)}
        if item["type"] == "file":
            item["file"] = entry.get("structure_file")
        else:
            item["basis"] = [[s, fa, fb, z] for (s, fa, fb, z) in entry["basis"]]
        lattices.append(item)

    groups = []
    for group in params["groups"]:
        groups.append({
            "name": group["name"],
            "max_shift_px": group["max_shift_px"],
            "max_rotation_deg": group["max_rotation_deg"],
            "init_shift_px": [0.0, 0.0],
            "init_rotation_deg": 0.0,
            "init_scale": [1.0, 1.0],
            "control_points": [10, 10],
            "mask_file": group.get("mask_file"),
            "target_file": group.get("target_file"),
            "isolate_when_active": group.get("isolate_when_active", len(params["groups"]) > 1),
            "init_contrast_factor": 1.0,
        })

    imaging = {
        "mode": "stem_haadf" if stem_mode else "tem",
        "beam_direction": "+z",
        "ctf_parametrization": "cartesian",
        "amp_contrast": params["amp_contrast"],
        "alpha_max_mrad": params["alpha_max_mrad"],
        "aberrations": params["aberrations"],
        "envelope": {"source_size": params["source_size_rad"], "Cc_um": params["cc_um"],
                     "deltaE": params["delta_e"], "instab_defocus_nm": params["instab_nm"],
                     "use_detector": False},
    }
    if stem_mode:
        imaging["stem"] = {"convergence_mrad": params["convergence_mrad"],
                           "detector_mrad": [params["detector_inner"], params["detector_outer"]],
                           "source_size_A": params["stem_source_size_A"],
                           "aberrations": params.get("probe_aberrations") or {}}

    return {
        "version": INIT_VERSION,
        "source": {"plugin": "TEM_Sim_Init", "image_title": imp.getTitle(),
                   "imagej": IJ.getVersion()},
        "shape": shape,
        "pixel_size_nm": params["pixel_size_nm"],
        "voltage_kV": params["voltage_kV"],
        "padding": params["padding"],
        "target_file": "target.tif",
        "target_zero_handling": "median",
        "config_dir": "config_imagej",
        "imaging": imaging,
        "lattices": lattices,
        "groups": groups,
        "layers": layers_info,
        "train": params["train"],
    }


def export(imp, params, layers_info, out_dir):
    """init.json, Zielbild und Masken schreiben. Liefert den Pfad der init.json."""

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    target = imp.duplicate()
    FileSaver(target).saveAsTiff(os.path.join(out_dir, "target.tif"))

    group_names = [g["name"] for g in params["groups"]]
    masks = masks_from_params(imp, params, group_names)
    if not masks:
        masks = masks_from_roi_manager(imp, group_names)
    for group in params["groups"]:
        mask = masks.get(group["name"])
        if mask is None:
            group["mask_file"] = None
            continue
        name = "mask_%s.tif" % group["name"]
        from ij import ImagePlus
        FileSaver(ImagePlus(name, mask)).saveAsTiff(os.path.join(out_dir, name))
        group["mask_file"] = name

    init = build_init(imp, params, layers_info)
    path = os.path.join(out_dir, "init.json")
    handle = open(path, "w")
    try:
        json.dump(init, handle, indent=1, sort_keys=True)
    finally:
        handle.close()
    return path


def start_simulation(params, init_path):
    """mains/imagej_main.py mit der init.json starten."""

    from java.lang import ProcessBuilder
    from java.io import File

    repo = params.get("repo_dir")
    python = params.get("python", "python3")
    script = os.path.join(repo, "mains", "imagej_main.py")

    builder = ProcessBuilder([python, script, init_path])
    builder.directory(File(repo))
    builder.redirectErrorStream(True)
    builder.start()
    IJ.log("Simulation gestartet: %s %s %s" % (python, script, init_path))


# ---------------------------------------------------------------------------
# Schichten und Gruppen aus dem Gitter ableiten
# ---------------------------------------------------------------------------

def layer_names_for(lattice_name, layers):
    """Ein Gitter mit einer Schicht heisst wie das Gitter, sonst Element plus Nummer.

    Fuer S/W/S kommt damit S1, W, S2 heraus - die uebliche Schreibweise.
    """

    if len(layers) == 1:
        return [lattice_name]

    counts = {}
    for layer in layers:
        symbol = layer["elements"][0] if layer["elements"] else "L"
        counts[symbol] = counts.get(symbol, 0) + 1

    running = {}
    names = []
    for layer in layers:
        symbol = layer["elements"][0] if layer["elements"] else "L"
        if counts[symbol] == 1:
            names.append(symbol)
        else:
            running[symbol] = running.get(symbol, 0) + 1
            names.append("%s%d" % (symbol, running[symbol]))
    return names


def layers_info_from(lattice_name, layers, names, entry):
    """Schichtbeschreibungen eines Gitters fuer die init.json."""

    info = []
    for index, layer in enumerate(layers):
        symbol = layer["elements"][0] if layer["elements"] else "C"
        # 0 heisst: Tabellenwert je Element nehmen (S und W schwingen
        # unterschiedlich stark, ein Wert pro Gitter waere zu grob)
        u = entry.get("debye_waller_u")
        if not u:
            u = None
        info.append({
            "name": names[index],
            "element": symbol,
            "z_nm": round(layer["z_nm"] + entry.get("z_offset_nm", 0.0), 6),
            "group": entry["name"],
            "lattice": lattice_name,
            "lattice_layer": index,
            "init_occupancy": entry.get("init_occupancy", 2.0),
            "occ_entropy": entry.get("occ_entropy", True),
            "elements": list(entry.get("mixture", [])),
            "debye_waller_u": u if u is not None else TYPICAL_DEBYE_WALLER.get(symbol, 0.06),
        })
    return info


def parse_aberrations(text):
    """"C10=-13.07, C12=0.35, phi12=0" -> dict."""

    result = {}
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        try:
            result[key.strip()] = float(value)
        except ValueError:
            IJ.log("Aberration nicht lesbar: %s" % chunk)
    return result


def format_aberrations(aberrations):
    keys = ["C10", "C12", "phi12", "C21", "phi21", "C23", "phi23",
            "C30", "C34", "phi34", "C45", "phi45"]
    parts = ["%s=%g" % (k, aberrations[k]) for k in keys if k in aberrations]
    return ", ".join(parts)


def normalize_params(params):
    """Alte Einzelgitter-Angaben in die Liste ueberfuehren."""

    if params.get("lattices"):
        return params

    params["lattices"] = [{
        "name": params.get("group_name", "Gitter"),
        "kind": params.get("lattice_kind", "Graphen"),
        "formula": params.get("formula", "WS2"),
        "structure_file": params.get("structure_file", ""),
        "a": params.get("lattice_a"),
        "gamma_deg": params.get("gamma_deg", 120.0),
        "rotation_deg": params.get("rotation_deg", 0.0),
        "offset_px": params.get("offset_px", (0.0, 0.0)),
        "z_offset_nm": 0.0,
        "max_shift_px": params.get("max_shift_px", 50.0),
        "max_rotation_deg": params.get("max_rotation_deg", 20.0),
        "debye_waller_u": params.get("debye_waller_u"),
        "init_occupancy": params.get("init_occupancy", 100.0),
        "mixture": params.get("mixture", []),
        "isolate": False,
    }]
    return params


def finalize_params(params, imp):
    """Basis, Schichten und Gruppen aus allen Gittern ableiten."""

    normalize_params(params)
    shape = (imp.getHeight(), imp.getWidth())

    layers_info = []
    groups = []
    for entry in params["lattices"]:
        a, basis = resolve_lattice(entry)
        entry["a"] = a
        entry["basis"] = basis

        if basis:
            layers = lattice_layers_for(entry, params, shape, limit=1)
            names = layer_names_for(entry["name"], layers)
            layers_info.extend(layers_info_from(entry["name"], layers, names, entry))
        else:
            IJ.log("Gitter %s kommt aus einer Datei - Schichten legt die Python-Seite an."
                   % entry["name"])

        groups.append({
            "name": entry["name"],
            "max_shift_px": entry.get("max_shift_px", 50.0),
            "max_rotation_deg": entry.get("max_rotation_deg", 20.0),
            "isolate_when_active": entry.get("isolate", len(params["lattices"]) > 1),
            "control_points": entry.get("control_points", [10, 10]),
        })

    params["layers_info"] = layers_info
    params["groups"] = groups
    return params


def default_train_block(params):
    groups = [g["name"] for g in params["groups"]]
    first = groups[0] if groups else None
    enable = ["train_normalisation_parameters"]
    rates = {}
    for name in groups:
        enable += ["groups.%s.shift" % name, "groups.%s.rotation" % name,
                   "groups.%s.scale" % name]
        rates["groups.%s.shift" % name] = 0.05
        rates["groups.%s.rotation" % name] = 1e-3
        rates["groups.%s.scale" % name] = 1e-3
    return {"active_group": first, "enable": enable, "learning_rates": rates,
            "losses": {"loss_ncc": 0.5, "loss_var": 0.5, "loss_l1": 0.1}}


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class Fields(object):
    """Merkt sich die Reihenfolge der numerischen Felder eines Dialogs."""

    def __init__(self):
        self.order = []

    def add(self, gd, label, value, digits):
        self.order.append(label)
        gd.addNumericField(label, value, digits)

    def read(self, gd):
        return dict((label, gd.getNextNumber()) for label in self.order)

    def position(self, label):
        return self.order.index(label)


def show_global_dialog(imp):
    """Bild, Abbildung und Ausgabe - alles, was nicht am einzelnen Gitter haengt."""

    from ij.gui import NonBlockingGenericDialog

    info = imp.getOriginalFileInfo()
    base_dir = info.directory if (info is not None and info.directory) else (IJ.getDirectory("home") or "")
    repo_guess = os.path.dirname(os.path.dirname(os.path.abspath(
        __file__ if "__file__" in dir() else ".")))

    fields = Fields()
    gd = NonBlockingGenericDialog("Simulation initialisieren (1/2)")
    gd.addMessage("Bild: %s   %d x %d px" % (imp.getTitle(), imp.getWidth(), imp.getHeight()))

    fields.add(gd, "Pixelgroesse [nm] (0 = aus FFT)", 0.0, 6)
    fields.add(gd, "Hochspannung [kV]", 80, 1)
    fields.add(gd, "Padding [px]", 100, 0)
    fields.add(gd, "Anzahl Gitter", 1, 0)

    gd.addMessage("--- Abbildung ---")
    gd.addChoice("Modus", IMAGING_MODES, IMAGING_MODES[0])
    fields.add(gd, "Amplitudenkontrast", 0.1, 3)
    fields.add(gd, "Aperturwinkel alpha_max [mrad]", 50.0, 1)
    gd.addStringField("Aberrationen (TEM)", format_aberrations(DEFAULT_ABERRATIONS), 60)
    fields.add(gd, "Quellgroesse [mrad]", 0.1, 3)
    fields.add(gd, "Energiebreite [eV]", 0.35, 3)
    fields.add(gd, "Cc [um]", 1000.0, 1)
    fields.add(gd, "Defokus-Instabilitaet [nm]", 0.01, 3)

    gd.addMessage("--- nur STEM ---")
    gd.addStringField("Aberrationen der Sonde", "C10=0, C12=0, phi12=0, C30=0.001", 60)
    fields.add(gd, "Konvergenzwinkel [mrad]", 25.0, 1)
    fields.add(gd, "Detektor innen [mrad]", 60.0, 1)
    fields.add(gd, "Detektor aussen [mrad]", 200.0, 1)
    fields.add(gd, "Quellgroesse Sonde [A]", 0.4, 3)

    gd.addMessage("--- Ausgabe ---")
    gd.addStringField("Ordner", base_dir, 40)
    gd.addStringField("Repository", repo_guess, 40)
    gd.addStringField("Python", "python3", 20)
    gd.addCheckbox("Gitter als Overlay anzeigen", True)
    gd.addCheckbox("Simulation nach dem Export starten", False)

    gd.showDialog()
    if gd.wasCanceled():
        return None

    numbers = fields.read(gd)
    mode = gd.getNextChoice()
    aberrations = gd.getNextString()
    probe_aberrations = gd.getNextString()
    out_dir = gd.getNextString()
    repo_dir = gd.getNextString()
    python = gd.getNextString()
    show_overlay = gd.getNextBoolean()
    run_simulation = gd.getNextBoolean()

    return {
        "pixel_size_nm": numbers["Pixelgroesse [nm] (0 = aus FFT)"],
        "voltage_kV": numbers["Hochspannung [kV]"],
        "padding": int(numbers["Padding [px]"]),
        "lattice_count": max(1, int(numbers["Anzahl Gitter"])),
        "imaging_mode": mode,
        "amp_contrast": numbers["Amplitudenkontrast"],
        "alpha_max_mrad": numbers["Aperturwinkel alpha_max [mrad]"],
        "aberrations": parse_aberrations(aberrations),
        "probe_aberrations": parse_aberrations(probe_aberrations),
        "source_size_rad": numbers["Quellgroesse [mrad]"] * 1e-3,
        "delta_e": numbers["Energiebreite [eV]"],
        "cc_um": numbers["Cc [um]"],
        "instab_nm": numbers["Defokus-Instabilitaet [nm]"],
        "convergence_mrad": numbers["Konvergenzwinkel [mrad]"],
        "detector_inner": numbers["Detektor innen [mrad]"],
        "detector_outer": numbers["Detektor aussen [mrad]"],
        "stem_source_size_A": numbers["Quellgroesse Sonde [A]"],
        "out_dir": out_dir, "repo_dir": repo_dir, "python": python,
        "show_overlay": show_overlay, "run_simulation": run_simulation,
    }


def show_lattice_dialog(imp, index, count, params):
    """Ein Gitter: Typ, Orientierung, Suchbereich, Schichteigenschaften."""

    from java.awt.event import ActionListener
    from ij.gui import NonBlockingGenericDialog

    defaults = ["Gr", "WS2", "Gitter3", "Gitter4"]
    fields = Fields()
    gd = NonBlockingGenericDialog("Gitter %d von %d (2/2)" % (index + 1, count))

    gd.addStringField("Name (wird zur Gruppe)", defaults[index] if index < len(defaults)
                      else "Gitter%d" % (index + 1), 20)
    gd.addChoice("Typ", LATTICE_KINDS, LATTICE_KINDS[0] if index == 0 else LATTICE_KINDS[2])
    gd.addStringField("Formel (MX2)", "WS2", 10)
    gd.addStringField("Strukturdatei", "", 30)

    fields.add(gd, "Gitterkonstante a [A] (0 = Vorgabe)", 0.0, 4)
    fields.add(gd, "Orientierung [Grad]", 0.0, 3)
    fields.add(gd, "Versatz Zeile [px]", 0.0, 2)
    fields.add(gd, "Versatz Spalte [px]", 0.0, 2)
    fields.add(gd, "z-Versatz [nm]", 0.0, 3)

    gd.addMessage("--- Suchbereich und Schicht ---")
    fields.add(gd, "max. Verschiebung [px]", 50.0, 1)
    fields.add(gd, "max. Drehung [Grad]", 20.0 if index == 0 else 5.0, 1)
    fields.add(gd, "Debye-Waller u_rms [A] (0 = je Element)", 0.0, 3)
    fields.add(gd, "Start-Occupancy (Logit)", 100.0 if index == 0 else 2.0, 1)
    gd.addStringField("Elementmischung (Komma, leer = aus)", "", 20)
    gd.addCheckbox("bei aktiver Gruppe allein simulieren", count > 1)

    gd.addMessage("--- FFT-Suche (0 = automatisch) ---")
    fields.add(gd, "kleinster Radius [px]", 0.0, 0)
    fields.add(gd, "groesster Radius [px]", 0.0, 0)

    class FftButton(ActionListener):
        def actionPerformed(self, event):
            numeric = gd.getNumericFields()
            try:
                a = float(numeric.get(fields.position("Gitterkonstante a [A] (0 = Vorgabe)")).getText())
            except Exception:
                a = 0.0
            kind = LATTICE_KINDS[gd.getChoices().get(0).getSelectedIndex()]
            if a <= 0:
                a = resolve_lattice({"kind": kind,
                                     "formula": gd.getStringFields().get(1).getText()})[0]
            try:
                min_r = float(numeric.get(fields.position("kleinster Radius [px]")).getText())
                max_r = float(numeric.get(fields.position("groesster Radius [px]")).getText())
            except Exception:
                min_r, max_r = 0.0, 0.0

            estimate = estimate_from_fft(imp, a, 120.0, min_r or 6.0, max_r or None)
            if estimate is None:
                IJ.showMessage("Kein Gitterreflex gefunden.")
                return
            numeric.get(fields.position("Orientierung [Grad]")).setText("%.3f" % estimate["rotation_deg"])
            IJ.log("FFT (%s): Reflex bei r = %.1f px -> Orientierung %.2f Grad, "
                   "Pixelgroesse waere %.6f nm"
                   % (kind, estimate["peak"]["radius_px"], estimate["rotation_deg"],
                      estimate["pixel_size_nm"]))

    gd.addButton("Orientierung aus FFT", FftButton())
    gd.showDialog()
    if gd.wasCanceled():
        return None

    numbers = fields.read(gd)
    name = gd.getNextString()
    kind = gd.getNextChoice()
    formula = gd.getNextString()
    structure_file = gd.getNextString()
    mixture = gd.getNextString()
    isolate = gd.getNextBoolean()

    return {
        "name": name,
        "kind": kind,
        "formula": formula,
        "structure_file": structure_file,
        "a": numbers["Gitterkonstante a [A] (0 = Vorgabe)"] or None,
        "gamma_deg": 120.0,
        "rotation_deg": numbers["Orientierung [Grad]"],
        "offset_px": (numbers["Versatz Zeile [px]"], numbers["Versatz Spalte [px]"]),
        "z_offset_nm": numbers["z-Versatz [nm]"],
        "max_shift_px": numbers["max. Verschiebung [px]"],
        "max_rotation_deg": numbers["max. Drehung [Grad]"],
        "debye_waller_u": numbers["Debye-Waller u_rms [A] (0 = je Element)"],
        "init_occupancy": numbers["Start-Occupancy (Logit)"],
        "mixture": [s.strip() for s in mixture.split(",") if s.strip()],
        "isolate": isolate,
        "fft_min_radius_px": numbers["kleinster Radius [px]"],
        "fft_max_radius_px": numbers["groesster Radius [px]"],
    }


def show_dialog(imp):
    params = show_global_dialog(imp)
    if params is None:
        return None

    lattices = []
    for index in range(params["lattice_count"]):
        entry = show_lattice_dialog(imp, index, params["lattice_count"], params)
        if entry is None:
            return None
        lattices.append(entry)

    params["lattices"] = lattices
    return params


# ---------------------------------------------------------------------------
# Ablauf
# ---------------------------------------------------------------------------

def run(imp, params):
    """Gemeinsamer Weg fuer Dialog und Parameterdatei."""

    normalize_params(params)

    if params.get("fft_estimate") or not params.get("pixel_size_nm"):
        first = params["lattices"][0]
        a = first.get("a") or resolve_lattice(first)[0]
        estimate = estimate_from_fft(imp, a, first.get("gamma_deg", 120.0),
                                     first.get("fft_min_radius_px") or 6.0,
                                     first.get("fft_max_radius_px") or None)
        if estimate is None:
            IJ.log("FFT: kein Gitterreflex gefunden, Werte bleiben unveraendert")
        else:
            params["pixel_size_nm"] = estimate["pixel_size_nm"]
            if params.get("fft_estimate"):
                params["lattices"][0]["rotation_deg"] = estimate["rotation_deg"]
            IJ.log("FFT: Reflex bei r = %.2f px (d = %.2f px, %.2f A) -> "
                   "Pixel %.6f nm, Orientierung %.3f Grad"
                   % (estimate["peak"]["radius_px"], estimate["d_px"],
                      estimate["d_theory_A"], estimate["pixel_size_nm"],
                      estimate["rotation_deg"]))
            IJ.log("FFT: Peakwinkel %.3f Grad, FFT-Groesse %d"
                   % (estimate["peak"]["angle_deg"], estimate["fft_size"]))

    params = finalize_params(params, imp)
    params.setdefault("train", default_train_block(params))

    if params.get("show_overlay", True):
        count = update_overlay(imp, params)
        IJ.log("Overlay: %d Gitterpunkte" % count)

    init_path = export(imp, params, params["layers_info"], params["out_dir"])
    IJ.log("geschrieben: %s" % init_path)
    IJ.log("  Schichten: %s" % ", ".join(l["name"] for l in params["layers_info"]))
    IJ.log("  Gruppen:   %s" % ", ".join(g["name"] for g in params["groups"]))

    if params.get("run_simulation"):
        start_simulation(params, init_path)
    return init_path


def main():
    if params_path:
        handle = open(params_path)
        try:
            params = json.load(handle)
        finally:
            handle.close()

        image_path = params.pop("image_path", None)
        imp = IJ.openImage(image_path) if image_path else WindowManager.getCurrentImage()
        if imp is None:
            IJ.log("Kein Bild: image_path in der Parameterdatei setzen.")
            return
        params.setdefault("show_overlay", False)
        run(imp, params)
        return

    imp = WindowManager.getCurrentImage()
    if imp is None:
        IJ.showMessage("Simulation initialisieren",
                       "Bitte zuerst ein Bild oeffnen - das Plugin arbeitet auf dem aktiven Fenster.")
        return

    params = show_dialog(imp)
    if params is None:
        return
    run(imp, params)


main()
