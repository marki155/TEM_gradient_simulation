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

INIT_VERSION = 1

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


def find_lattice_peak(spectrum, size, min_radius_px=6, max_radius_frac=0.45):
    """Staerkster Reflex ausserhalb des Zentrums, subpixelgenau ueber den Schwerpunkt."""

    centre = size // 2
    best, best_value = None, -1.0
    limit = max_radius_frac * size

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


def estimate_from_fft(imp, lattice_a_A, gamma_deg=120.0):
    """Pixelgroesse und Orientierung aus dem staerksten Gitterreflex.

    Bei hexagonalen Gittern liegt der erste Reflex bei d = a * sin(gamma), also
    2.13 A fuer Graphen - und seine Richtung ist gegen die Gitterachse um
    90 - gamma/2 verdreht (30 Grad bei gamma = 120).
    """

    spectrum, size = power_spectrum(imp)
    peak = find_lattice_peak(spectrum, size)
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

def lattice_layers_for(params, shape, margin_px=0.0, limit=None):
    kind = params["lattice_kind"]
    if kind == "aus Datei (.cif/.xyz)":
        return []
    a, basis = basis_for(kind, params.get("formula", "WS2"),
                         {"a": params.get("custom_a", 3.0), "basis": params.get("custom_basis", [])})
    if params.get("lattice_a"):
        a = params["lattice_a"]
    return layer_points(a, a, params.get("gamma_deg", 120.0), basis,
                        params["pixel_size_nm"], shape, params["rotation_deg"],
                        params["offset_px"], margin_px=margin_px, limit=limit)


def update_overlay(imp, params, limit=4000):
    """Gitterpunkte als Overlay ueber das Bild legen."""

    shape = (imp.getHeight(), imp.getWidth())
    layers = lattice_layers_for(params, shape, limit=limit)
    overlay = Overlay()
    colors = [Color.yellow, Color.cyan, Color.magenta, Color.green, Color.orange]

    for index, layer in enumerate(layers):
        xs = [p[1] for p in layer["points"]]     # Spalte -> x
        ys = [p[0] for p in layer["points"]]     # Zeile  -> y
        if not xs:
            continue
        roi = PointRoi(xs, ys, len(xs))
        roi.setStrokeColor(colors[index % len(colors)])
        roi.setPointType(0)
        roi.setSize(0)
        overlay.add(roi)

    imp.setOverlay(overlay)
    return sum(len(l["points"]) for l in layers)


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

    lattice = {"type": params["lattice_type"],
               "a": params["lattice_a"],
               "gamma_deg": params.get("gamma_deg", 120.0),
               "rotation_deg": params["rotation_deg"],
               "offset_px": list(params["offset_px"])}
    if params["lattice_type"] == "file":
        lattice["file"] = params.get("structure_file")
    else:
        lattice["basis"] = [[s, fa, fb, z] for (s, fa, fb, z) in params["basis"]]

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
                           "source_size_A": params["stem_source_size_A"]}

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
        "lattice": lattice,
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

def default_layer_names(layers):
    """Sprechende Namen: Element plus laufende Nummer, wenn mehrfach."""

    counts = {}
    names = []
    for layer in layers:
        symbol = layer["elements"][0] if layer["elements"] else "L"
        counts[symbol] = counts.get(symbol, 0) + 1
        names.append(symbol if counts[symbol] == 1 else "%s%d" % (symbol, counts[symbol]))

    # zweites C wird zu C2 usw. - aber ein einzelnes bleibt schlicht
    seen = {}
    for layer in layers:
        symbol = layer["elements"][0] if layer["elements"] else "L"
        seen[symbol] = seen.get(symbol, 0) + 1
    for index, layer in enumerate(layers):
        symbol = layer["elements"][0] if layer["elements"] else "L"
        if seen[symbol] == 1:
            names[index] = symbol
    return names


def layers_info_from(layers, names, groups_per_layer, mixture, occupancy, debye_waller):
    info = []
    for index, layer in enumerate(layers):
        symbol = layer["elements"][0] if layer["elements"] else "C"
        info.append({
            "name": names[index],
            "element": symbol,
            "z_nm": round(layer["z_nm"], 6),
            "group": groups_per_layer[index],
            "init_occupancy": occupancy,
            "occ_entropy": True,
            "elements": list(mixture),
            "debye_waller_u": debye_waller if debye_waller is not None
            else TYPICAL_DEBYE_WALLER.get(symbol, 0.06),
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


LATTICE_TYPE_BY_KIND = {
    "Graphen": "custom", "hBN": "custom", "2H-MX2 (WS2, MoS2 ...)": "custom",
    "eigene Zelle": "custom", "aus Datei (.cif/.xyz)": "file",
}


def finalize_params(params, imp):
    """Basis, Schichten und Gruppen aus den Dialogwerten ableiten."""

    kind = params["lattice_kind"]
    params["lattice_type"] = LATTICE_TYPE_BY_KIND.get(kind, "custom")

    if kind == "aus Datei (.cif/.xyz)":
        params["basis"] = []
        params["layers_info"] = params.get("layers_info", [])
        return params

    a, basis = basis_for(kind, params.get("formula", "WS2"),
                         {"a": params.get("custom_a", 3.0), "basis": params.get("custom_basis", [])})
    if not params.get("lattice_a"):
        params["lattice_a"] = a
    params["basis"] = basis

    shape = (imp.getHeight(), imp.getWidth())
    layers = lattice_layers_for(params, shape, limit=1)
    names = params.get("layer_names") or default_layer_names(layers)

    if params.get("one_group_per_layer"):
        groups_per_layer = list(names)
    else:
        groups_per_layer = [params.get("group_name", "Struktur")] * len(layers)

    params["layers_info"] = layers_info_from(
        layers, names, groups_per_layer, params.get("mixture", []),
        params.get("init_occupancy", 100.0), params.get("debye_waller_u"))

    group_names = []
    for name in groups_per_layer:
        if name not in group_names:
            group_names.append(name)
    params["groups"] = [{"name": name,
                         "max_shift_px": params["max_shift_px"],
                         "max_rotation_deg": params["max_rotation_deg"],
                         "isolate_when_active": len(group_names) > 1}
                        for name in group_names]
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

FIELD = {}          # Name -> Index im GenericDialog (numerische Felder)


def _add_number(gd, label, value, digits, key):
    FIELD[key] = len([k for k in FIELD])
    gd.addNumericField(label, value, digits)


def show_dialog(imp):
    from java.awt.event import ActionListener
    from ij.gui import NonBlockingGenericDialog

    shape = (imp.getHeight(), imp.getWidth())
    base_dir = None
    info = imp.getOriginalFileInfo()
    if info is not None and info.directory:
        base_dir = info.directory
    if not base_dir:
        base_dir = IJ.getDirectory("home") or ""

    repo_guess = os.path.dirname(os.path.dirname(os.path.abspath(
        __file__ if "__file__" in dir() else ".")))

    FIELD.clear()
    gd = NonBlockingGenericDialog("Simulation initialisieren")
    gd.addMessage("Bild: %s   %d x %d px" % (imp.getTitle(), shape[1], shape[0]))

    _add_number(gd, "Pixelgroesse [nm]", 0.015, 6, "pixel_size_nm")
    _add_number(gd, "Hochspannung [kV]", 80, 1, "voltage_kV")
    _add_number(gd, "Padding [px]", 100, 0, "padding")

    gd.addMessage("--- Gitter ---")
    gd.addChoice("Typ", LATTICE_KINDS, LATTICE_KINDS[0])
    gd.addStringField("Formel (MX2)", "WS2", 10)
    gd.addStringField("Strukturdatei", "", 30)
    _add_number(gd, "Gitterkonstante a [A] (0 = Vorgabe)", 0.0, 4, "lattice_a")
    _add_number(gd, "Orientierung [Grad]", 0.0, 3, "rotation_deg")
    _add_number(gd, "Versatz Zeile [px]", 0.0, 2, "offset_x")
    _add_number(gd, "Versatz Spalte [px]", 0.0, 2, "offset_y")

    gd.addMessage("--- Suchbereich und Schichten ---")
    _add_number(gd, "max. Verschiebung [px]", 50.0, 1, "max_shift_px")
    _add_number(gd, "max. Drehung [Grad]", 20.0, 1, "max_rotation_deg")
    _add_number(gd, "Debye-Waller u_rms [A]", 0.06, 3, "debye_waller_u")
    _add_number(gd, "Start-Occupancy (Logit)", 100.0, 1, "init_occupancy")
    gd.addStringField("Elementmischung (Komma, leer = aus)", "", 20)
    gd.addStringField("Gruppenname", "Struktur", 20)
    gd.addCheckbox("eine Gruppe je Schicht", False)

    gd.addMessage("--- Abbildung ---")
    gd.addChoice("Modus", IMAGING_MODES, IMAGING_MODES[0])
    _add_number(gd, "Amplitudenkontrast", 0.1, 3, "amp_contrast")
    _add_number(gd, "Aperturwinkel alpha_max [mrad]", 50.0, 1, "alpha_max_mrad")
    gd.addStringField("Aberrationen", format_aberrations(DEFAULT_ABERRATIONS), 60)
    _add_number(gd, "Quellgroesse [mrad]", 0.1, 3, "source_size_mrad")
    _add_number(gd, "Energiebreite [eV]", 0.35, 3, "delta_e")
    _add_number(gd, "Cc [um]", 1000.0, 1, "cc_um")
    _add_number(gd, "Defokus-Instabilitaet [nm]", 0.01, 3, "instab_nm")

    gd.addMessage("--- nur STEM ---")
    _add_number(gd, "Konvergenzwinkel [mrad]", 25.0, 1, "convergence_mrad")
    _add_number(gd, "Detektor innen [mrad]", 60.0, 1, "detector_inner")
    _add_number(gd, "Detektor aussen [mrad]", 200.0, 1, "detector_outer")
    _add_number(gd, "Quellgroesse Sonde [A]", 0.4, 3, "stem_source_size_A")

    gd.addMessage("--- Ausgabe ---")
    gd.addStringField("Ordner", base_dir, 40)
    gd.addStringField("Repository", repo_guess, 40)
    gd.addStringField("Python", "python3", 20)
    gd.addCheckbox("Gitter als Overlay anzeigen", True)
    gd.addCheckbox("Simulation nach dem Export starten", False)

    class FftButton(ActionListener):
        def actionPerformed(self, event):
            fields = gd.getNumericFields()
            try:
                a = float(fields.get(FIELD["lattice_a"]).getText())
            except Exception:
                a = 0.0
            kind = LATTICE_KINDS[gd.getChoices().get(0).getSelectedIndex()]
            if a <= 0:
                a = basis_for(kind, gd.getStringFields().get(0).getText())[0]
            estimate = estimate_from_fft(imp, a)
            if estimate is None:
                IJ.showMessage("Kein Gitterreflex gefunden.")
                return
            fields.get(FIELD["pixel_size_nm"]).setText("%.6f" % estimate["pixel_size_nm"])
            fields.get(FIELD["rotation_deg"]).setText("%.3f" % estimate["rotation_deg"])
            IJ.log("FFT: d = %.2f px -> Pixel %.5f nm, Orientierung %.2f Grad"
                   % (estimate["d_px"], estimate["pixel_size_nm"], estimate["rotation_deg"]))

    gd.addButton("Gitter aus FFT schaetzen", FftButton())
    gd.showDialog()
    if gd.wasCanceled():
        return None

    numbers = [gd.getNextNumber() for _ in range(len(FIELD))]
    def number(key):
        return numbers[FIELD[key]]

    kind = gd.getNextChoice()
    formula = gd.getNextString()
    structure_file = gd.getNextString()
    mixture_text = gd.getNextString()
    group_name = gd.getNextString()
    one_group_per_layer = gd.getNextBoolean()
    mode = gd.getNextChoice()
    aberration_text = gd.getNextString()
    out_dir = gd.getNextString()
    repo_dir = gd.getNextString()
    python = gd.getNextString()
    show_overlay = gd.getNextBoolean()
    run_simulation = gd.getNextBoolean()

    params = {
        "pixel_size_nm": number("pixel_size_nm"),
        "voltage_kV": number("voltage_kV"),
        "padding": int(number("padding")),
        "lattice_kind": kind,
        "formula": formula,
        "structure_file": structure_file,
        "lattice_a": number("lattice_a") or None,
        "gamma_deg": 120.0,
        "rotation_deg": number("rotation_deg"),
        "offset_px": (number("offset_x"), number("offset_y")),
        "max_shift_px": number("max_shift_px"),
        "max_rotation_deg": number("max_rotation_deg"),
        "debye_waller_u": number("debye_waller_u"),
        "init_occupancy": number("init_occupancy"),
        "mixture": [s.strip() for s in mixture_text.split(",") if s.strip()],
        "group_name": group_name,
        "one_group_per_layer": one_group_per_layer,
        "imaging_mode": mode,
        "amp_contrast": number("amp_contrast"),
        "alpha_max_mrad": number("alpha_max_mrad"),
        "aberrations": parse_aberrations(aberration_text),
        "source_size_rad": number("source_size_mrad") * 1e-3,
        "delta_e": number("delta_e"),
        "cc_um": number("cc_um"),
        "instab_nm": number("instab_nm"),
        "convergence_mrad": number("convergence_mrad"),
        "detector_inner": number("detector_inner"),
        "detector_outer": number("detector_outer"),
        "stem_source_size_A": number("stem_source_size_A"),
        "out_dir": out_dir,
        "repo_dir": repo_dir,
        "python": python,
        "show_overlay": show_overlay,
        "run_simulation": run_simulation,
    }
    return params


# ---------------------------------------------------------------------------
# Ablauf
# ---------------------------------------------------------------------------

def run(imp, params):
    """Gemeinsamer Weg fuer Dialog und Parameterdatei."""

    if params.get("fft_estimate"):
        a = params.get("lattice_a") or basis_for(params["lattice_kind"],
                                                 params.get("formula", "WS2"))[0]
        estimate = estimate_from_fft(imp, a, params.get("gamma_deg", 120.0))
        if estimate is None:
            IJ.log("FFT: kein Gitterreflex gefunden, Werte bleiben unveraendert")
        else:
            params["pixel_size_nm"] = estimate["pixel_size_nm"]
            params["rotation_deg"] = estimate["rotation_deg"]
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
