"""Test der ganzen Kette: Fiji-Plugin -> init.json -> Simulationsmodell.

Erzeugt ein synthetisches Bild mit bekanntem Gitter, laesst das Plugin es
kopflos initialisieren (inklusive FFT-Schaetzung) und prueft, ob das daraus
gebaute Modell dasselbe Bild reproduziert.

    python3 imagej/test_imagej_chain.py [--fiji /pfad/zu/Fiji.app]

Ohne Fiji wird der Plugin-Teil uebersprungen und nur die Python-Seite geprueft.
"""

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tifffile
import torch

DEFAULT_FIJI = os.path.expanduser("~/Fiji.app")
TRUE_ROTATION = -12.0
TRUE_PIXEL_NM = 0.0150
SHAPE = (768, 768)


def make_test_image(path):
    """Bild aus einem bekannten Graphengitter simulieren."""

    from ctf import DEFAULT_ABERRATIONS
    from lattice import LatticeSpec, build_layer_positions
    from layers import LayerConfig, LayerGroupConfig
    from nn import NN_particles, ParticlesConfig
    from training_parameter import TrainMethod

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    positions = build_layer_positions(
        LatticeSpec(type="graphene", a=2.46, rotation_deg=TRUE_ROTATION),
        SHAPE, TRUE_PIXEL_NM, max_shift_px=60, max_rotation_deg=20)[0]["positions"]

    layer = LayerConfig("Gr", "C", 0.0, "Gr", init_occupancy=100.0,
                        occ_entropy=False, debye_waller_u=0.06)
    group = LayerGroupConfig("Gr", max_shift=1.0, max_rotation=0.01, isolate_when_active=True)
    config = ParticlesConfig(
        init_positions=[positions], init_intensities=[np.full(len(positions), 6.0)],
        shape=SHAPE, pixel_size_nm=TRUE_PIXEL_NM, voltage_kV=80,
        init_aberrations=dict(DEFAULT_ABERRATIONS),
        init_ctf_env_params={"source_size": 1e-4, "Cc_um": 1e3, "deltaE": 0.35,
                             "instab_defocus_nm": 0.01, "use_detector": False},
        init_amp_contrast=0.1, ctf_alpha_max_mrad=50.0, padding=64,
        layers=[layer], groups=[group])
    train_method = TrainMethod().ensure_for_config(config)
    model = NN_particles(config=config, pixel_size_nm=TRUE_PIXEL_NM, voltage_kV=80,
                         train_method=train_method, device=device, train_methods=train_method)
    model.build()

    image = model.model(None)[0].detach().cpu().numpy()
    image = (image - image.mean()) / image.std()
    tifffile.imwrite(path, image.astype(np.float32), imagej=True)
    return image


def plugin_parameters(image_path, out_dir, stem=False, fft=True):
    aberrations = ({"C10": 0.0, "C12": 0.0, "phi12": 0.0, "C30": 0.001} if stem else
                   {"C10": -13.07, "C12": 0.35, "phi12": 0.0, "C21": 3.71, "phi21": 0.0,
                    "C23": 0.17, "phi23": 0.0, "C30": 4.16, "C34": -0.2, "phi34": 0.0,
                    "C45": 0.25, "phi45": 0.0})
    return {
        "image_path": image_path,
        "pixel_size_nm": 0.0 if fft else TRUE_PIXEL_NM, "voltage_kV": 80, "padding": 64,
        "fft_estimate": fft,
        "lattice_kind": "Graphen", "formula": "WS2", "structure_file": "",
        "lattice_a": 2.46, "gamma_deg": 120.0,
        "rotation_deg": 0.0 if fft else TRUE_ROTATION, "offset_px": [0.0, 0.0],
        "max_shift_px": 60.0, "max_rotation_deg": 20.0,
        "debye_waller_u": 0.06, "init_occupancy": 100.0,
        "mixture": ["B", "C", "N", "O"], "group_name": "Gr", "one_group_per_layer": False,
        "imaging_mode": "STEM HAADF (Sonde x Objekt)" if stem else "TEM (Multislice + CTF)",
        "amp_contrast": 0.1, "alpha_max_mrad": 50.0, "aberrations": aberrations,
        "source_size_rad": 1e-4, "delta_e": 0.35, "cc_um": 1000.0, "instab_nm": 0.01,
        "convergence_mrad": 25.0, "detector_inner": 60.0, "detector_outer": 200.0,
        "stem_source_size_A": 0.4,
        "rois": [{"name": "Gr", "rect": [100, 100, 500, 500]}],
        "out_dir": out_dir,
        "repo_dir": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "python": sys.executable, "show_overlay": False, "run_simulation": False,
    }


def run_plugin(fiji_dir, params_path):
    script = os.path.join(fiji_dir, "scripts", "Plugins", "TEM_Simulation", "TEM_Sim_Init.py")
    if not os.path.exists(script):
        raise RuntimeError(f"Plugin nicht installiert: {script}")

    result = subprocess.run(
        [os.path.join(fiji_dir, "ImageJ-linux64"), "--headless", "--console",
         "--run", script, "params_path='%s'" % params_path],
        cwd=fiji_dir, capture_output=True, text=True, timeout=600)
    return result.stdout + result.stderr


def check_model(init_path, reference):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mains"))
    from imagej_main import build_everything

    _init, _base, config, _tm, target, model = build_everything(init_path)
    simulated, target_used, _p, _loss = model.loss_multislice(target)

    a = simulated.detach().cpu().numpy()
    b = target_used.detach().cpu().numpy()
    margin = config.padding
    a = a[margin:-margin, margin:-margin].ravel()
    b = b[margin:-margin, margin:-margin].ravel()
    ncc = float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))
    return config, ncc


def main():
    fiji_dir = DEFAULT_FIJI
    if "--fiji" in sys.argv:
        fiji_dir = sys.argv[sys.argv.index("--fiji") + 1]

    work = tempfile.mkdtemp(prefix="imagej_chain_")
    image_path = os.path.join(work, "testbild.tif")
    make_test_image(image_path)
    print(f"Testbild: {SHAPE}, Gitter {TRUE_ROTATION} Grad, Pixel {TRUE_PIXEL_NM} nm")

    if not os.path.exists(os.path.join(fiji_dir, "ImageJ-linux64")):
        print(f"Fiji nicht gefunden ({fiji_dir}) - Plugin-Teil uebersprungen")
        return 0

    # 1) exakte Werte: die Kette muss das Bild identisch reproduzieren
    # 2) FFT-Schaetzung: Orientierung und Pixelgroesse muessen getroffen werden
    for label, stem, fft in (("TEM exakt", False, False),
                             ("TEM aus FFT", False, True),
                             ("STEM aus FFT", True, True)):
        key = label.lower().replace(" ", "_")
        out_dir = os.path.join(work, "export_" + key)
        params_path = os.path.join(work, "params_%s.json" % key)
        with open(params_path, "w") as handle:
            json.dump(plugin_parameters(image_path, out_dir, stem, fft), handle)

        log = run_plugin(fiji_dir, params_path)
        init_path = os.path.join(out_dir, "init.json")
        assert os.path.exists(init_path), "Plugin hat keine init.json geschrieben:\n" + log

        with open(init_path) as handle:
            init = json.load(handle)

        rotation = init["lattices"][0]["rotation_deg"]
        pixel = init["pixel_size_nm"]
        assert abs(rotation - TRUE_ROTATION) < 1.0, rotation
        assert abs(pixel - TRUE_PIXEL_NM) / TRUE_PIXEL_NM < 0.02, pixel
        assert init["groups"][0]["mask_file"], "Maske fehlt"
        assert os.path.exists(os.path.join(out_dir, "target.tif"))

        config, ncc = check_model(init_path, None)
        print(f"  {label:12s}: {rotation:+7.3f} Grad (wahr {TRUE_ROTATION:+.1f}), "
              f"{pixel:.6f} nm (wahr {TRUE_PIXEL_NM}), {config.imaging_mode}, NCC {ncc:.4f}")

        assert len(init.get("lattices", [])) == 1, init.get("lattices")

        if not stem and not fft:
            # gleiche Werte rein, gleiches Bild raus
            assert ncc > 0.999, ncc
        elif not stem:
            # mit geschaetztem Gitter laeuft die Phase ueber das Feld leicht weg
            assert ncc > 0.85, ncc

    check_two_lattices(work, image_path, fiji_dir)

    print("Kette Plugin -> init.json -> Modell funktioniert")
    return 0


def check_two_lattices(work, image_path, fiji_dir):
    """Zwei unabhaengig orientierte Gitter (Graphen + WS2) in einem Lauf."""

    out_dir = os.path.join(work, "export_zwei")
    params = plugin_parameters(image_path, out_dir, stem=False, fft=True)
    params["lattices"] = [
        {"name": "Gr", "kind": "Graphen", "a": 2.46, "gamma_deg": 120.0, "rotation_deg": 0.0,
         "offset_px": [0, 0], "z_offset_nm": 0.0, "max_shift_px": 60.0,
         "max_rotation_deg": 20.0, "debye_waller_u": 0.0, "init_occupancy": 100.0,
         "mixture": ["B", "C", "N", "O"], "isolate": True},
        {"name": "WS2", "kind": "2H-MX2 (WS2, MoS2 ...)", "formula": "WS2", "a": None,
         "gamma_deg": 120.0, "rotation_deg": 17.0, "offset_px": [25.0, -10.0],
         "z_offset_nm": 0.33, "max_shift_px": 80.0, "max_rotation_deg": 5.0,
         "debye_waller_u": 0.0, "init_occupancy": 2.0, "mixture": [], "isolate": False},
    ]
    params["rois"] = [{"name": "Gr", "rect": [50, 50, 600, 600]},
                      {"name": "WS2", "rect": [200, 200, 300, 300]}]

    params_path = os.path.join(work, "params_zwei.json")
    with open(params_path, "w") as handle:
        json.dump(params, handle)

    log = run_plugin(fiji_dir, params_path)
    init_path = os.path.join(out_dir, "init.json")
    assert os.path.exists(init_path), "Plugin hat nichts geschrieben:\n" + log

    config, _ncc = check_model(init_path, None)

    assert config.layer_names == ["Gr", "S1", "W", "S2"], config.layer_names
    assert config.group_names == ["Gr", "WS2"], config.group_names

    z_values = [round(layer.z_nm, 4) for layer in config.layers]
    assert z_values == [0.0, 0.33, 0.4875, 0.645], z_values

    elements = [layer.element for layer in config.layers]
    assert elements == ["C", "S", "W", "S"], elements

    # Debye-Waller je Element aus der Tabelle
    assert abs(config.layers[2].debye_waller_u - 0.055) < 1e-9, config.layers[2].debye_waller_u
    assert abs(config.layers[1].debye_waller_u - 0.070) < 1e-9

    # eigene Suchbereiche und Masken je Gruppe
    assert abs(config.group("Gr").max_shift - 60.0) < 1e-6
    assert abs(config.group("WS2").max_shift - 80.0) < 1e-6
    assert config.group("Gr").isolate_when_active
    assert not config.group("WS2").isolate_when_active
    for group in config.groups:
        assert group.mask_file, group.name
        assert os.path.exists(os.path.join(out_dir, group.mask_file))

    with open(init_path) as handle:
        init = json.load(handle)
    rotations = dict((e["name"], e["rotation_deg"]) for e in init["lattices"])
    assert abs(rotations["WS2"] - 17.0) < 1e-6, rotations
    assert abs(rotations["Gr"] - TRUE_ROTATION) < 1.0, rotations

    print(f"  zwei Gitter : {config.layer_names} in {config.group_names}, "
          f"z = {z_values}, Drehung Gr {rotations['Gr']:+.2f} / WS2 {rotations['WS2']:+.1f} Grad")


if __name__ == "__main__":
    raise SystemExit(main())
