"""Simulation aus einer ImageJ-Initialisierung starten.

Das Fiji-Plugin (plugins/TEM_Sim_Init.py) schreibt neben das Bild eine
``init.json`` plus Masken und Zielbild. Dieses Skript liest das und baut
daraus Config, TrainMethod und Modell - danach geht die uebliche Trainings-GUI
auf.

    python3 mains/imagej_main.py /pfad/zu/init.json
    python3 mains/imagej_main.py /pfad/zu/init.json --no-gui   # nur aufbauen

Das Format ist bewusst schlicht gehalten, damit man es auch von Hand
schreiben oder aus einem anderen Werkzeug erzeugen kann.
"""

from __future__ import annotations

import json
import os
import sys
from inspect import getsourcefile

current_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
sys.path.insert(0, current_dir[:current_dir.rfind(os.path.sep)])

import numpy as np
import tifffile
import torch

from lattice import LatticeSpec, build_layer_positions
from layers import LayerConfig, LayerGroupConfig, TYPICAL_DEBYE_WALLER_U
from nn import NN_particles, ParticlesConfig
from training_parameter import TrainMethod, save_train_methods

INIT_VERSION = 2


# ---------------------------------------------------------------------------
# Lesen
# ---------------------------------------------------------------------------

def load_init(path: str):
    with open(path, "r") as f:
        init = json.load(f)

    version = init.get("version", 1)
    if version > INIT_VERSION:
        print(f"[imagej] Warnung: init.json ist Version {version}, dieses Skript kennt "
              f"nur {INIT_VERSION} - unbekannte Felder werden ignoriert.")
    return init


def _read_image(path: str):
    if path.endswith(".npy"):
        return np.load(path)
    return tifffile.imread(path)


def _prepare_target(image, zero_handling="median"):
    image = np.asarray(image, dtype=np.float32)
    if zero_handling == "median":
        zeros = image == 0.0
        if zeros.any():
            image = image.copy()
            image[zeros] = np.median(image)
    return image


# ---------------------------------------------------------------------------
# Config bauen
# ---------------------------------------------------------------------------

def lattice_spec_from(entry) -> LatticeSpec:
    """Ein Eintrag aus der init.json -> LatticeSpec."""

    entry = dict(entry or {})
    return LatticeSpec(
        type=entry.get("type", "graphene"),
        a=float(entry.get("a", 2.46)),
        b=entry.get("b"),
        gamma_deg=float(entry.get("gamma_deg", 120.0)),
        formula=entry.get("formula", "WS2"),
        file=entry.get("file"),
        basis=[tuple(b) for b in entry.get("basis", [])],
        rotation_deg=float(entry.get("rotation_deg", 0.0)),
        offset_px=tuple(entry.get("offset_px", (0.0, 0.0))),
    )


def lattice_entries(init):
    """Liste der Gitter - eine ``lattice``-Angabe wird als einzelnes gefuehrt."""

    entries = init.get("lattices")
    if entries:
        return [dict(e) for e in entries]

    single = dict(init.get("lattice", {}))
    single.setdefault("name", init["groups"][0]["name"] if init.get("groups") else "Gitter")
    return [single]


def build_config(init, base_dir: str) -> ParticlesConfig:
    """ParticlesConfig aus der Initialisierung, inklusive Startpositionen."""

    shape = tuple(init["shape"])
    pixel_size_nm = float(init["pixel_size_nm"])
    imaging = init.get("imaging", {})

    groups_raw = init["groups"]
    layers_raw = init["layers"]

    group_by_name = {g["name"]: g for g in groups_raw}
    entries = lattice_entries(init)
    default_lattice = entries[0].get("name", "Gitter")

    # Suchbereich je Gitter aus den Gruppen, die es benutzen
    reach = {}
    for layer in layers_raw:
        name = layer.get("lattice", default_lattice)
        group = group_by_name.get(layer.get("group"), {})
        shift = float(group.get("max_shift_px", 0.0) or 0.0)
        rotation = float(group.get("max_rotation_deg", 0.0) or 0.0)
        previous = reach.get(name, (0.0, 0.0))
        reach[name] = (max(previous[0], shift), max(previous[1], rotation))

    built = {}
    for entry in entries:
        name = entry.get("name", default_lattice)
        shift, rotation = reach.get(name, (0.0, 0.0))
        built[name] = {
            "layers": build_layer_positions(lattice_spec_from(entry), shape, pixel_size_nm,
                                            max_shift_px=shift, max_rotation_deg=rotation),
            "z_offset_nm": float(entry.get("z_offset_nm", 0.0)),
        }

    counters = {}
    generated_for = []
    for entry in layers_raw:
        name = entry.get("lattice", default_lattice)
        if name not in built:
            raise ValueError(f"Schicht {entry['name']!r} verweist auf das unbekannte "
                             f"Gitter {name!r}. Bekannt: {sorted(built)}")
        index = entry.get("lattice_layer")
        if index is None:
            index = counters.get(name, 0)
        counters[name] = index + 1

        available = built[name]["layers"]
        if index >= len(available):
            raise ValueError(f"Gitter {name!r} hat {len(available)} Schichten, "
                             f"Schicht {entry['name']!r} will Nummer {index}.")
        generated = dict(available[index])
        generated["z_nm"] = generated["z_nm"] + built[name]["z_offset_nm"]
        generated_for.append(generated)

    init_positions, init_intensities, layers = [], [], []
    for entry, generated in zip(layers_raw, generated_for):
        element = entry.get("element") or generated["elements"][0]
        z_nm = float(entry.get("z_nm", generated["z_nm"]))

        init_positions.append(np.asarray(generated["positions"], dtype=np.float64))
        atomic = _atomic_numbers(generated["elements"], element)
        init_intensities.append(atomic)

        layers.append(LayerConfig(
            name=entry["name"],
            element=element,
            z_nm=z_nm,
            group=entry.get("group") or entry["name"],
            init_occupancy=float(entry.get("init_occupancy", 2.0)),
            occ_entropy=bool(entry.get("occ_entropy", True)),
            elements=list(entry.get("elements", [])),
            debye_waller_u=float(entry.get("debye_waller_u",
                                           TYPICAL_DEBYE_WALLER_U.get(element, 0.0))),
        ))

    groups = []
    for entry in groups_raw:
        groups.append(LayerGroupConfig(
            name=entry["name"],
            control_points=list(entry.get("control_points", [10, 10])),
            max_shift=float(entry["max_shift_px"]) if entry.get("max_shift_px") is not None else None,
            max_rotation=np.deg2rad(float(entry["max_rotation_deg"]))
            if entry.get("max_rotation_deg") is not None else None,
            init_shift=list(entry.get("init_shift_px", [0.0, 0.0])),
            init_rotation=float(np.deg2rad(float(entry.get("init_rotation_deg", 0.0)))),
            init_scale=list(entry.get("init_scale", [1.0, 1.0])),
            init_contrast_factor=float(entry.get("init_contrast_factor", 1.0)),
            mask_file=entry.get("mask_file"),
            target_file=entry.get("target_file"),
            isolate_when_active=bool(entry.get("isolate_when_active", False)),
        ))

    stem = imaging.get("stem", {})
    config = ParticlesConfig(
        init_positions=init_positions,
        init_intensities=init_intensities,
        shape=shape,
        pixel_size_nm=pixel_size_nm,
        voltage_kV=float(init.get("voltage_kV", 80.0)),
        init_aberrations=dict(imaging.get("aberrations", {})),
        init_ctf_env_params=dict(imaging.get("envelope", {})),
        init_amp_contrast=float(imaging.get("amp_contrast", 0.1)),
        ctf_alpha_max_mrad=float(imaging.get("alpha_max_mrad", 50.0)),
        padding=int(init.get("padding", 100)),
        layers=layers,
        groups=groups,
        beam_direction=imaging.get("beam_direction", "+z"),
        ctf_parametrization=imaging.get("ctf_parametrization", "cartesian"),
        imaging_mode=imaging.get("mode", "tem"),
        stem_convergence_mrad=float(stem.get("convergence_mrad", 25.0)),
        stem_detector_mrad=tuple(stem.get("detector_mrad", (60.0, 200.0))),
        stem_source_size_A=float(stem.get("source_size_A", 0.5)),
        init_probe_aberrations=dict(stem["aberrations"]) if stem.get("aberrations") else None,
    )
    return config


def _atomic_numbers(symbols, fallback):
    from ase.data import atomic_numbers
    return np.array([float(atomic_numbers[s]) for s in symbols], dtype=np.float64) \
        if symbols else np.full(0, float(atomic_numbers[fallback]))


# ---------------------------------------------------------------------------
# TrainMethod bauen
# ---------------------------------------------------------------------------

def build_train_method(init, config, base_dir: str) -> TrainMethod:
    train = init.get("train", {})
    tm = TrainMethod().ensure_for_config(config)

    for key, value in train.get("losses", {}).items():
        if hasattr(tm, key):
            setattr(tm, key, float(value))

    for key in train.get("enable", []):
        param = tm.get_train_param(key)
        if param is None:
            print(f"[imagej] unbekannter Trainingsschalter: {key}")
            continue
        param.enabled = True

    for key, lr in train.get("learning_rates", {}).items():
        param = tm.get_train_param(key)
        if param is not None:
            param.lr = float(lr)

    tm.set_active_group(train.get("active_group"))

    # Masken und Gruppen-Targets aus den Dateien der Config
    for group in config.groups:
        if group.mask_file:
            mask = _read_image(os.path.join(base_dir, group.mask_file))
            tm.masks[group.name] = torch.tensor(np.asarray(mask) > 0)
        if group.target_file:
            image = _prepare_target(_read_image(os.path.join(base_dir, group.target_file)))
            tm.targets[group.name] = torch.tensor(image, dtype=torch.float32)

    return tm


def load_target(init, base_dir: str):
    image = _prepare_target(_read_image(os.path.join(base_dir, init["target_file"])),
                            init.get("target_zero_handling", "median"))
    target = torch.tensor(image, dtype=torch.float32)
    minimum, maximum = torch.min(target), torch.max(target)
    return (target - minimum) / (maximum - minimum)


# ---------------------------------------------------------------------------
# Zusammensetzen
# ---------------------------------------------------------------------------

def build_everything(init_path: str):
    init = load_init(init_path)
    base_dir = os.path.dirname(os.path.abspath(init_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    config = build_config(init, base_dir)
    train_method = build_train_method(init, config, base_dir)
    target = load_target(init, base_dir)

    model = NN_particles(config=config, pixel_size_nm=config.pixel_size_nm,
                         voltage_kV=config.voltage_kV, train_method=train_method,
                         device=device, train_methods=train_method)
    model.build()

    print(f"[imagej] {config.imaging_mode}-Modell auf {device}: "
          f"Schichten {config.layer_names}, Gruppen {config.group_names}, "
          f"{sum(len(p) for p in config.init_positions)} Atome, Bild {config.shape}")

    return init, base_dir, config, train_method, target, model


def save_state(init, base_dir, config, train_method):
    """Config und TrainMethod neben der init.json ablegen, damit der Lauf reproduzierbar ist."""

    out_dir = os.path.join(base_dir, init.get("config_dir", "config_imagej"))
    os.makedirs(out_dir, exist_ok=True)

    from mains.main import serialize
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(serialize(config), f)
    save_train_methods(train_method, os.path.join(out_dir, "train_method.json"))
    return out_dir


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 1

    init_path = argv[0]
    show_gui = "--no-gui" not in argv

    init, base_dir, config, train_method, target, model = build_everything(init_path)
    out_dir = save_state(init, base_dir, config, train_method)
    print(f"[imagej] Config und TrainMethod gespeichert in {out_dir}")

    if not show_gui:
        simulated, _tgt, _p, loss = model.loss_multislice(target)
        print(f"[imagej] Startloss {float(loss):.5f}, simuliertes Bild {tuple(simulated.shape)}")
        return 0

    from gui_qt import launch_gui_qt
    launch_gui_qt(model=model, target=target,
                  pot_layer=model.potential_layer, ctf_layer=model.imaging_layer,
                  train_method=train_method, plot_every=25, heavy_every=50,
                  max_steps=2_000_000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
