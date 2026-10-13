"""Modell aus genau EINER Graphen-Schicht.

Gleicher Ablauf wie ``main.py``, nur dass die Config nur eine Schicht (und
damit nur eine Gruppe) enthaelt: kein WS2 im Multislice, kein WS2 in der GUI,
kein WS2 im Optimizer.

Wer weitere Schichten will, haengt einfach ``LayerConfig``/``LayerGroupConfig``
Eintraege an die Listen unten an - im uebrigen Code ist nichts fest verdrahtet.
"""

from dataclasses import is_dataclass, asdict

import tifffile as tifffile
import torch
import ase.build
import numpy as np
import os

import sys
from inspect import getsourcefile

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]

sys.path.insert(0, parent_dir)

import json

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from matplotlib import pyplot as plt

from layers import LayerConfig, LayerGroupConfig
from nn import NN_particles, ParticlesConfig, CONFIG_CLASS_REGISTRY
from training_parameter import (
    TrainMethod,
    load_group_data,
    load_train_method,
    save_train_methods,
)
from gui_qt import launch_gui_qt
from mains.main import serialize, deserialize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)

VOLTAGE_KV = 80

path = "../platelet_11"

#: Zielbild fuer das reine Graphen-Modell (Bild ohne WS2)
TARGET_FILE = "target_no_ws2.tif"
MASK_FILE = "gr_mask.tif"

#: eigener Config-Ordner, damit die 4-Schicht-Config unberuehrt bleibt
CONFIG_DIR = os.path.join(path, "config_graphene")

length_of_10_unity_cells_WS2 = 612  # px
pixel_size_nm = 3.18 / length_of_10_unity_cells_WS2

DEBUG = False
SELECTOR = True
LOAD_CONFIGS = True

GRAPHEN_ROTATION_DEG = 12
MAX_ROTATION_DEG = 20
#: zusaetzliche Verschiebung der Graphen-Lage in Pixeln (x, y)
GRAPHEN_OFFSET_PX = (0.0, 0.0)


def build_graphene_config(image):
    H, W = image.shape

    max_rot = np.deg2rad(MAX_ROTATION_DEG)
    canvas_H_rot = np.cos(max_rot) * H + np.sin(max_rot) * W
    canvas_W_rot = np.cos(max_rot) * W + np.sin(max_rot) * H

    max_shift = 3 * 2.47 * 0.1 / pixel_size_nm
    canvas_H_shift = H + 5 * max_shift
    canvas_W_shift = W + 5 * max_shift

    canvas_size = (int(max(canvas_H_rot, canvas_H_shift)),
                   int(max(canvas_W_rot, canvas_W_shift)))
    canvas_shift = ((canvas_size[0] - H) / 2, (canvas_size[1] - W) / 2)

    single_graphen = ase.build.graphene()
    graphen_H_atoms = int((canvas_size[0] * pixel_size_nm * 10) / single_graphen.cell.lengths()[0]) * 2
    graphen_W_atoms = int((canvas_size[1] * pixel_size_nm * 10) / single_graphen.cell.lengths()[1]) * 2

    graphen = ase.build.graphene(vacuum=1.0) * (graphen_H_atoms, graphen_W_atoms, 1)
    graphen.rotate(GRAPHEN_ROTATION_DEG, 'z')

    graphen_pos = graphen.get_positions() / (pixel_size_nm * 10)

    # Lage mittig ueber die Canvas legen (die Feinjustage macht spaeter der
    # trainierbare Shift der Gruppe).
    sheet_center = 0.5 * (graphen_pos[:, :2].min(axis=0) + graphen_pos[:, :2].max(axis=0))
    canvas_center = np.array([H / 2.0, W / 2.0])
    graphen_pos[:, :2] += canvas_center - sheet_center + np.asarray(GRAPHEN_OFFSET_PX)

    selector = (
            (graphen_pos[:, 0] >= -canvas_shift[0]) &
            (graphen_pos[:, 0] <= canvas_size[0] - canvas_shift[0]) &
            (graphen_pos[:, 1] >= -canvas_shift[1]) &
            (graphen_pos[:, 1] <= canvas_size[1] - canvas_shift[1])
    )

    if SELECTOR:
        init_positions = [graphen_pos[selector][:, :2]]
        init_intensities = [graphen.get_atomic_numbers()[selector]]
    else:
        init_positions = [graphen_pos[:, :2]]
        init_intensities = [graphen.get_atomic_numbers()]

    aberrations = {"C10": -13.,
                   'C12': 1.0, "phi12": 0.,
                   'C23': 2., 'phi23': 0.,
                   'C21': 2., 'phi21': 0.,
                   "C30": 4.,  # um
                   'C34': .02, 'phi34': 0.,  # um
                   'C45': .05, 'phi45': 0.}  # um

    env_params = {
        'source_size': 0.1e-3,  # rad
        'Cc_um': 1.e3,  # um
        'deltaE': .35,  # eV
        'instab_defocus_nm': .01,  # nm
        'use_detector': False,
    }

    groups = [
        LayerGroupConfig(
            name="Gr",
            control_points=[10, 10],
            max_shift=max_shift,
            max_rotation=max_rot,
            init_scale=[1.12, 1.14], init_rotation=0.025, init_shift=[-9.5, -11.0],
            mask_file=MASK_FILE,
            target_file=TARGET_FILE,
            isolate_when_active=True,
        ),
    ]

    layers = [
        LayerConfig(name="Gr", element="C", z_nm=0.0, group="Gr",
                    init_occupancy=100.0, occ_entropy=False),
    ]

    return ParticlesConfig(
        init_positions=init_positions, init_intensities=init_intensities,
        shape=(H, W), pixel_size_nm=pixel_size_nm, voltage_kV=VOLTAGE_KV,
        init_aberrations=aberrations, init_ctf_env_params=env_params,
        init_amp_contrast=0.1, ctf_alpha_max_mrad=50.,
        padding=100, layers=layers, groups=groups,
    )


def init_config(image, config_path):
    os.makedirs(config_path, exist_ok=True)
    config_save_path = os.path.join(config_path, "config.json")

    if os.path.exists(config_save_path) and LOAD_CONFIGS:
        with open(config_save_path, "r") as f:
            config = deserialize(json.load(f), CONFIG_CLASS_REGISTRY)
    else:
        config = build_graphene_config(image)
        with open(config_save_path, "w") as f:
            json.dump(serialize(config), f)

    target_save_path = os.path.join(config_path, "target.npy")

    if os.path.exists(target_save_path):
        target = torch.tensor(np.load(target_save_path))
    else:
        target = torch.tensor(image, dtype=torch.float32)
        target = (target - torch.min(target)) / (torch.max(target) - torch.min(target))
        np.save(target_save_path, target.cpu().numpy())

    return config, target


def init_train_method(save_path, config) -> TrainMethod:
    if os.path.exists(save_path) and LOAD_CONFIGS:
        train_method = load_train_method(save_path, config)
    else:
        train_method = TrainMethod().ensure_for_config(config)
        # Startpunkt: nur die Graphen-Geometrie trainieren
        train_method.set_active_group("Gr")
        train_method.group("Gr").shift.enabled = True
        train_method.group("Gr").rotation.enabled = True
        train_method.group("Gr").scale.enabled = True
        train_method.train_ctf_aberrations.enabled = False

    load_group_data(train_method, config, base_dir=path)
    save_train_methods(train_method, save_path)
    return train_method


if __name__ == '__main__':
    image = tifffile.imread(os.path.join(path, TARGET_FILE))
    image[np.where(image == 0.0)] = np.median(image)

    config, target = init_config(image, CONFIG_DIR)
    train_method = init_train_method(os.path.join(CONFIG_DIR, "train_method.json"), config)

    model = NN_particles(
        config=config,
        pixel_size_nm=pixel_size_nm,
        voltage_kV=VOLTAGE_KV,
        train_method=train_method,
        device=DEVICE,
        train_methods=train_method
    )

    model.save_particles_config(os.path.join(CONFIG_DIR, "particles_config.json"))
    model.build_multislice()

    pot_layer = model.model[0]
    ctf_layer = model.model[2]

    print("Schichten:", config.layer_names, "Gruppen:", config.group_names)

    launch_gui_qt(
        model=model,
        target=target,
        pot_layer=pot_layer,
        ctf_layer=ctf_layer,
        train_method=train_method,
        plot_every=25,
        heavy_every=50,
        max_steps=2_000_000,
    )
