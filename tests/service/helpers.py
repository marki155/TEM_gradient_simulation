"""Small data folders and runs for the service tests."""

import json
import os

import numpy as np
import torch

from temsim.data import store
from temsim.data.schema import TrainSettings
from temsim.learning import Model

from ..learning.helpers import SHAPE, PX, make_experiment

PAD = 12


def smooth_target(shape=SHAPE, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=shape)
    for axis in (0, 1):
        a = (np.roll(a, 1, axis) + a + np.roll(a, -1, axis)) / 3.0
    return (a - a.min() + 0.5).astype(np.float32)


def graphene_init(mode="tem", with_lattice=True, **extra):
    init = {"version": 2, "pixel_size_nm": PX, "voltage_kV": 80.0, "shape": list(SHAPE), "padding": PAD,
            "config_dir": "config_test", "target_file": "target.npy",
            "groups": [{"name": "Gr", "control_points": [4, 4], "max_shift_px": 6.0, "max_rotation_deg": 10.0}],
            "layers": [{"name": "Gr", "element": "C", "group": "Gr", "init_occupancy": 3.0, "z_nm": 0.0}],
            "imaging": {"mode": mode, "aberrations": {"C10": -8.0, "C30": 1.0}},
            "train": {"enable": ["groups.Gr.shift"], "learning_rates": {"groups.Gr.shift": 0.05},
                      "losses": {"loss_ncc": 0.5}, "active_group": "Gr"}}
    if with_lattice:
        init["lattices"] = [{"name": "Gr", "type": "custom", "a": 2.46, "gamma_deg": 120.0, "rotation_deg": 0.0,
                             "basis": [["C", 0.0, 0.0, 0.0], ["C", 1 / 3, 2 / 3, 0.0]]}]
    else:
        init.pop("layers"), init.pop("groups")
    init.update(extra)
    return init


def make_folder(path, mode="tem", with_lattice=True, mask=True, target=None, **extra):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "target.npy"), smooth_target() if target is None else target)
    if mask:
        m = np.zeros(SHAPE, dtype=np.uint8)
        m[8:-8, 8:-8] = 1
        np.save(os.path.join(path, "mask_Gr.npy"), m)
    with open(os.path.join(path, "init.json"), "w") as f:
        json.dump(graphene_init(mode, with_lattice, **extra), f)
    return str(path)


def make_run(path, mode="tem", n=(30, 20)):
    """A run directory whose target is the image of its own structure (plus a little noise)."""

    exp = make_experiment(mode=mode, n=n)
    with torch.no_grad():
        image = Model(exp, target=torch.zeros(SHAPE)).image().numpy()
    target = (image + 0.01 * np.random.default_rng(1).normal(size=image.shape)).astype(np.float32)
    settings = TrainSettings().ensure_for(exp)
    settings.active_group = "g1"
    settings.group("g1").shift.enabled = True
    settings.group("g1").shift.lr = 0.05
    mask = np.zeros(SHAPE, dtype=bool)
    mask[6:-6, 6:-6] = True
    settings.masks = {"g1": mask}
    store.save_run(str(path), store.Run(exp, settings, None, target))
    return str(path), exp, target
