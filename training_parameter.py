from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, asdict, fields
from typing import List, Optional

import numpy as np
import torch
import json
import os


@dataclass
class TrainState:
    running: bool = False
    paused: bool = False
    step: int = 0
    loss: float = 0.0

@dataclass
class TrainMethodOLD:

    loss_mse: float = 0.0
    loss_ncc: float = .5
    loss_local_ncc_100: float = 0.
    loss_local_ncc_50: float = 0.
    loss_local_ncc_20: float = 0.
    loss_ssim: float = 0.0
    loss_l1: float = 0.1
    loss_fft: float = 0.0



    train_amplitude_contrast: bool = False

    learning_rate: float = 0.00001

    train_graphen_intensities: bool = False
    train_S1_intensities: bool = False
    train_W_intensities: bool = False
    train_S2_intensities: bool = False

    train_graphen_occupancies: bool = False
    train_S1_occupancies: bool = False
    train_W_occupancies: bool = False
    train_S2_occupancies: bool = False

    WS2_strain_field: bool = False

    graphen_shift: bool = True
    graphen_rotation: bool = True
    graphen_free_positions: bool = False
    graphen_scale: bool = True

    WS2_shift: bool = False
    WS2_rotation: bool = False
    WS2_free_positions: bool = False
    WS2_scale: bool = False


    train_ctf_aberrations: bool = True
    train_aberrations = {"C10": True,
                   'C12': True,
                   'C23': True,
                   'C21': True,
                   "C30": True,
                   'C34': True,
                   'C45': True}  # um
    train_env_parameters: bool = False
    train_env_params = {
        'source_size': False,
        'Cc': True,
        'deltaE': True,
        'instab_defocus_nm': False
    }

    train_normalisation_parameters: bool = True

    only_train_small_area: int = 1 # TODO



    #train_small_area_squares = [(50, 50, 350, 350), (500, 600, 1000, 1000)]
    train_small_area_squares = [(50, 50, 350, 350), (600, 600, 1000, 1000)]

@dataclass
class TrainParam:
    enabled: bool = False
    lr: float = 1e-4


def serialize(obj):
    if isinstance(obj, torch.Tensor):
        return {"__type__": "tensor", "data": obj.detach().cpu().tolist()}

    elif is_dataclass(obj):
        d = {
            "__type__": "dataclass",
            "__class__": type(obj).__name__,
            "data": {}
        }
        # fields() gibt nur dataclass-Felder zurück, obj bleibt erhalten
        for f in fields(obj):
            d["data"][f.name] = serialize(getattr(obj, f.name))
        # Klassenattribute manuell
        for k in ["train_aberrations", "train_env_params"]:
            if hasattr(obj, k):
                d["data"][k] = serialize(getattr(obj, k))
        return d

    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [serialize(v) for v in obj]

    elif isinstance(obj, tuple):
        return {"__type__": "tuple", "data": [serialize(v) for v in obj]}

    else:
        return obj
        return obj

def deserialize(obj, class_registry: dict):
    if isinstance(obj, dict):
        t = obj.get("__type__")

        if t == "tensor":
            return torch.tensor(obj["data"])

        elif t == "tuple":
            return tuple(deserialize(v, class_registry) for v in obj["data"])

        elif t == "dataclass":
            cls = class_registry[obj["__class__"]]
            # Klassenattribute rausfiltern, die kein __init__-Parameter sind
            init_fields = {f.name for f in fields(cls)}
            kwargs = {}
            class_attrs = {}
            for k, v in obj["data"].items():
                if k in init_fields:
                    kwargs[k] = deserialize(v, class_registry)
                else:
                    class_attrs[k] = deserialize(v, class_registry)
            instance = cls(**kwargs)
            # Klassenattribute setzen
            for k, v in class_attrs.items():
                setattr(instance, k, v)
            return instance

        return {k: deserialize(v, class_registry) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [deserialize(v, class_registry) for v in obj]

    else:
        return obj




@dataclass
class TrainMethod:
    area_masks: List
    target_gr: Optional[torch.Tensor] | None

    loss_var: float = 0.5
    loss_local_var_100: float = 0.0
    loss_mse: float = 0.0
    loss_ncc: float = 0.5
    loss_local_ncc_100: float = 0.0
    loss_ssim: float = 0.0
    loss_l1: float = 0.1

    occ_entropy_a: float = 1.0
    position_shifts_regularization_a: float = 1.0
    strain_energy_a: float = 0.0

    train_amplitude_contrast: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-5))

    train_graphen_intensities: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))
    train_S1_intensities: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))
    train_W_intensities: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))
    train_S2_intensities: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))

    train_graphen_occupancies: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))
    train_S1_occupancies: TrainParam = field(default_factory=lambda: TrainParam(False, .1))
    train_W_occupancies: TrainParam = field(default_factory=lambda: TrainParam(False, .1))
    train_S2_occupancies: TrainParam = field(default_factory=lambda: TrainParam(False, .1))

    Gr_strain_field: TrainParam = field(default_factory=lambda: TrainParam(False, .1))
    WS2_strain_field: TrainParam = field(default_factory=lambda: TrainParam(False, .1))

    graphen_shift: TrainParam = field(default_factory=lambda: TrainParam(True, 1e-3))
    graphen_rotation: TrainParam = field(default_factory=lambda: TrainParam(True, 1e-5))
    graphen_free_positions: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))
    graphen_scale: TrainParam = field(default_factory=lambda: TrainParam(True, 1e-4))

    WS2_shift: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-3))
    WS2_rotation: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-5))
    WS2_free_positions: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-3))
    WS2_scale: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))

    train_ctf_aberrations: TrainParam = field(default_factory=lambda: TrainParam(True, 1e-5))
    train_aberrations = {"C10": True,
                         'C12': True,
                         'C23': True,
                         'C21': True,
                         "C30": True,
                         'C34': True,
                         'C45': True}  # um
    train_env_parameters: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-5))
    train_env_params = {
        'source_size': False,
        'Cc': True,
        'deltaE': True,
        'instab_defocus_nm': False
    }
    train_normalisation_parameters: TrainParam = field(default_factory=lambda: TrainParam(True, 1e-5))

    only_train_small_area: int = 1
    #train_small_area_squares = [(50, 50, 350, 350), (600, 600, 1000, 1000)]


registry = {
    "TrainMethod": TrainMethod,
    "TrainParam": TrainParam,
}


def save_train_methods(tm, save_path):
    data = serialize(tm)
    data["data"].pop("area_masks", None)
    data["data"].pop("target_gr", None)
    with open(save_path, "w") as f:
        json.dump(data, f)

    # Masken und target_gr als .npy neben der JSON speichern
    config_dir = os.path.dirname(save_path)
    np.save(os.path.join(config_dir, "area_masks.npy"),
            np.stack([m.cpu().numpy() for m in tm.area_masks]))
    np.save(os.path.join(config_dir, "target_gr.npy"),
            tm.target_gr.cpu().numpy())


def load_train_method(save_path) -> TrainMethod:
    with open(save_path, "r") as f:
        raw = json.load(f)
    raw["data"]["area_masks"] = []
    raw["data"]["target_gr"] = None
    tm = deserialize(raw, registry)

    config_dir = os.path.dirname(save_path)
    masks = np.load(os.path.join(config_dir, "area_masks.npy"))
    tm.area_masks = [torch.tensor(masks[i]) for i in range(masks.shape[0])]
    tm.target_gr = torch.tensor(
        np.load(os.path.join(config_dir, "target_gr.npy")),
        dtype=torch.float32
    )
    return tm