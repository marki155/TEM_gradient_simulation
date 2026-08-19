from dataclasses import is_dataclass, asdict

import tifffile as tifffile
import torch
import ase.build
import numpy as np
import os

import sys
from inspect import getsourcefile
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]

sys.path.insert(0, parent_dir)

import json
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from matplotlib import pyplot as plt
from layers import LayerConfig, LayerGroupConfig, TYPICAL_DEBYE_WALLER_U
from nn import NN_particles, ParticlesConfig, CONFIG_CLASS_REGISTRY
from testing import show_init_positions
from training_parameter import (
    TrainMethod,
    load_group_data,
    load_train_method,
    save_train_methods,
)
from gui_qt import launch_gui_qt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)

VOLTAGE_KV = 80


path = "../platelet_11"

length_of_10_unity_cells_WS2 =  612 #px
pixel_size_nm = 3.18 / length_of_10_unity_cells_WS2

DEBUG = False
SELECTOR = True
LOAD_CONFIGS = True

def serialize(obj):
    if is_dataclass(obj):
        return {
            "__type__": "dataclass",
            "__class__": type(obj).__name__,
            "data": {k: serialize(v) for k, v in asdict(obj).items()}
        }

    elif isinstance(obj, np.ndarray):
        return {"__type__": "ndarray", "data": obj.tolist(), "dtype": str(obj.dtype)}

    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [serialize(v) for v in obj]

    elif isinstance(obj, tuple):
        return {"__type__": "tuple", "data": [serialize(v) for v in obj]}

    else:
        return obj


def deserialize(obj, class_registry: dict):
    if isinstance(obj, dict):
        if obj.get("__type__") == "ndarray":
            return np.array(obj["data"], dtype=obj["dtype"])

        elif obj.get("__type__") == "tuple":
            return tuple(deserialize(v, class_registry) for v in obj["data"])

        elif obj.get("__type__") == "dataclass":
            cls = class_registry[obj["__class__"]]
            deserialized_fields = {k: deserialize(v, class_registry) for k, v in obj["data"].items()}
            return cls(**deserialized_fields)

        return {k: deserialize(v, class_registry) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [deserialize(v, class_registry) for v in obj]

    else:
        return obj


def init_config(image, config_path):
    config_save_path = os.path.join(config_path, "config.json")

    if os.path.exists(config_save_path) and LOAD_CONFIGS:
        with open(config_save_path, "r") as f:
            config_json = json.load(f)
            config = deserialize(config_json, CONFIG_CLASS_REGISTRY)

    else:
        H, W = image.shape

        max_rot_graphen = np.deg2rad(20)
        canvas_H_rot_graphen = np.cos(max_rot_graphen) * H + np.sin(max_rot_graphen) * W
        canvas_W_rot_graphen = np.cos(max_rot_graphen) * W + np.sin(max_rot_graphen) * H

        max_shift_graphen = 3 * 2.47 * 0.1 / pixel_size_nm
        canvas_H_shift_graphen = H + 5 * max_shift_graphen
        canvas_W_shift_graphen = W + 5 * max_shift_graphen

        canvas_size_graphen = (
            int(max(canvas_H_rot_graphen, canvas_H_shift_graphen)),
            int(max(canvas_W_rot_graphen, canvas_W_shift_graphen))
        )
        canvas_shift_graphen = (
            (canvas_size_graphen[0] - H) / 2,
            (canvas_size_graphen[1] - W) / 2
        )

        max_rot_ws2 = np.deg2rad(5)
        canvas_H_rot_ws2 = np.cos(max_rot_ws2) * H + np.sin(max_rot_ws2) * W
        canvas_W_rot_ws2 = np.cos(max_rot_ws2) * W + np.sin(max_rot_ws2) * H

        max_shift_ws2 = 3 * 3.18 * 0.1 / pixel_size_nm
        canvas_H_shift_ws2 = H + 2 * max_shift_ws2
        canvas_W_shift_ws2 = W + 2 * max_shift_ws2

        canvas_size_ws2 = (
            int(max(canvas_H_rot_ws2, canvas_H_shift_ws2)),
            int(max(canvas_W_rot_ws2, canvas_W_shift_ws2))
        )
        canvas_shift_WS2 = (
            (canvas_size_ws2[0] - H) / 2,
            (canvas_size_ws2[1] - W) / 2
        )

        single_WS2 = ase.build.mx2(formula='WS2')
        WS2_H_atoms = int((canvas_size_ws2[0] * pixel_size_nm * 10) / single_WS2.cell.lengths()[0]) * 2
        WS2_W_atoms = int((canvas_size_ws2[1] * pixel_size_nm * 10) / single_WS2.cell.lengths()[1]) * 2

        single_graphen = ase.build.graphene()
        graphen_H_atoms = int((canvas_size_graphen[0] * pixel_size_nm * 10) / single_graphen.cell.lengths()[0]) * 2
        graphen_W_atoms = int((canvas_size_graphen[1] * pixel_size_nm * 10) / single_graphen.cell.lengths()[1]) * 2

        ws2 = ase.build.mx2(formula='WS2', vacuum=1.0) * (WS2_H_atoms, WS2_W_atoms, 1)
        ws2.translate((-20, -200, 0))
        ws2.rotate(102-30+7, 'z')

        graphen = ase.build.graphene(vacuum=1.0) * (graphen_H_atoms, graphen_W_atoms, 1)
        # graphen.translate((H/2 * pixel_size_nm * 10, W/2 * pixel_size_nm * 10, 0))
        graphen.rotate(12, 'z')
        graphen.translate((H * 1.5 * pixel_size_nm * 10, 0, 0))
        # graphen.translate((-H / 2, -W / 2, 0))
        graphen.translate((-200, -200, 0))

        model_atoms = ws2

        slices = True
        if slices:
            init_positions = model_atoms.get_positions() / (pixel_size_nm * 10)

            lowS = np.min(init_positions[:, 2])
            highS = np.max(init_positions[:, 2])
            middleW = (lowS + highS) / 2
            slice_thickness = highS - middleW

            init_positions = init_positions - lowS + slice_thickness / 2

            slices_pos = []
            slices_intensities = []

            graphen_pos = graphen.get_positions() / (pixel_size_nm * 10)
            selector = (
                    (graphen_pos[:, 0] >= -canvas_shift_graphen[0]) &
                    (graphen_pos[:, 0] <= canvas_size_graphen[0] - canvas_shift_graphen[0]) &
                    (graphen_pos[:, 1] >= -canvas_shift_graphen[1]) &
                    (graphen_pos[:, 1] <= canvas_size_graphen[1] - canvas_shift_graphen[1])
            )
            if SELECTOR:
                slices_pos.append(graphen_pos[selector][:, :2])
                slices_intensities.append(graphen.get_atomic_numbers()[selector])
            else:
                slices_pos.append(graphen_pos[:, :2])
                slices_intensities.append(graphen.get_atomic_numbers())

            for i in range(3):
                selector = (
                        (init_positions[:, 2] > i * slice_thickness) &
                        (init_positions[:, 2] < (i + 1) * slice_thickness) &
                        (init_positions[:, 0] >= -canvas_shift_WS2[0]) &
                        (init_positions[:, 0] <= canvas_size_ws2[0] - canvas_shift_WS2[0]) &
                        (init_positions[:, 1] >= -canvas_shift_WS2[1]) &
                        (init_positions[:, 1] <= canvas_size_ws2[1] - canvas_shift_WS2[1])
                )
                slices_pos.append(init_positions[selector][:, :2])
                slices_intensities.append(model_atoms.get_atomic_numbers()[selector])

            init_positions = slices_pos
            init_intensities = slices_intensities
        else:
            init_positions = model_atoms.get_positions()[:, :2] / (pixel_size_nm * 10)
            init_positions = np.stack([init_positions], axis=-1)
            init_intensities = model_atoms.get_atomic_numbers()
            init_intensities = np.stack([init_intensities], axis=-1)

        if DEBUG:
            show_init_positions(H, W, init_positions, init_intensities, canvas_shift_graphen, canvas_size_graphen,
                                canvas_shift_WS2, canvas_size_ws2)


        amp_contrast = .1

        z_positions_nm = [
            0.00,  # Graphen
            0.33,  # S1
            0.488,  # W
            0.646,  # S2
        ]

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

        # --- Schichten und Struktur-Gruppen ---------------------------------
        # Eine Gruppe bewegt sich als starre Einheit (Shift/Rotation/Scale/Strain),
        # eine Schicht ist ein Multislice-Slice. Hier: Graphen + dreilagiges WS2.
        groups = [
            LayerGroupConfig(
                name="Gr",
                control_points=[10, 10],
                max_shift=max_shift_graphen,
                max_rotation=max_rot_graphen,
                init_scale=[1.12, 1.14], init_rotation=0.025, init_shift=[-9.5, -11.0],
                mask_file="gr_mask.tif",
                target_file="target_no_ws2.tif",
                isolate_when_active=True,   # aktiv -> nur Graphen wird simuliert
            ),
            LayerGroupConfig(
                name="WS2",
                control_points=[20, 20],
                max_shift=max_shift_ws2,
                max_rotation=max_rot_ws2,
                init_scale=[1.05, 1.09], init_rotation=0.0, init_shift=[15.9, -5.0],
                mask_file="ws2_mask.tif",
                isolate_when_active=False,
            ),
        ]

        # debye_waller_u: thermische Auslenkung (u_rms in A). Ohne sie ist das
        # projizierte Potential das eines ruhenden Atoms - die Spitze wird zu
        # hoch und haengt an der Pixelgroesse.
        layers = [
            LayerConfig(name="Gr", element="C", z_nm=z_positions_nm[0], group="Gr",
                        init_occupancy=100.0, occ_entropy=False,
                        debye_waller_u=TYPICAL_DEBYE_WALLER_U["C"]),
            LayerConfig(name="S1", element="S", z_nm=z_positions_nm[1], group="WS2",
                        debye_waller_u=TYPICAL_DEBYE_WALLER_U["S"]),
            LayerConfig(name="W", element="W", z_nm=z_positions_nm[2], group="WS2",
                        debye_waller_u=TYPICAL_DEBYE_WALLER_U["W"]),
            LayerConfig(name="S2", element="S", z_nm=z_positions_nm[3], group="WS2",
                        debye_waller_u=TYPICAL_DEBYE_WALLER_U["S"]),
        ]

        config = ParticlesConfig(init_positions=init_positions, init_intensities=init_intensities,
                                 shape=(H, W), pixel_size_nm=pixel_size_nm,
                                 voltage_kV=VOLTAGE_KV,
                                 init_aberrations=aberrations, init_ctf_env_params=env_params,
                                 init_amp_contrast=amp_contrast, ctf_alpha_max_mrad=50.,
                                 padding=100,
                                 layers=layers, groups=groups,
                                 )




        with open(config_save_path, "w") as f:
            json.dump(serialize(config), f)

    target_save_path = os.path.join(config_path, "target.npy")

    if os.path.exists(target_save_path):
        with open(target_save_path, "rb") as f:
            target = np.load(f)
            target = torch.tensor(target)
    else:
        target = torch.tensor(image, dtype=torch.float32)

        min_target = torch.min(target)
        max_target = torch.max(target)
        target = (target - min_target) / (max_target - min_target)

        if DEBUG:
            fig, ax = plt.subplots()
            ax.imshow(target.cpu(), cmap='gray')
            ax.set_title("Target Image")
            ax.scatter(init_positions[0][:, 1], init_positions[0][:, 0], s=0.5, c='red', label='Graphene')
            plt.show()

            fig, ax = plt.subplots()
            ax.imshow(target.cpu(), cmap='gray')
            ax.set_title("Target Image")
            labels = ["Gr", "S", "W", "S"]
            for i in range(1, 4):
                ax.scatter(init_positions[i][:, 1], init_positions[i][:, 0], s=0.5, label=labels[i])
            ax.legend()
            plt.show()

        np.save(target_save_path, target.cpu().numpy())

    return config, target



def init_train_method(save_path, config) -> TrainMethod:
    """TrainMethod laden (oder neu anlegen) und die Gruppendaten dazu holen.

    Maske und Zielbild je Gruppe stehen in der Config (``mask_file`` /
    ``target_file``) und werden hier aus dem Datensatz-Ordner geladen.
    """

    if os.path.exists(save_path) and LOAD_CONFIGS:
        train_method = load_train_method(save_path, config)
        load_group_data(train_method, config, base_dir=path)
        return train_method

    train_method = TrainMethod().ensure_for_config(config)
    load_group_data(train_method, config, base_dir=path)
    save_train_methods(train_method, save_path)
    return train_method


if __name__ == '__main__':
    image = tifffile.imread(os.path.join(path, "target_div-b.tif"))
    image[np.where(image == 0.0)] = np.median(image)

    config, target = init_config(image, os.path.join(path, "config"))
    train_method = init_train_method(os.path.join(path, "config", "train_method.json"), config)


    model = NN_particles(
        config=config,
        pixel_size_nm=pixel_size_nm,
        voltage_kV=VOLTAGE_KV,
        train_method=train_method,
        device=DEVICE,
        train_methods=train_method
    )

    model.save_particles_config(os.path.join(path, "particles_config.json"))

    model.build_multislice()

    ctf_layer = model.model[2]
    pot_layer = model.model[0]

    #pot_layer.occupy_layers_in_area(gr_mask, [1, 2, 3])

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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