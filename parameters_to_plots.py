import json
import os

import matplotlib.pyplot as plt
import numpy as np
import sys
from PySide6.QtWidgets import QApplication, QFileDialog
from scipy.spatial import ConvexHull
import tifffile

from mains.main import deserialize
from nn import ParticlesConfig, ParticlesTunableParameters
from parameter_frame import serialize
from potential import DifferentiablePotential
from training_parameter import TrainMethod

from ase import Atoms

from shapely.geometry import Polygon, MultiPoint, Point


def load_json(path=None):

    if path is None:
        app = QApplication(sys.argv)
        file_path, _ = QFileDialog.getOpenFileName(
                    None,
                    "Parameter laden",
                    "parameters",
                    "JSON-Dateien (*.json)"
                )

        if not file_path:
            return
        file_path = path

    else:
        file_path = path


    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def get_pos_and_ints(target_img, pot_positions, pot_intensities, train_method):

    idx = train_method.only_train_small_area



    _idx = idx if idx >= 0 else 1

    x_min = 0
    y_min = 0
    x_max = target_img.shape[0]
    y_max = target_img.shape[1]


    if _idx == 0:
        pos = pot_positions[_idx]
        ints = pot_intensities[_idx]
    elif _idx == 1:
        poses = []
        intss = []
        for i in range(1,4):
            pos = pot_positions[i]
            ints = pot_intensities[i]
            poses.append(pos)
            intss.append(ints)
        pos = np.concatenate(poses, axis=0)
        ints = np.concatenate(intss, axis=0)

    mask_borders = ((pos[:, 0] >= x_min) & (pos[:, 0] <= x_max) &
            (pos[:, 1] >= y_min) & (pos[:, 1] <= y_max))
    positions = pos[mask_borders]
    intensities = ints[mask_borders]

    #mask_img = self.train_method.area_masks[_idx].cpu().numpy()
    mask_img = None
    if mask_img is not None:
        mask = mask_img[positions[:, 0].astype(int), positions[:, 1].astype(int)]

        positions = positions[mask]
        intensities = intensities[mask]

    return (positions[:,0] - x_min, positions[:,1]-y_min), intensities

if __name__ == "__main__":
    path = "platelet_11/"
    config_file = path + "config/config.json"
    param_file = path + "best_long_free_ws2_no_occ4.json"
    params_json = load_json(param_file)


    with open(config_file, "r") as f:
        config_json = json.load(f)
        config = deserialize(config_json, {"ParticlesConfig": ParticlesConfig})


    params = ParticlesTunableParameters(**params_json)

    pixel_size_nm = config.pixel_size_nm

    train_method = TrainMethod(area_masks=[None, None], target_gr=None)
    pot_layer = DifferentiablePotential.from_config_and_parameters(config, train_method, params.potential_parameters)

    pot_imgs = pot_layer.get_potential_images()

    pot_imgs.append(pot_imgs[1] + pot_imgs[3])

    import matplotlib.colors as mcolors

    base = plt.cm.seismic

    # nur obere Hälfte (weiß -> rot)
    colors_upper = base(np.linspace(0.5, 1.0, 256))
    colors_lower = base(np.linspace(0.0, 0.5, 256))

    red = mcolors.LinearSegmentedColormap.from_list(
        "seismic_upper",
        colors_upper
    )

    blue = mcolors.LinearSegmentedColormap.from_list(
        "seismic_lower",
        colors_lower[::-1])

    names = ["Gr", "S1", "W", "S2", "all_S", "WS2"]
    colors = ["Greys", red, blue, red, red]
    for i, pot_img in enumerate(pot_imgs):
        plt.figure()
        pot_img = (pot_img - np.min(pot_img)) / (np.max(pot_img) - np.min(pot_img))
        plt.imshow(pot_img,cmap=colors[i], origin="lower")
        plt.title("Potential Image " + names[i])
        #plt.colorbar(label="Potential Value")
        plt.xticks([])
        plt.yticks([])
        save_img_file = param_file.replace(".json", "_" + names[i] + ".png")
        plt.savefig(save_img_file, dpi=300)


    map_S = pot_imgs[1] + pot_imgs[3]
    map_S = (map_S - np.min(map_S)) / (np.max(map_S) -np.min(map_S))
    map_W = pot_imgs[2]
    map_W = (map_W - np.min(map_W)) / (np.max(map_W) -np.min(map_W))


    plt.figure()

    comb_map = map

    plt.imshow(np.abs(map_S) - np.abs(map_W), cmap="seismic", origin="lower")


    plt.title("Potential Image " + "WS2")
    #plt.colorbar(label="Potential Value")
    #plt.xlabel("Pixel")
    #plt.ylabel("Pixel")
    save_img_file = param_file.replace(".json", "_" + "WS2" + ".png")
    plt.savefig(save_img_file, dpi=300)

    pot_positions = []
    pot_intensities = []
    for i in range(4):
        (px, py), ints = pot_layer.get_pos_and_ints(i)
        pos = np.column_stack([px, py]).astype(np.float32)
        ints = np.asarray(ints, dtype=np.float32)
        pot_positions.append(pos)
        pot_intensities.append(ints)

    target_img = tifffile.imread(os.path.join(path, "target_div-b.tif"))
    pos, ints = get_pos_and_ints(target_img, pot_positions, pot_intensities, train_method)

    plt.imshow(target_img, cmap="grey", origin="lower")
    plt.scatter(pos[1], pos[0], c=ints, cmap="Wistia", s=0.5)
    plt.xticks([])
    plt.yticks([])
    plt.title("")
    plt.tight_layout()
    save_img_file = param_file.replace(".json", "_" + "target-scatter" + ".png")
    plt.savefig(save_img_file, dpi=300)






    for i in range(4):
        poses, shifts = pot_layer.occupied_free_shifts()
        pos = poses[i]
        shift = shifts[i]
        shift_len = np.linalg.norm(shift, axis=1)

        plt.figure(figsize=(6, 6))
        dx = shift[:, 0]
        dy = shift[:, 1]
        x = pos[:, 0] - shift[:, 0]
        y = pos[:, 1] - shift[:, 1]
        # Vektorfeld
        plt.quiver(x, y, dx, dy)

        # Optional: Punkte anzeigen
        plt.scatter(x, y, color='red')

        plt.axis('equal')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Vektormap / Shift-Field")
        save_img_file = param_file.replace(".json", "_" + "vector_map_"+names[i] + ".png")
        plt.savefig(save_img_file, dpi=300)


    for i in range(1,4):
        plt.figure()
        pad = pot_layer.config.padding
        positions = pot_layer.get_positions()[i].cpu().detach().cpu().numpy()
        limit_mask = ((positions[:, 0] >= pad) & (positions[:, 0] < pot_layer.H + pad) &
                      (positions[:, 1] >= pad) & (positions[:, 1] < pot_layer.W + pad))
        occ = pot_layer.occupancies[i].sigmoid().detach().cpu().numpy()[limit_mask]
        plt.hist(occ, bins= 100)
        plt.title("Occupancy Histogram " + names[i])
        plt.xlabel("Occupancy Value")
        plt.ylabel("Frequency")
        plt.yticks([])
        save_img_file = param_file.replace(".json", "_" + "occ_hist_" + names[i] + ".png")
        plt.savefig(save_img_file, dpi=300)

        plt.figure()
        plt.scatter(positions[limit_mask][:, 1], positions[limit_mask][:, 0], c=occ, cmap=blue, s=0.5)
        plt.colorbar(label="Occupancy Value")
        plt.title("Occupancy Scatter " + names[i])
        plt.xticks([])
        plt.yticks([])
        save_img_file = param_file.replace(".json", "_" + "occ_scatter_" + names[i] + ".png")
        plt.savefig(save_img_file, dpi=300)

