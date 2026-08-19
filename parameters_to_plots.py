import json
import os

import matplotlib.pyplot as plt
import numpy as np
import sys
from PySide6.QtWidgets import QApplication, QFileDialog
from scipy.spatial import ConvexHull
import tifffile

from mains.main import deserialize
from nn import ParticlesConfig, ParticlesTunableParameters, CONFIG_CLASS_REGISTRY
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

def get_pos_and_ints(target_img, pot_positions, pot_intensities, config, group=None):
    """Positionen/Intensitaeten einer Gruppe (oder aller Schichten)."""

    if group is None:
        indices = list(range(len(pot_positions)))
    else:
        indices = config.layer_indices(group)

    x_min = 0
    y_min = 0
    x_max = target_img.shape[0]
    y_max = target_img.shape[1]

    pos = np.concatenate([pot_positions[i] for i in indices], axis=0)
    ints = np.concatenate([pot_intensities[i] for i in indices], axis=0)

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
        config = deserialize(config_json, CONFIG_CLASS_REGISTRY)


    params = ParticlesTunableParameters(**params_json)

    pixel_size_nm = config.pixel_size_nm

    train_method = TrainMethod().ensure_for_config(config)
    pot_layer = DifferentiablePotential.from_config_and_parameters(config, train_method, params.potential_parameters)

    layer_names = config.layer_names
    pot_imgs = pot_layer.get_potential_images()

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

    # Farbschema nach Element - laeuft damit fuer beliebige Schichten
    element_colors = {"C": "Greys", "S": red, "W": blue}

    def normed(img):
        return (img - np.min(img)) / (np.max(img) - np.min(img))

    def save_map(img, name, cmap):
        plt.figure()
        plt.imshow(normed(img), cmap=cmap, origin="lower")
        plt.title("Potential Image " + name)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(param_file.replace(".json", "_" + name + ".png"), dpi=300)

    for i, layer in enumerate(config.layers):
        save_map(pot_imgs[i], layer.name, element_colors.get(layer.element, "viridis"))

    # Summenbild je Gruppe
    for group in config.groups:
        indices = config.layer_indices(group.name)
        if len(indices) < 2:
            continue
        save_map(sum(pot_imgs[i] for i in indices), group.name + "_sum", "seismic")

    # Klassischer WS2-Kontrast (S-Summe minus W), sofern die Schichten existieren
    if all(n in layer_names for n in ("S1", "S2", "W")):
        map_S = normed(pot_imgs[layer_names.index("S1")] + pot_imgs[layer_names.index("S2")])
        map_W = normed(pot_imgs[layer_names.index("W")])

        plt.figure()
        plt.imshow(np.abs(map_S) - np.abs(map_W), cmap="seismic", origin="lower")
        plt.title("Potential Image " + "WS2")
        plt.savefig(param_file.replace(".json", "_" + "WS2" + ".png"), dpi=300)

    pot_positions = []
    pot_intensities = []
    for i in range(len(config.layers)):
        (px, py), ints = pot_layer.get_pos_and_ints(i)
        pos = np.column_stack([px, py]).astype(np.float32)
        ints = np.asarray(ints, dtype=np.float32)
        pot_positions.append(pos)
        pot_intensities.append(ints)

    target_img = tifffile.imread(os.path.join(path, "target_div-b.tif"))
    pos, ints = get_pos_and_ints(target_img, pot_positions, pot_intensities, config)

    plt.imshow(target_img, cmap="grey", origin="lower")
    plt.scatter(pos[1], pos[0], c=ints, cmap="Wistia", s=0.5)
    plt.xticks([])
    plt.yticks([])
    plt.title("")
    plt.tight_layout()
    save_img_file = param_file.replace(".json", "_" + "target-scatter" + ".png")
    plt.savefig(save_img_file, dpi=300)






    for i in range(len(config.layers)):
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
        save_img_file = param_file.replace(".json", "_" + "vector_map_"+layer_names[i] + ".png")
        plt.savefig(save_img_file, dpi=300)


    for i in range(len(config.layers)):
        plt.figure()
        pad = pot_layer.config.padding
        positions = pot_layer.get_positions()[i].cpu().detach().cpu().numpy()
        limit_mask = ((positions[:, 0] >= pad) & (positions[:, 0] < pot_layer.H + pad) &
                      (positions[:, 1] >= pad) & (positions[:, 1] < pot_layer.W + pad))
        occ = pot_layer.occupancies[i].sigmoid().detach().cpu().numpy()[limit_mask]
        plt.hist(occ, bins= 100)
        plt.title("Occupancy Histogram " + layer_names[i])
        plt.xlabel("Occupancy Value")
        plt.ylabel("Frequency")
        plt.yticks([])
        save_img_file = param_file.replace(".json", "_" + "occ_hist_" + layer_names[i] + ".png")
        plt.savefig(save_img_file, dpi=300)

        plt.figure()
        plt.scatter(positions[limit_mask][:, 1], positions[limit_mask][:, 0], c=occ, cmap=blue, s=0.5)
        plt.colorbar(label="Occupancy Value")
        plt.title("Occupancy Scatter " + layer_names[i])
        plt.xticks([])
        plt.yticks([])
        save_img_file = param_file.replace(".json", "_" + "occ_scatter_" + layer_names[i] + ".png")
        plt.savefig(save_img_file, dpi=300)

