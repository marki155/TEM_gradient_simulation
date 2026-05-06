import tifffile as tifffile
import torch
import ase.build
import numpy as np
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from matplotlib import pyplot as plt
from nn import NN_particles, ParticlesConfig
from testing import show_init_positions
from training_parameter import TrainMethod
from gui_qt import launch_gui_qt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)

VOLTAGE_KV = 80


path = "..\platelet_11"

length_of_10_unity_cells_WS2 =  612 #px
pixel_size_nm = 3.18 / length_of_10_unity_cells_WS2

DEBUG = False
SELECTOR = True


if __name__ == '__main__':
    image = tifffile.imread(os.path.join(path, "target_div-b.tif"))
    image[np.where(image == 0.0)] = np.median(image)

    image_gr = tifffile.imread(os.path.join(path, "target_no_ws2.tif"))
    image_gr[np.where(image_gr == 0.0)] = np.median(image_gr)
    target_gr = torch.tensor(image_gr, dtype=torch.float32)

    gr_mask = tifffile.imread(os.path.join(path, "gr_mask.tif"))
    gr_mask = gr_mask > 0
    gr_mask = torch.tensor(gr_mask)
    ws2_mask = tifffile.imread(os.path.join(path, "ws2_mask.tif"))
    ws2_mask = ws2_mask > 0
    ws2_mask = torch.tensor(ws2_mask)

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
    ws2.translate((20, -90, 0))
    ws2.rotate(30, 'z')

    graphen = ase.build.graphene(vacuum=1.0) * (graphen_H_atoms, graphen_W_atoms, 1)
    #graphen.translate((H/2 * pixel_size_nm * 10, W/2 * pixel_size_nm * 10, 0))
    graphen.rotate(90, 'z')
    graphen.translate((H * 1.5 * pixel_size_nm * 10,0, 0))
    #graphen.translate((-H / 2, -W / 2, 0))
    graphen.translate((-10, -18, 0))

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
        slice_thickness = 1.0

    if DEBUG:
        show_init_positions(H, W, init_positions, init_intensities, canvas_shift_graphen, canvas_size_graphen,
                            canvas_shift_WS2, canvas_size_ws2)

    target = torch.tensor(image, dtype=torch.float32)

    brightness = torch.mean(target)
    contrast = torch.std(target)

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

    train_method = TrainMethod(area_masks=[gr_mask, ws2_mask], target_gr=target_gr)

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


    config = ParticlesConfig(init_positions=init_positions, init_intensities=init_intensities,
                             shape=(H, W), pixel_size_nm=pixel_size_nm,
                             z_positions_nm=z_positions_nm, voltage_kV=VOLTAGE_KV,
                             init_aberrations=aberrations, init_ctf_env_params=env_params,
                             init_amp_contrast=amp_contrast, ctf_alpha_max_mrad=50.,
                             padding=100, control_points_graphen=[10, 10], control_points_ws2=[10, 10])




    model = NN_particles(
        config=config,
        image_shape=image.shape,
        pixel_size_nm=pixel_size_nm,
        voltage_kV=VOLTAGE_KV,
        train_method=train_method,
        device=DEVICE,
        train_methods=train_method
    )

    model.save_particles_config(os.path.join(path, "particles_config.json"))

    model.build_multislice(
        max_shifts=(max_shift_graphen, max_shift_ws2),
        max_rotations=(max_rot_graphen, max_rot_ws2)
    )

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