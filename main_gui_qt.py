import tifffile as tifffile
import torch
from ase.io import read, write
import ase
import ase.build
import numpy as np


from nn import NN_particles, NN_free_potential
from testing import show_init_positions
from training_parameter import TrainMethod
from gui_qt import launch_gui_qt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)

VOLTAGE_KV = 80


if __name__ == '__main__':
    image = tifffile.imread('NPL11-1_crop_big_blur-3.tif')
    image[np.where(image == 0.0)] = np.median(image)

    H, W = image.shape
    fov = 7.58
    pixel_size_nm = fov / image.shape[1]

    max_rot_graphen = np.deg2rad(10)
    canvas_H_rot_graphen = np.cos(max_rot_graphen) * H + np.sin(max_rot_graphen) * W
    canvas_W_rot_graphen = np.cos(max_rot_graphen) * W + np.sin(max_rot_graphen) * H

    max_shift_graphen = 3 * 2.47 * 0.1 / pixel_size_nm
    canvas_H_shift_graphen = H + 2 * max_shift_graphen
    canvas_W_shift_graphen = W + 2 * max_shift_graphen

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
    ws2.rotate(60, 'z')

    graphen = ase.build.graphene(vacuum=1.0) * (graphen_H_atoms, graphen_W_atoms, 1)
    graphen.rotate(0, 'z')
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
        slices_pos.append(graphen_pos[selector][:, :2])
        slices_intensities.append(graphen.get_atomic_numbers()[selector])

        for i in range(3):
            selector = (
                (init_positions[:, 2] > i * slice_thickness) &
                (init_positions[:, 2] < (i + 1) * slice_thickness) &
                (init_positions[:, 0] >= -canvas_shift_WS2[0]) &
                (init_positions[:, 0] <= canvas_size_ws2[0] - canvas_shift_WS2[0]) &
                (init_positions[:, 1] >= -canvas_shift_WS2[1]) &
                (init_positions[:, 1] <= canvas_size_ws2[0] - canvas_shift_WS2[1])
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

    #show_init_positions(H, W, init_positions, init_intensities, canvas_shift_graphen, canvas_size_graphen,
    #                    canvas_shift_WS2, canvas_size_ws2)

    target = torch.tensor(image, dtype=torch.float32)

    brightness = torch.mean(target)
    contrast = torch.std(target)

    min_target = torch.min(target)
    max_target = torch.max(target)
    target = (target - min_target) / (max_target - min_target)

    train_method = TrainMethod()

    model = NN_particles(
        image_shape=image.shape,
        fov=fov,
        voltage_kV=VOLTAGE_KV,
        train_method=train_method,
        device=DEVICE,
        train_methods=train_method
    )

    model.build_multislice(
        init_positions=init_positions,
        init_intensities=init_intensities,
        dz=slice_thickness / 10,
        max_shifts=(max_shift_graphen, max_shift_ws2),
        max_rotations=(max_rot_graphen, max_rot_ws2)
    )

    ctf_layer = model.model[2]
    pot_layer = model.model[0]

    pot_layer.occupy_layers_in_area(train_method.train_small_area_squares[0], [1, 2, 3])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    launch_gui_qt(
        model=model,
        optimizer=optimizer,
        target=target,
        brightness=brightness,
        contrast=contrast,
        pot_layer=pot_layer,
        ctf_layer=ctf_layer,
        train_method=train_method,
        plot_every=20,
        heavy_every=200,
        max_steps=2_000_000,
    )