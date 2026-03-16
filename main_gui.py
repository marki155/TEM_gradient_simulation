# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tifffile as tifffile
import torch
import matplotlib.pyplot as plt
from ase.io import read, write

from gui import LiveTrainGUI
from nn import NN_particles, NN_free_potential
import ase
import abtem
import numpy as np

from scipy.ndimage import gaussian_filter

from testing import show_init_positions
from training_parameter import TrainMethod

torch.set_default_device("cpu")


VOLTAGE_KV = 80

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Beispielzielbild (ein Partikel an Position (0.3, 0.6))
    #H, W = 40,40
    #target = torch.zeros((H, W))
    #target[10, 10] = 1.0

    #image = -tifffile.imread('NPL11-1_crop_edge.tif').T
    image = tifffile.imread('NPL11-1_crop_big.tif')
    image[np.where(image == 0.0)] = np.median(image)
    H, W = image.shape
    fov = 7.58#7.58  # 4.163565463642421 # TODO als Parameter

    pixel_size_nm = fov / image.shape[1]

    max_rot_graphen = np.deg2rad(10)
    canvas_H_rot_graphen = np.cos(max_rot_graphen) * H + np.sin(max_rot_graphen) * W
    canvas_W_rot_graphen = np.cos(max_rot_graphen) * W + np.sin(max_rot_graphen) * H

    max_shift_graphen = 3*2.47 * .1 / pixel_size_nm
    canvas_H_shift_graphen = H + 2 * max_shift_graphen
    canvas_W_shift_graphen = W + 2 * max_shift_graphen

    canvas_size_graphen = (int(max(canvas_H_rot_graphen, canvas_H_shift_graphen)),
                           int(max(canvas_W_rot_graphen, canvas_W_shift_graphen)))
    canvas_shift_graphen = (canvas_size_graphen[0] - H) / 2, (canvas_size_graphen[1] - W) / 2

    max_rot_ws2 = np.deg2rad(5)
    canvas_H_rot_ws2 = np.cos(max_rot_ws2) * H + np.sin(max_rot_ws2) * W
    canvas_W_rot_ws2 = np.cos(max_rot_ws2) * W + np.sin(max_rot_ws2) * H
    max_shift_ws2 = 3*3.18 * .1 / pixel_size_nm
    canvas_H_shift_ws2 = H + 2 * max_shift_ws2
    canvas_W_shift_ws2 = W + 2 * max_shift_ws2
    canvas_size_ws2 = (int(max(canvas_H_rot_ws2, canvas_H_shift_ws2)),
                       int(max(canvas_W_rot_ws2, canvas_W_shift_ws2)))
    canvas_shift_WS2 = (canvas_size_ws2[0] - H) / 2, (canvas_size_ws2[1] - W) / 2


    single_WS2 = ase.build.mx2(formula='WS2')
    WS2_H_atoms = int((canvas_size_ws2[0] * pixel_size_nm * 10) / single_WS2.cell.lengths()[0]) * 2
    WS2_W_atoms =  int((canvas_size_ws2[1] * pixel_size_nm * 10) / single_WS2.cell.lengths()[1]) * 2

    single_graohen = ase.build.graphene()
    graphen_H_atoms = int((canvas_size_graphen[0] * pixel_size_nm * 10) / single_graohen.cell.lengths()[0]) * 2
    graphen_W_atoms = int((canvas_size_graphen[1] * pixel_size_nm * 10) / single_graohen.cell.lengths()[1]) * 2


    #model = read("model_optimized.xyz")
    ws2 = ase.build.mx2(formula='WS2', vacuum=1.0)*(WS2_H_atoms, WS2_W_atoms,1)
    ws2.translate((-10, -10, 0))
    graphen = ase.build.graphene(vacuum=1.0)*(graphen_H_atoms, graphen_W_atoms,1)
    graphen.rotate(0, 'z', )
    graphen.translate((-10, -18,0))

    model = ws2


    slices = True
    if slices:
        init_positions = model.get_positions() / (pixel_size_nm * 10)
        lowS = np.min(init_positions[:,2])
        highS = np.max(init_positions[:,2])
        middleW = (lowS + highS) / 2
        slice_thickness = highS - middleW
        init_positions = init_positions - lowS + slice_thickness / 2
        slices_pos = []
        slices_intensities = []
        ###graphen
        graphen_pos = graphen.get_positions() / (pixel_size_nm * 10)
        selector = ((graphen_pos[:, 0] >=  - canvas_shift_graphen[0]) &
                    (graphen_pos[:, 0] <= canvas_size_graphen[0] - canvas_shift_graphen[0]) & \
                   (graphen_pos[:, 1] >=  - canvas_shift_graphen[1]) &
                    (graphen_pos[:, 1] <= canvas_size_graphen[1] - canvas_shift_graphen[1]))
        slices_pos.append(graphen_pos[selector][:, :2])
        slices_intensities.append(graphen.get_atomic_numbers()[selector])




        for i in range(3):
            selector = ((init_positions[:, 2] > i * slice_thickness) & (init_positions[:, 2] < (i + 1) * slice_thickness) & \
                       (init_positions[:, 0] >= - canvas_shift_WS2[0]) &
                        (init_positions[:, 0] <= canvas_size_ws2[0]  - canvas_shift_WS2[0]) & \
                          (init_positions[:, 1] >= - canvas_shift_WS2[1]) &
                        (init_positions[:, 1] <= canvas_size_ws2[0] - canvas_shift_WS2[1]))
            slices_pos.append(init_positions[selector][:, :2])
            slices_intensities.append(model.get_atomic_numbers()[selector])
        init_positions = slices_pos
        init_intensities = slices_intensities
    else:
        init_positions = model.get_positions()[:,:2] / (pixel_size_nm * 10)
        init_positions = np.stack([init_positions], axis=-1)
        #init_positions = torch.rand(1000, 2)  * torch.tensor([image.shape[1]-1, image.shape[0]-1])

        init_intensities = model.get_atomic_numbers() #** contrast
        init_intensities = np.stack([init_intensities], axis=-1)
    #init_intensities = torch.tensor(torch.ones(1000), dtype=torch.float32)

    #show_init_positions(H, W, init_positions, init_intensities, canvas_shift_graphen, canvas_size_graphen, canvas_shift_WS2, canvas_size_ws2)

    #target = F.gaussian_blur(target.unsqueeze(0).unsqueeze(0), kernel_size=(5, 5), sigma=1.0).squeeze()
    target = torch.tensor(image, dtype=torch.float32)  # Hinzufügen einer Batch-Dimension

    brightness = torch.mean(target)
    contrast = torch.std(target)
    min_target = torch.min(target)
    max_target = torch.max(target)
    target = (target - min_target) / (max_target - min_target)  # Normalisierung des Zielbildes
    #target = (target - brightness) / contrast  # Normalisierung des Zielbildes
    canvas_size = 1 #int(np.max(image.shape) / 2)

    #init_potential = torch.zeros((image.shape[1] + canvas_size - 1,
    #                                  image.shape[0] + canvas_size - 1))
    #init_potential = torch.randint(low=0, high=10, size=(image.shape[1],
    #                              image.shape[0])).float()
    # Modell
    #model = NN_free_potential(init_potential,device='cpu')
    train_method = TrainMethod()





    model = NN_particles(image_shape=image.shape, fov=fov, voltage_kV=VOLTAGE_KV, train_method=train_method,
                         device='cpu', train_methods=train_method)
    model.build_multislice(init_positions=init_positions, init_intensities=init_intensities, dz=slice_thickness/10,
                           max_shifts=(max_shift_graphen, max_shift_ws2),
                           max_rotations=(max_rot_graphen, max_rot_ws2))

    ctf_layer = model.model[2]
    pot_layer = model.model[0]
    pot_layer.occupy_layers_in_area(train_method.train_small_area_squares[0], [1,2,3])

    # Optimierer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



    # SETUP GUI
    import tkinter as tk
    root = tk.Tk()

    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)




    app = LiveTrainGUI(
        root=root,
        model=model,
        optimizer=optimizer,
        target=target,
        brightness=brightness,
        contrast=contrast,
        pot_layer=pot_layer,
        ctf_layer=ctf_layer,
        plot_every=5,
        max_steps=2_000_000,
        train_method=train_method
    )

    root.mainloop()




