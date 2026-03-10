# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tifffile as tifffile
import torch
import matplotlib.pyplot as plt
from ase.io import read, write

from gui import LiveTrainerGUI
from nn import NN_particles, NN_free_potential
import ase
import abtem
import numpy as np
from scipy.ndimage import gaussian_filter

torch.set_default_device("cuda")


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
    #model = read("model_optimized.xyz")
    ws2 = ase.build.mx2(formula='WS2', vacuum=1.0)*(50,50,1)
    ws2.translate((-10, -10, 0))
    graphen = ase.build.graphene(vacuum=1.0)*(60,60,1)
    graphen.rotate(0, 'z', )
    graphen.translate((-10, 0,0))

    model = ws2

    fov = 7.58#4.163565463642421
    slices = True
    if slices:
        init_positions = model.get_positions() / ((fov*10) / image.shape[1])
        lowS = np.min(init_positions[:,2])
        highS = np.max(init_positions[:,2])
        middleW = (lowS + highS) / 2
        slice_thickness = highS - middleW
        init_positions = init_positions - lowS + slice_thickness / 2
        slices_pos = []
        slices_intensities = []
        ###graphen
        graphen_pos = graphen.get_positions() / ((fov*10) / image.shape[1])
        selector = (graphen_pos[:, 0] > 0.0) & (graphen_pos[:, 0] < image.shape[0]) & \
                   (graphen_pos[:, 1] > 0.0) & (graphen_pos[:, 1] < image.shape[1])
        slices_pos.append(graphen_pos[selector][:, :2])
        slices_intensities.append(graphen.get_atomic_numbers()[selector])


        for i in range(3):
            selector = (init_positions[:, 2] > i * slice_thickness) & (init_positions[:, 2] < (i + 1) * slice_thickness) & \
                       (init_positions[:, 0] > 0.0) & (init_positions[:, 0] < image.shape[0]) & \
                          (init_positions[:, 1] > 0.0) & (init_positions[:, 1] < image.shape[1])
            slices_pos.append(init_positions[selector][:, :2])
            slices_intensities.append(model.get_atomic_numbers()[selector])
        init_positions = slices_pos
        init_intensities = slices_intensities
    else:
        init_positions = model.get_positions()[:,:2] / ((fov*10) / image.shape[1])
        init_positions = np.stack([init_positions], axis=-1)
        #init_positions = torch.rand(1000, 2)  * torch.tensor([image.shape[1]-1, image.shape[0]-1])

        init_intensities = model.get_atomic_numbers() #** contrast
        init_intensities = np.stack([init_intensities], axis=-1)
    #init_intensities = torch.tensor(torch.ones(1000), dtype=torch.float32)



    #target = F.gaussian_blur(target.unsqueeze(0).unsqueeze(0), kernel_size=(5, 5), sigma=1.0).squeeze()
    target = torch.tensor(image, dtype=torch.float32)  # Hinzufügen einer Batch-Dimension

    brightness = torch.mean(target)
    contrast = torch.std(target)
    min_target = torch.min(target)
    max_target = torch.max(target)
    target = (target - min_target) / (max_target - min_target)  # Normalisierung des Zielbildes
    #target = (target - brightness) / contrast  # Normalisierung des Zielbildes
    canvas_size = int(np.max(image.shape) / 2)

    init_potential = torch.zeros((image.shape[1] + canvas_size - 1,
                                      image.shape[0] + canvas_size - 1))
    init_potential = torch.randint(low=0, high=10, size=(image.shape[1],
                                  image.shape[0])).float()
    # Modell
    #model = NN_free_potential(init_potential,device='cpu')
    model = NN_particles(image_shape=image.shape, fov=fov, voltage_kV=VOLTAGE_KV, device='cuda')
    model.build_multislice(init_positions=init_positions, init_intensities=init_intensities, dz=slice_thickness/10)

    ctf_layer = model.model[2]
    pot_layer = model.model[0]

    # Optimierer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # --- Plot Setup ---
    plt.ion()
    fig, ax2s = plt.subplots(3, 4, figsize=(8, 4))
    (ax_loss, ax_target, ax_img, ax_diff, ax_ctf, ax_env, ax_hist, ax_pot, ax_pot0, ax_pot1, ax_pot2, ax_pot3) = ax2s.flatten()
    losses = []





    # --- Trainingsloop ---
    for step in range(2000000):
        optimizer.zero_grad()
        #simulated_image,loss = model.loss(target, brightness, contrast)
        simulated_image, loss = model.loss_multislice(target, brightness, contrast)
        loss.backward()
        optimizer.step()
        #model.clamp_parameters()
        losses.append(loss.item())

        # Alle paar Schritte plotten
        if step % 5 == 0 or step == 0:
            ax_loss.clear()
            ax_loss.plot(losses)
            ax_loss.set_xlabel("Step")
            ax_loss.set_ylabel("Log Loss")
            ax_loss.set_title("Loss-Verlauf")
            ax_loss.legend()

            ax_img.clear()
            if True:
                sim_img = simulated_image.detach().cpu().numpy()
            else:
                sim_img = model.simulation.simulate(model.simulation.calc_potential(model.positions, model.intensities),
                                                    model.aberrations).cpu().detach().numpy()
            #sim_img = model.simulation.simulate(model.simulation.calc_potential(model.positions, model.intensities),
            #                                    model.aberrations).cpu().detach().numpy()
            ax_img.clear()
            ax_img.imshow(sim_img, cmap="viridis")
            ax_img.set_title(f"Simuliertes Bild (Step {step})")
            ax_img.axis("off")
            diff_image = (sim_img - target.detach().cpu().numpy()) ** 2
            ax_diff.clear()
            ax_diff.imshow(diff_image, cmap="viridis")
            ax_diff.set_title(f"Diff Bild (Step {step})")
            ax_diff.axis("off")

            ax_target.clear()
            ax_target.imshow(target.detach().cpu().numpy(), cmap="viridis")
            ax_target.set_title(f"Orig Bild (Step {step})")
            ax_target.axis("off")

            potentials, _ = pot_layer(None)
            pot_axs = [ax_pot0, ax_pot1, ax_pot2, ax_pot3]

            for i, pot_i in enumerate(potentials):
                ax_p = pot_axs[i]
                ax_p.clear()
                pot_i_blurred = gaussian_filter(pot_i.detach().cpu().numpy(), sigma=5.)
                ax_p.imshow(pot_i_blurred, cmap="viridis")
                ax_p.set_title(f"Blurred Potential{i} (Step {step})")
                ax_p.axis("off")



            dp, env, ring_mask, psf = ctf_layer.to_diffraction_patterns(max_angle_mrad=42., gpts=1300)

            ctf_map = dp.detach().cpu().numpy()
            env_map = env.detach().cpu().numpy()
            psf_map = psf.abs().detach().cpu().numpy()

            ax_ctf.clear()
            ax_ctf.imshow(np.angle(ctf_map), cmap='RdBu', vmin=-np.pi, vmax=np.pi)
            ax_ctf.imshow(ring_mask.cpu().numpy(), cmap="Greens", alpha=0.4, origin="lower")



            ax_ctf.set_title(f"CTF (Step {step} \n C10 : {ctf_layer.aberr_params['C10'].detach().cpu().numpy():.2f} nm) \n"
                             f"C30 : {ctf_layer.aberr_params['C30'].detach().cpu().numpy():.2f} um)")
            ax_env.clear()
            ax_env.imshow(np.log10(env_map), cmap='viridis')
            ax_env.set_title(f"CTF Env (Step {step})")

            ax_hist.clear()
            ax_hist.imshow(psf_map, cmap='RdBu', vmin=-np.pi, vmax=np.pi)



            #ax_hist.imshow(np.abs(model.model[1]._build_ctf().detach().numpy()))
            intensities = pot_layer.get_intensities()

            ax_pot.clear()
            if pot_layer.intensity_classes == "vacancies":
                ax_pot.hist(torch.sigmoid(pot_layer.vacanciesS1).detach().cpu().numpy(), bins=10, alpha=0.5, label='S low')
                ax_pot.hist(torch.sigmoid(pot_layer.vacanciesW).detach().cpu().numpy(), bins=10, alpha=0.5, label='W')
                ax_pot.hist(torch.sigmoid(pot_layer.vacanciesS2).detach().cpu().numpy(), bins=10, alpha=0.5, label='S high')
                ax_pot.legend()
            elif pot_layer.intensity_classes == "free":
                ax_pot.hist(intensities[0].detach().cpu().numpy(), bins=100, alpha=0.5, label='Graphen')
                ax_pot.hist(intensities[1].detach().cpu().numpy(), bins=100, alpha=0.5, label='S low')
                ax_pot.hist(intensities[2].detach().cpu().numpy(), bins=100, alpha=0.5, label='W')
                ax_pot.hist(intensities[3].detach().cpu().numpy(), bins=100, alpha=0.5, label='S high')
                ax_pot.legend()
            elif pot_layer.intensity_classes == "free_ws2":
                ax_pot.hist(intensities[1].detach().cpu().numpy(), bins=100, alpha=0.5, label='S low')
                ax_pot.hist(intensities[2].detach().cpu().numpy(), bins=100, alpha=0.5, label='W')
                ax_pot.hist(intensities[3].detach().cpu().numpy(), bins=100, alpha=0.5, label='S high')
                ax_pot.legend()

            ax_pot.set_title("Intensities")


            plt.pause(1.01)

    plt.ioff()
    plt.show()



