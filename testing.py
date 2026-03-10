from nn import CTFLayer, DifferentiableMultislice, NN_particles

import matplotlib.pyplot as plt
import torch
import numpy as np

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

def relativistic_wavelength_nm(voltage_kV):
    """Relativistische Elektronenwellenlänge in nm."""
    V = voltage_kV * 1e3
    h = 6.62607015e-34
    m0 = 9.10938356e-31
    e = 1.602176634e-19
    c = 299792458.0
    E = e * V
    p = math.sqrt(E**2 + 2 * E * m0 * c**2) / c
    lam_m = h / p
    return lam_m * 1e9  # nm

class CTFLayer(nn.Module):
    def __init__(self, H, W, pixel_size_nm, voltage_kV, df_nm, Cs_nm,
                 amp_contrast=0.07, device='cpu'):
        super().__init__()
        self.H = H
        self.W = W
        self.pixel_size_nm = pixel_size_nm
        self.voltage_kV = voltage_kV
        self.df_nm = df_nm
        self.Cs_nm = Cs_nm
        self.amp_contrast = amp_contrast
        self.device = device

        # Frequenzen in 1/nm
        fx = torch.fft.fftshift(torch.linspace(-0.5/pixel_size_nm, 0.5/pixel_size_nm, W, device=device))
        fy = torch.fft.fftshift(torch.linspace(-0.5/pixel_size_nm, 0.5/pixel_size_nm, H, device=device))
        kx, ky = torch.meshgrid(fx, fy, indexing='xy')
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def _build_ctf(self):
        lam = relativistic_wavelength_nm(self.voltage_kV)  # nm
        k2 = self.kx**2 + self.ky**2
        k = torch.sqrt(k2 + 1e-30)

        # Wellenaberrationsfunktion W(k)
        W = -0.5 * self.df_nm * lam * (k**2) + (self.Cs_nm * lam**3 / 4.0) * (k**4)

        chi = 2.0 * math.pi * W
        A = self.amp_contrast
        ctf = (1.0 - A) * torch.sin(chi) + A * torch.cos(chi)
        return ctf


if False and __name__ == "__main__":
    H, W = 512, 512
          # 0.1 nm / Pixel (1 Å/Pixel)
    voltage_kV = 80.         # 300 kV TEM
    df_nm = -57               # Defokus: 1500 nm = 1.5 µm
    Cs_nm = 5.0e3              # Cs: 2 mm = 2e6 nm
    amp_contrast = .1

    pixel_size_nm = 40/512

    ctf_layer = CTFLayer(H, W, pixel_size_nm, voltage_kV,
                         df_nm, Cs_nm, amp_contrast)

    ctf_map = ctf_layer._build_ctf().detach().numpy()

    plt.imshow(np.fft.fftshift(ctf_map), cmap='seismic', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f"CTF at {voltage_kV} kV, df={df_nm} nm, Cs={Cs_nm/1e6} mm")
    plt.show()


if __name__ == '__main__':
    # Beispielzielbild (ein Partikel an Position (0.3, 0.6))
    H, W = 400, 400
    imag = torch.zeros((H, W))
    imag[200, 200] = 0.0

    aberrations = {'C10': -57., 'C12': 100.0, 'C23': 3.0e3,
                   'C21': 50.0e3, 'C30': 5.0e6, 'C34': 20.0e3, 'C32': 50.0e3}

    aberrations = {"C10": -57.,
                   'C12': 1.0, "phi12": 0.,
                   'C23': .5, 'phi23': 0.,
                   'C21': .5, 'phi21': 0.,
                   "C30": 2.}

    fov = 4.163565463642421
    init_positions = np.array([[200,200]])
    init_intensities = np.array([1])  # ** contrast

    # target = F.gaussian_blur(target.unsqueeze(0).unsqueeze(0), kernel_size=(5, 5), sigma=1.0).squeeze()
    target = torch.tensor(imag, dtype=torch.float32)  # Hinzufügen einer Batch-Dimension

    brightness = torch.mean(target)
    contrast = torch.std(target)
    min_target = torch.min(target)
    max_target = torch.max(target)
    # target = (target - min_target) / (max_target - min_target)  # Normalisierung des Zielbildes
    target = (target - brightness) / contrast  # Normalisierung des Zielbildes
    canvas_size = int(np.max(imag.shape) / 2)

    init_potential =  torch.zeros((imag.shape[1],
                                  imag.shape[0]))
    # Modell
    # model = NN_free_potential(init_potential, image_shape=image.shape, fov=fov, device='cpu')
    model = NN_particles(init_positions, init_intensities, image_shape=imag.shape, fov=fov, device='cpu')
    model.build_multislice(1, init_potential.shape, .1, 1, fov / imag.shape[0], voltage_kV=80)

    simulated_image, loss = model.loss_multislice(target, brightness, contrast)

    fig, (ax_loss, ax_target, ax_img, ax_diff, ax_pot, ax_hist) = plt.subplots(1, 6, figsize=(8, 4))
    ax_loss.clear()

    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss-Verlauf")
    ax_loss.legend()

    ax_img.clear()
    if True:
        sim_img = simulated_image.detach().numpy()
    else:
        sim_img = model.simulation.simulate(model.simulation.calc_potential(model.positions, model.intensities),
                                            model.aberrations).cpu().detach().numpy()
    # sim_img = model.simulation.simulate(model.simulation.calc_potential(model.positions, model.intensities),
    #                                    model.aberrations).cpu().detach().numpy()
    ax_img.imshow(sim_img, cmap="viridis")
    ax_img.set_title(f"Simuliertes Bild")
    #ax_img.axis("off")
    diff_image = (sim_img - target.numpy()) ** 2
    ax_diff.imshow(diff_image, cmap="viridis")
    ax_diff.set_title(f"Diff Bild")
    #ax_diff.axis("off")

    ax_target.imshow(imag, cmap="viridis")
    ax_target.set_title(f"Orig Bild")
    #ax_target.axis("off")

    ax_pot.imshow(model.simulation.calc_potential(model.positions, model.intensities).detach().numpy(), cmap="viridis")

    if True:
        dp = model.model[1].to_diffraction_patterns(max_angle_mrad=42., gpts=512)

        ctf_map = dp.detach().numpy()

        ax_hist.imshow(np.angle(ctf_map), cmap='RdBu', vmin=-np.pi, vmax=np.pi)
        # ax_hist.imshow(np.abs(model.model[1]._build_ctf().detach().numpy()))
    # ax_hist.clear()
    # ax_hist.hist(model.intensities.detach().numpy(), bins=100, alpha=0.5, label='Simuliertes Bild')

    plt.show()

    #print("Simulated Image Shape:", simulated_image.shape)
    #print("Simulated Image:", simulated_image)

    # Plot the target and simulated images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Target Image")
    ctf_img = np.imag(ctf._build_ctf().detach().numpy())

    plt.imshow(ctf_img, cmap='gray')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("Simulated Image")
    plt.colorbar()
    plt.show()
# This code simulates a simple CTF layer and visualizes the target and simulated images.

from matplotlib.patches import Rectangle

def show_init_positions(H, W, init_pos, init_intensities, canvas_shift_graphen, canvas_size_graphen, canvas_shift_WS2, canvas_size_ws2):


    fig, ax = plt.subplots(1, 2)

    init_graphen = init_pos[0]
    init_ws2 = init_pos[1:]

    ax[0].scatter(init_graphen[:, 0], init_graphen[:, 1], c=init_intensities[0], cmap="viridis")



    rect = Rectangle(
        (0, 0),  # linke untere Ecke: (x, y)
        H,  # Breite
        W,  # Höhe
        fill=False,
        edgecolor='red',
        linewidth=2
    )
    rect_canv = Rectangle(
        (- canvas_shift_graphen[0], - canvas_shift_graphen[1]),  # linke untere Ecke: (x, y)
        canvas_size_graphen[0],  # Breite
        canvas_size_graphen[1],  # Höhe
        fill=False,
        edgecolor='blue',
        linewidth=2
    )
    ax[0].add_patch(rect)
    ax[0].add_patch(rect_canv)



    ax[0].set_title("Initiale Positionen Graphen")

    for i, pos in enumerate(init_ws2):
        ax[1].scatter(pos[:,0], pos[:,1], c=init_intensities[i+1], cmap="viridis")

    rect = Rectangle(
        (0, 0),  # linke untere Ecke: (x, y)
        H,  # Breite
        W,  # Höhe
        fill=False,
        edgecolor='red',
        linewidth=2
    )
    rect_canv = Rectangle(
        (- canvas_shift_WS2[0], - canvas_shift_WS2[1]),  # linke untere Ecke: (x, y)
        canvas_size_ws2[0],  # Breite
        canvas_size_ws2[1],  # Höhe
        fill=False,
        edgecolor='blue',
        linewidth=2
    )
    ax[1].add_patch(rect)
    ax[1].add_patch(rect_canv)

    ax[1].set_title("Initiale Positionen WS2")

    plt.show()