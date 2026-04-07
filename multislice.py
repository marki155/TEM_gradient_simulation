from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math




def electron_wavelength_nm(voltage_kV):
    h = 6.62607015e-34
    m0 = 9.1093837015e-31
    e = 1.602176634e-19
    c = 299792458.0

    U = voltage_kV * 1e3
    p = math.sqrt(2 * m0 * e * U * (1 + e * U / (2 * m0 * c * c)))
    lam_m = h / p
    return lam_m * 1e9

def sigma_electron(wavelength_m):
    m = 9.10938356e-31      # Elektronenmasse (kg)
    e = 1.602176634e-19     # Elementarladung (C)
    h = 6.62607015e-34      # Planck-Konstante (Js)

    sigma = (m * e * wavelength_m) / (h ** 2)
    print(sigma)
    return sigma

def interaction_parameter_rad_per_Vnm(voltage_kV):
    h = 6.62607015e-34
    m0 = 9.1093837015e-31
    e = 1.602176634e-19
    c = 299792458.0

    U = voltage_kV * 1e3
    E0 = m0 * c * c
    E = e * U

    p = math.sqrt(2 * m0 * E * (1 + E / (2 * E0)))
    lam_m = h / p
    gamma = 1.0 + E / E0

    sigma_SI = 2 * math.pi * gamma * m0 * e * lam_m / (h * h)  # rad / (V*m)
    return sigma_SI * 1e-9  # rad / (V*nm)


def propagate_fresnel(wave, dz_nm, pixel_size_nm, wavelength_nm):
    H, W = wave.shape
    device = wave.device

    fy = torch.fft.fftfreq(H, d=pixel_size_nm, device=device)
    fx = torch.fft.fftfreq(W, d=pixel_size_nm, device=device)
    Y, X = torch.meshgrid(fy, fx, indexing="ij")
    k2 = X**2 + Y**2

    P = torch.exp(-1j * math.pi * wavelength_nm * dz_nm * k2)

    wave_ft = torch.fft.fft2(wave)
    return torch.fft.ifft2(wave_ft * P)




class PhysicalPhaseObjectLayer(nn.Module):
    def __init__(self, H, W, pixel_size_nm, voltage_kV, device):
        super().__init__()
        self.H = H
        self.W = W
        self.pixel_size_nm = pixel_size_nm
        self.voltage_kV = voltage_kV
        self.device = device

        self.lambda_nm = electron_wavelength_nm(voltage_kV)
        self.sigma = interaction_parameter_rad_per_Vnm(voltage_kV)

        self.layer_names = ["Gr", "S1", "W", "S2"]
        self.atom_names_by_layer = ["C", "S", "W", "S"]

    def forward(self, potential_slices, z_positions_nm):
        """
        positions: Liste [Gr, S1, W, S2], each Tensor (N,2)
        occupancies: Liste [Gr, S1, W, S2], each Tensor (N,)
        z_positions_nm: z-Koordinaten der vier Ebenen in nm, z.B. [0.0, 0.55, 0.70, 0.85]
        """

        order = torch.argsort(torch.tensor(z_positions_nm, device=self.device))
        z_sorted = [z_positions_nm[i] for i in order.tolist()]

        wave = torch.ones((self.H, self.W), dtype=torch.complex64, device=self.device)

        for i in range(len(z_sorted)):
            V_slice = potential_slices[i]

            phase = self.sigma * V_slice

            phase_np = self.sigma * V_slice.detach().cpu().numpy()
            mask_outside = (phase_np < 2 * -np.pi) | (phase_np > 2 * np.pi)
            if mask_outside.any():
                print("PHASE too high:")
                print(phase_np[mask_outside])
                positions = np.argwhere(mask_outside)
                print(positions)


            T = torch.exp(1j * phase)
            wave = wave * T

            if i < len(z_sorted) - 1:
                dz_nm = z_sorted[i + 1] - z_sorted[i]
                wave = propagate_fresnel(
                    wave=wave,
                    dz_nm=dz_nm,
                    pixel_size_nm=self.pixel_size_nm,
                    wavelength_nm=self.lambda_nm,
                )

        return wave


class DifferentiableMultislice(nn.Module):
    def __init__(self, dz, shape, pixel_size_nm, wavelength_nm, device, train_method, voltage_kV):
        super().__init__()
        #self.slices_V = [v.to(device) for v in slices_V]  # Liste komplexer Slices (H,W)
        #self.dz = dz

        self.z_positions_nm = [
            0.00,  # Graphen
            0.55,  # S1
            0.70,  # W
            0.85,  # S2
        ]
        # TODO self.z_positions_nm = z_positions_nm

        self.pixel_size_nm = pixel_size_nm
        self.wavelength_nm = wavelength_nm
        self.device = device

        self.H, self.W = shape
        #fx = torch.fft.fftshift(torch.linspace(-0.5 / pixel_size_nm, 0.5 / pixel_size_nm, self.H, device=device))
        #fy = torch.fft.fftshift(torch.linspace(-0.5 / pixel_size_nm, 0.5 / pixel_size_nm, self.W, device=device))

        # kx, ky = torch.meshgrid(fx, fy, indexing='xy')
        #H_prop = torch.exp(-1j * torch.pi * self.wavelength_nm * self.dz* (kx ** 2 + ky ** 2))
        #self.register_buffer("H_prop", H_prop)


        self.train_method = train_method

        self.phase_object_layer = PhysicalPhaseObjectLayer(self.H, self.W, pixel_size_nm, voltage_kV, device)



    def forward(self, input):
        potential_list, pot_aux_loss = input


        exit_wave = self.phase_object_layer(
            potential_slices=potential_list,
            z_positions_nm=self.z_positions_nm,
        )
        print(pot_aux_loss)

        return exit_wave, pot_aux_loss



