from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def sigma_electron(wavelength_m):
    m = 9.10938356e-31      # Elektronenmasse (kg)
    e = 1.602176634e-19     # Elementarladung (C)
    h = 6.62607015e-34      # Planck-Konstante (Js)

    sigma = (m * e * wavelength_m) / (h ** 2)
    print(sigma)
    return sigma


@dataclass
class TunableMultisliceParameters:
    sigma: nn.Parameter = nn.Parameter(torch.tensor(.2, dtype=torch.float32), requires_grad=True)

# TODO vllt gleich am anfang init von den parametern hier?


class DifferentiableMultislice(nn.Module):
    def __init__(self, dz, shape, pixel_size_nm, wavelength_nm, device, train_method):
        super().__init__()
        #self.slices_V = [v.to(device) for v in slices_V]  # Liste komplexer Slices (H,W)
        self.dz = dz
        self.pixel_size_nm = pixel_size_nm
        self.wavelength_nm = wavelength_nm
        self.device = device

        self.H, self.W = shape
        fx = torch.fft.fftshift(torch.linspace(-0.5 / pixel_size_nm, 0.5 / pixel_size_nm, self.H, device=device))
        fy = torch.fft.fftshift(torch.linspace(-0.5 / pixel_size_nm, 0.5 / pixel_size_nm, self.W, device=device))

        kx, ky = torch.meshgrid(fx, fy, indexing='xy')
        H_prop = torch.exp(-1j * torch.pi * self.wavelength_nm * self.dz* (kx ** 2 + ky ** 2))
        self.register_buffer("H_prop", H_prop)

        self.tunable_parameters = TunableMultisliceParameters()

        self.train_method = train_method

        for name, value in vars(self.tunable_parameters).items():
            if isinstance(value, (nn.Parameter, nn.ParameterList, nn.ParameterDict)):
                setattr(self, name, value)

    def get_tunable_parameters(self):
        return self.tunable_parameters



    def forward(self, input):
        wave = torch.ones((self.H, self.W), dtype=torch.complex64, device=self.device)

        if self.train_method.train_multislice_sigma:
            sigma = self.tunable_parameters.sigma
        else:
            sigma = self.tunable_parameters.sigma.detach()


        #sigma = torch.clamp(sigma, 0.0001, 1.0)  # Begrenzen Sie den Bereich von sigma
        sigma = sigma_electron(wavelength_m=self.wavelength_nm*1e-9)
        sigma = sigma
        m0 = 9.10938356e-31  # kg
        eC = 1.602176634e-19  # C
        h = 6.62607015e-34  # J·s
        c = 2.99792458e8  # m/s

        V_acc = 80 * 1e3  # V

        # relativistische Wellenlänge λ

        #E0 = m0 * c ** 2
        #E = eC * V_acc
        #lambda_rel = h / torch.sqrt(2 * m0 * E * (1 + E / (2 * E0)))  # m
        #l#ambda_nm = lambda_rel * 1e9  # nm
        #s#igma = 2 * torch.pi / lambda_nm * eC  # nm^-1·V^-1

        for potential in input[0]:
            phase = sigma * potential * self.dz
            phase_np = sigma * potential.detach().cpu().numpy()
            mask_outside = (phase_np < 2 * -np.pi) | (phase_np > 2 * np.pi)
            if mask_outside.any():
                print("PHASE too high:")
                print(phase_np[mask_outside])
                positions = np.argwhere(mask_outside)
                print(positions)
            wave = wave * torch.exp(1j * phase) # Transmission durch Slice

            # Freiraumpropagation mit Transferfunktion
            wave_ft = torch.fft.fft2(wave)

            wave_ft = wave_ft * self.H_prop
            wave = torch.fft.ifft2(wave_ft)
        return wave, input[1]

class DifferentiableMultislice_Free(nn.Module):
    def __init__(self, potential, dz, shape, pixel_size_nm, wavelength_nm, device):
        super().__init__()
        # self.slices_V = [v.to(device) for v in slices_V]  # Liste komplexer Slices (H,W)
        self.dz = dz
        self.pixel_size_nm = pixel_size_nm
        self.wavelength_nm = wavelength_nm
        self.device = device
        self.potential = potential

        self.H, self.W = shape
        fx = torch.fft.fftshift(torch.linspace(-0.5 / pixel_size_nm, 0.5 / pixel_size_nm, self.W, device=device))
        fy = torch.fft.fftshift(torch.linspace(-0.5 / pixel_size_nm, 0.5 / pixel_size_nm, self.H, device=device))

        kx, ky = torch.meshgrid(fx, fy, indexing='xy')
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, wave_in):
        plane_wave = torch.ones((self.H, self.W), dtype=torch.complex64, device=self.device)

        wave = plane_wave * self.potential  # Transmission durch Slice

        # Freiraumpropagation mit Transferfunktion
        wave_ft = torch.fft.fft2(wave)
        H_free = torch.exp(-1j * torch.pi * self.wavelength_m * self.dz * (self.kx ** 2 + self.ky ** 2))
        wave_ft = wave_ft * H_free
        wave = torch.fft.ifft2(wave_ft)
        return wave