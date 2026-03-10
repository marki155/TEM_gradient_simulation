import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableMultislice(nn.Module):
    def __init__(self, dz, shape, pixel_size_nm, wavelength_nm, device, train_method):
        super().__init__()
        #self.slices_V = [v.to(device) for v in slices_V]  # Liste komplexer Slices (H,W)
        self.dz = dz
        self.pixel_size_m = pixel_size_nm
        self.wavelength_m = wavelength_nm
        self.device = device
        self.sigma = torch.nn.Parameter(torch.tensor(0.0177, dtype=torch.float32, device=device), requires_grad=True) #0.0177#

        self.H, self.W = shape
        fx = torch.fft.fftshift(torch.linspace(-0.5 / pixel_size_nm, 0.5 / pixel_size_nm, self.W, device=device))
        fy = torch.fft.fftshift(torch.linspace(-0.5 / pixel_size_nm, 0.5 / pixel_size_nm, self.H, device=device))

        kx, ky = torch.meshgrid(fx, fy, indexing='xy')
        H_prop = torch.exp(-1j * torch.pi * self.wavelength_m * self.dz* (kx ** 2 + ky ** 2))
        self.register_buffer("H_prop", H_prop)

        self.train_method = train_method



    def forward(self, input):
        wave = torch.ones((self.H, self.W), dtype=torch.complex64, device=self.device)

        if self.train_method.train_multislice_sigma:
            sigma = self.sigma
        else:
            sigma = self.sigma.detach()


        sigma = torch.clamp(sigma, 0.001, 0.05)  # Begrenzen Sie den Bereich von sigma

        for potential in input[0]:
            wave = wave * torch.exp(1j * sigma * potential ) # Transmission durch Slice

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
        self.pixel_size_m = pixel_size_nm
        self.wavelength_m = wavelength_nm
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