import numpy as np
import torch
import math
import torch.nn.functional as F
import torchvision


def fftconvolve_torch(a, b):
    # Beide Tensoren sollten 2D sein
    # Padding auf passende Größe (Summe der Dimensionen minus 1)
    size = [a.size(0) + b.size(0) - 1, a.size(1) + b.size(1) - 1]

    # FFT mit Padding
    fa = torch.fft.fft2(a, s=size)
    fb = torch.fft.fft2(b, s=size)

    # Elementweise Multiplikation
    fc = fa * fb

    # Inverse FFT
    conv = torch.fft.ifft2(fc).real

    return conv


def fftconvolve_valid(a, b):
    conv = fftconvolve_torch(a, b)
    valid_shape = [a.size(0) - b.size(0) + 1, a.size(1) - b.size(1) + 1]
    start0 = b.size(0) - 1
    start1 = b.size(1) - 1
    return conv[start0:start0+valid_shape[0], start1:start1+valid_shape[1]]


def gaussian_kernel(kernel_size=5, sigma=1.0, channels=1):
    # 1D Gaußkernel
    x = torch.arange(kernel_size) - kernel_size // 2
    gauss_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    gauss_1d /= gauss_1d.sum()

    # 2D Gaußkernel
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
    gauss_2d = gauss_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return gauss_2d

def gaussian_filter_torch(img, sigma):
    kernel_size = int(4 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = gaussian_kernel(kernel_size, sigma, channels=1).to(img.device).to(img.dtype)

    img = img.unsqueeze(0).unsqueeze(0)  # (N,C,H,W) mit N=1,C=1
    filtered = F.conv2d(img, kernel, padding=kernel_size//2)
    return filtered.squeeze(0).squeeze(0)


class Simulation:
    def __init__(self, shape=(512, 512), imsize=4):
        self.background = 0.0
        self.imsize = imsize
        self.kernelsize = 2
        self.device = "cpu"
        self.energy = 80000  # in keV
        self.convergence_angle = 0.02  # in radians
        self.shape = shape
        self.canvas_size = int(np.max(self.shape) / self.kernelsize)
        #self.potential_shape = np.array(shape) + np.array([self.canvas_size-1, self.canvas_size-1])
        self.potential_shape = np.array(shape)



    def calc_potential_numpy(self, pxpositions, intensities):
        self.potential.fill(self.background)
        px, py = pxpositions[:, 0], pxpositions[:, 1]
        # (px.astype(int)>0)*(px.astype(int)>self.shape[1])*(py.astype(int)>0)*(py.astype(int)>self.shape[0])
        # px = px[]
        # py = py[]

        np.add.at(self.potential,tuple([int(self.canvas_size/self.kernelsize)+py.astype(int),
                              int(self.canvas_size/self.kernelsize)+px.astype(int)]),
                    (1-(px-px.astype(int)))*(1-(py-py.astype(int)))*intensities)

        np.add.at(self.potential,tuple([int(self.canvas_size/self.kernelsize)+py.astype(int),
                              int(self.canvas_size/self.kernelsize)+px.astype(int)+1]),
                    (px-px.astype(int))*(1-(py-py.astype(int)))*intensities)

        np.add.at(self.potential,tuple([int(self.canvas_size/self.kernelsize)+(py.astype(int)+1),
                              int(self.canvas_size/self.kernelsize)+px.astype(int)+1]),
                    (px-px.astype(int))*(py-py.astype(int))*intensities)

        np.add.at(self.potential,tuple([int(self.canvas_size/self.kernelsize)+(py.astype(int)+1),
                              int(self.canvas_size/self.kernelsize)+px.astype(int)]),
                    (1-(px-px.astype(int)))*(py-py.astype(int))*intensities)


    def solve_potential(self, potential, iterations=100):
        # initial guess
        V = torch.zeros_like(potential)

        # auf (N, C, H, W) bringen
        V = V.unsqueeze(0).unsqueeze(0)
        potential = potential.unsqueeze(0).unsqueeze(0)

        for _ in range(iterations):
            V_pad = F.pad(V, (1, 1, 1, 1), mode='replicate')
            V = 0.25 * (
                V_pad[:, :, :-2, 1:-1] +   # oben
                V_pad[:, :, 2:, 1:-1]  +   # unten
                V_pad[:, :, 1:-1, :-2] +   # links
                V_pad[:, :, 1:-1, 2:]      # rechts
            ) + potential  # Update mit Source-Term

        # zurück zu (H, W)
        return V.squeeze(0).squeeze(0)



    def calc_rho(self, pxpositions, intensities, constrast=1.6, background=0, amplify=1.0):
        # pxpositions: (N, 2) Tensor
        # intensities: (N,) Tensor

        potential = torch.zeros((self.shape[1] + self.canvas_size - 1,
                                      self.shape[0] + self.canvas_size - 1))

        potential = torch.zeros((self.shape[1],
                                 self.shape[0]))


        #potential.fill_(self.background)  # Torch-Äquivalent

        py, px = pxpositions[:, 0], pxpositions[:, 1]

        #offset = int(self.canvas_size / self.kernelsize)
        offset = 0

        # Integer-Anteile
        #ix = torch.clamp(px.floor().to(torch.int64), min=0, max=self.shape[0] + self.canvas_size - 2)
        #iy = torch.clamp(py.floor().to(torch.int64), min=0, max=self.shape[1] + self.canvas_size - 2)

        ix = torch.clamp(px.floor().to(torch.int64), min=0, max=self.shape[0]-2)
        iy = torch.clamp(py.floor().to(torch.int64), min=0, max=self.shape[1]-2)

        # Bruchteile
        fx = px - ix
        fy = py - iy

        # Pixelgewichte
        w00 = torch.matmul((1 - fx), (1 - fy)) * intensities
        w10 = torch.matmul(fx, (1 - fy)) * intensities
        w11 = torch.matmul(fx, fy) * intensities
        w01 = torch.matmul((1 - fx), fy) * intensities

        # Indizes für alle vier Nachbarn
        idx00 = (offset + iy, offset + ix)
        idx10 = (offset + iy, offset + ix + 1)
        idx11 = (offset + iy + 1, offset + ix + 1)
        idx01 = (offset + iy + 1, offset + ix)

        # In-place Akkumulation wie np.add.at
        potential.index_put_(idx00, w00[:,0], accumulate=True)
        potential.index_put_(idx10, w10[:,0], accumulate=True)
        potential.index_put_(idx11, w11[:,0], accumulate=True)
        potential.index_put_(idx01, w01[:,0], accumulate=True)

        return torch.abs(potential)


    def simulate_numpy(self, aberrations):
        kernelpixel = int(self.shape[0] / self.kernelsize)

        wavelength = 1.23 / np.sqrt(self.energy * (1 + 9.78e-7 * self.energy))
        aperturesize = (self.convergence_angle / self.kernelsize) * self.imsize / wavelength

        frequencies = np.matrix(np.fft.fftshift(np.fft.fftfreq(kernelpixel, self.imsize / self.shape[0])))
        x = np.array(np.tile(frequencies, np.size(frequencies)).reshape((kernelpixel, kernelpixel)))
        y = np.array(np.tile(frequencies.T, np.size(frequencies)).reshape((kernelpixel, kernelpixel)))


        if True:
            raw_kernel = (-aberrations.get('EHTFocus', 0) * (x ** 2 + y ** 2) +

                          np.sqrt(aberrations.get('C12_a', 0) ** 2 + aberrations.get('C12_b', 0) ** 2) *
                          (x ** 2 + y ** 2) * np.cos(2 * (np.arctan2(y, x) -
                                                          np.arctan2(aberrations.get('C12_b', 0),
                                                                     aberrations.get('C12_a', 0)))) +
                          (2.0 / 3.0) *
                          np.sqrt(aberrations.get('C21_a', 0) ** 2 + aberrations.get('C21_b', 0) ** 2) *
                          wavelength *
                          np.sqrt(x ** 2 + y ** 2) ** 3 * np.cos(np.arctan2(y, x) -
                                                                 np.arctan2(aberrations.get('C21_b', 0),
                                                                            aberrations.get('C21_a', 0))) +
                          (2.0 / 3.0) *
                          np.sqrt(aberrations.get('C23_a', 0) ** 2 + aberrations.get('C23_b', 0) ** 2) *
                          wavelength *
                          np.sqrt(x ** 2 + y ** 2) ** 3 * np.cos(3 * (np.arctan2(y, x) -
                                                                      np.arctan2(aberrations.get('C23_b', 0),
                                                                                 aberrations.get('C23_a', 0))))) * \
                         np.pi * wavelength

            kernel = np.cos(raw_kernel) + 1j * np.sin(raw_kernel)
            aperture = np.zeros(kernel.shape)
            # Calculate size of X mrad aperture in k-space for Y eV electrons

            # "Apply" aperture
            center = tuple((np.array(kernel.shape) / 2).astype('int'))
            radius = int(np.rint(aperturesize))
            color = 1
            subarray = aperture[center[0] - radius:center[0] + radius + 1, center[1] - radius:center[1] + radius + 1]
            y_, x_ = np.mgrid[-radius:radius + 1, -radius:radius + 1]
            distances = np.sqrt(x_ ** 2 + y_ ** 2)
            if subarray.shape != distances.shape:
                print('ERROR: field of view too large?')
                return np.ones(self.shape) * np.nan
            subarray[distances <= radius] = color

            kernel *= aperture
            kernel_rs = kernel.copy()
            kernel = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(kernel)))) ** 2
            kernel /= np.sum(kernel)
            self.probe = kernel.copy()

    def simulate(self, potential, aberrations=None, blur=1, brightness=None, contrast=1, amplify=1, background=1):
        device = self.device  # z.B. "cuda" oder "cpu"
        dtype = torch.float32

        kernelpixel = int(self.shape[0] / self.kernelsize)

        wavelength = 1.23 / torch.sqrt(
            torch.tensor(self.energy, dtype=dtype, device=device) *
            (1 + 9.78e-7 * torch.tensor(self.energy, dtype=dtype, device=device))
        )

        aperturesize = (self.convergence_angle / self.kernelsize) * self.imsize / wavelength

        # Frequenzgitter
        frequencies = torch.fft.fftshift(
            torch.fft.fftfreq(kernelpixel, self.imsize / self.shape[0])
        ).to(device)

        x = frequencies.repeat(kernelpixel, 1)  # shape (kernelpixel, kernelpixel)
        y = frequencies.view(-1, 1).repeat(1, kernelpixel)

        # Hilfsfunktionen für Aberrationen
        atan2 = torch.atan2
        sqrt = torch.sqrt

        # raw_kernel wie im Original
        raw_kernel = (-aberrations['EHTFocus'] * (x ** 2 + y ** 2)
                             + sqrt(aberrations['C12_a'] ** 2 + aberrations['C12_b'] ** 2)
                             * (x ** 2 + y ** 2)
                             * torch.cos(
                         2 * (atan2(y, x) - atan2(aberrations['C12_b'], aberrations['C12_a']))
                     )
                             + (2.0 / 3.0)
                             * sqrt(aberrations['C21_a'] ** 2 + aberrations['C21_b'] ** 2)
                             * wavelength
                             * sqrt(x ** 2 + y ** 2) ** 3
                             * torch.cos(
                         atan2(y, x) - atan2(aberrations['C21_b'], aberrations['C21_a'])
                     )
                             + (2.0 / 3.0)
                             * sqrt(aberrations['C23_a'] ** 2 + aberrations['C23_b'] ** 2)
                             * wavelength
                             * sqrt(x ** 2 + y ** 2) ** 3
                             * torch.cos(
                         3 * (atan2(y, x) - atan2(aberrations['C23_b'], aberrations['C23_a']))
                     )
                     ) * math.pi * wavelength

        # komplexes Kernel
        kernel = torch.cos(raw_kernel) + 1j * torch.sin(raw_kernel)

        # Aperturmaske
        aperture = torch.zeros_like(kernel, dtype=dtype)
        center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
        radius = int(torch.round(aperturesize).item())

        y_, x_ = torch.meshgrid(
            torch.arange(-radius, radius+1, device=device),
            torch.arange(-radius, radius+1, device=device),
            indexing='ij'
        )
        distances = torch.sqrt(x_ ** 2 + y_ ** 2)
        mask = (distances <= radius).to(dtype)

        # ins Aperturarray einsetzen
        aperture[
        center[0] - radius:center[0] + radius + 1,
        center[1] - radius:center[1] + radius + 1
        ] = mask

        kernel *= aperture

        # Fourier-Rücktransformation
        kernel = torch.abs(
            torch.fft.fftshift(
                torch.fft.ifft2(torch.fft.fftshift(kernel))
            )
        ) ** 2

        # Normierung
        kernel /= kernel.sum()

        self.probe = kernel.clone()

        conv = fftconvolve_valid(potential, self.probe)
        blurred =conv# gaussian_filter_torch(conv, abs(blur))
        #blurred = torchvision.transforms.functional.gaussian_blur(conv, abs(blur))

        #brightness = 0#torch.mean(blurred)
        #contrast = torch.std(blurred)
        #blurred = (blurred - brightness) / contrast

        blurred = (blurred * amplify) + background


        return blurred[:self.shape[0], :self.shape[1]]

