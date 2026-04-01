from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import math


from losses import RadialPowerSpectrumLoss, NCCLoss, SSIMLoss, LocalNCCLoss, FFTBraggLoss
from multislice import DifferentiableMultislice
from potential import DifferentiablePotential, TunablePotentialParameters
from simulation import Simulation

from ctf import CTFLayer, TunableCTFParameters
from torch.nn.functional import max_pool2d







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


class NN_free_potential(nn.Module):
    def __init__(self, init_potential, device='cpu'):
        super().__init__()
        self.potential = nn.Parameter(init_potential.to(device))  # Potential als Parameter
        self.blur = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
        self.contrast = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=device))
        self.background = nn.Parameter(torch.tensor(1, dtype=torch.float32, device=device))
        self.amplify = nn.Parameter(torch.tensor(1, dtype=torch.float32, device=device))



    def build_multislice(self, num_slices, shape, sigma, dz, pixel_size_nm, voltage_kV, device):
        H = self.potential.shape[0]
        W = self.potential.shape[1]


        multislice = DifferentiableMultislice_Free(
            potential=self.potential,
            dz=dz,
            shape=shape,
            pixel_size_nm=pixel_size_nm,
            wavelength_nm=relativistic_wavelength_nm(voltage_kV),
            device=device
        )

        aberrations = {'C10': -57., 'C12': 1.0, 'C23': 3.0e3,
                       'C21': 50.0e3, 'C30': 5.0e3, 'C34': 20.0e3, 'C32': 50.0e3}

        aberrations = {'C10': -57., 'C30': 2.}

        aberrations = {"C10": -57.,
                       'C12': 1.0, "phi12": 0.,
                       'C23': .5, 'phi23': 0.,
                       'C21': .5, 'phi21': 0.,
                       "C30": 2.,
                       'C34': 20., 'phi34': 0.,
                       'C45': 50., 'phi45': 0.}

        # ctf = CTFLayer(H, W, pixel_size, voltage_kV, amp_contrast=1,
        #               aberrations=aberrations, device=device)

        Cc = 1.3e6  # torch.nn.Parameter(torch.tensor(2.0e7, dtype=torch.float32, device=device))  # Beispielwert für Cc in nm
        # source_size = torch.nn.Parameter(torch.tensor(0.1e-3, dtype=torch.float32, device=device))  # Beispielwert für die Quellgröße in m
        delta_E = torch.nn.Parameter(
            torch.tensor(3., dtype=torch.float32, device=device))  # Beispielwert für die Energieauflösung in eV

        env_params = {
            'source_size': 0.1e-3,  # rad
            'Cc': Cc,  # nm
            'deltaE': delta_E,  # eV
            'instab_defocus': 0.5,  # nm
            'use_detector': False,
        }

        # self.df_nm = nn.Parameter(torch.tensor(-57., dtype=torch.float32, device=device))
        # self.Cs_nm = nn.Parameter(torch.tensor(5e3, dtype=torch.float32, device=device))
        amp_contrast = .5
        ctf_layer = CTFLayer(H, W, pixel_size_nm, voltage_kV, amp_contrast, aberrations=aberrations,
                             envelope_params=env_params)

        self.model = nn.Sequential(
            multislice,
            ctf_layer)

    def loss(self, target_image, brightness=None, constrast=None):
        """Berechnet den Unterschied zwischen Simulation und Zielbild."""
        simulated_image = self.simulation.simulate(self.potential, self.aberrations, blur=self.blur,
                                                    brightness=brightness,
                                                   contrast=constrast, background=self.background, amplify=self.amplify)

    def clamp_parameters(self):
        self.model[1].aberr_params['C10'] = torch.clamp(self.model[1].aberr_params['C10'], -100, 0)
        self.model[1].aberr_params['C12'] = torch.clamp(self.model[1].aberr_params['C12'], 0, 10)
        self.model[1].aberr_params['C23'] = torch.clamp(self.model[1].aberr_params['C23'], -100, 100)
        self.model[1].aberr_params['C21'] = torch.clamp(self.model[1].aberr_params['C21'], -100, 100)
        self.model[1].aberr_params['C30'] = torch.clamp(self.model[1].aberr_params['C30'], -1000, 1000)
        self.model[1].aberr_params['C34'] = torch.clamp(self.model[1].aberr_params['C34'], -1000, 1000)
        self.model[1].aberr_params['C45'] = torch.clamp(self.model[1].aberr_params['C45'], -1000, 1000)

    def laplacian(self):
        V_pad = F.pad(self.potential.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
        lap = V_pad[:, :, :-2, 1:-1] + V_pad[:, :, 2:, 1:-1] + \
              V_pad[:, :, 1:-1, :-2] + V_pad[:, :, 1:-1, 2:] - 4 * self.potential
        return lap



    def calc_error(self, simulation, image, imageHeight, imageWidth):
        error = torch.sum((torch.mean(simulation) - simulation) * (torch.mean(image) - image)) / \
                (torch.std(simulation) * torch.std(image) * (imageHeight * imageWidth - 1))
        energy = torch.sum(self.potential ** 2)
        # error = torch.corrcoef(torch.stack([simulation.flatten(), image.flatten()]))[0, 1]

        sparsity_loss = torch.sum(torch.abs(self.potential))




        # Smoothness: L2 auf Gradienten verhindert Rauschen
        grad_x = self.potential[:, 1:] - self.potential[:, :-1]
        grad_y = self.potential[1:, :] - self.potential[:-1, :]
        smoothness_loss = torch.sum(grad_x ** 2) + torch.sum(grad_y ** 2)

        #lap = torch.sum(self.laplacian())

        return - error + 0.00 * energy + 0.01 * sparsity_loss + 0.1 * smoothness_loss

    def loss_multislice(self, target_image, brightness=None, contrast=None):


        image_wave = self.model(None)
        simulated_image = torch.abs(image_wave)
        simulated_image = simulated_image[:target_image.shape[0], :target_image.shape[1]]
        brightness_sim_image = torch.mean(simulated_image)
        contrast_sim_image = torch.std(simulated_image)
        minimum_sim_image = torch.min(simulated_image)
        maximum_sim_image = torch.max(simulated_image)
        simulated_image = (simulated_image - minimum_sim_image) / (maximum_sim_image - minimum_sim_image)
        # simulated_image = (simulated_image - brightness_sim_image) / contrast_sim_image
        return simulated_image, self.calc_error(simulated_image, target_image, target_image.shape[0],
                                                target_image.shape[1])
        return simulated_image, F.mse_loss(simulated_image, target_image)


@dataclass
class ParticlesTunableParameters:
    potential_parameters: TunablePotentialParameters
    ctf_parameters: TunableCTFParameters
    normalisation_parameters: nn.Parameter

class NN_particles(nn.Module):
    def __init__(self, image_shape, fov, voltage_kV, train_method, device='cpu', train_methods=None):
        super().__init__()
        self.shape = image_shape
        self.fov = fov
        self.pixel_size_nm = fov / image_shape[0]
        self.voltage_kV = voltage_kV
        self.device = device
        self.train_method = train_method
        self.wavelength_nm = relativistic_wavelength_nm(voltage_kV)
        self.train_methods = train_methods





    def build_multislice(self, init_positions, init_intensities, dz, max_shifts=None, max_rotations=None):
        H = self.shape[0]
        W = self.shape[1]

        potential_layer = DifferentiablePotential(self.train_method, init_positions, init_intensities,self.shape, self.pixel_size_nm,
                                                  self.wavelength_nm, device=self.device,
                                                  max_shifts=max_shifts, max_rotations=max_rotations)


        multislice = DifferentiableMultislice(
            dz=dz,
            shape=self.shape,
            pixel_size_nm=self.pixel_size_nm,
            wavelength_nm=self.wavelength_nm,
            device=self.device,
            train_method=self.train_method,
            voltage_kV=self.voltage_kV
        )

        aberrations = {'C10': -57., 'C12': 1.0, 'C23': 3.0e3,
                       'C21': 50.0e3, 'C30': 5.0e3, 'C34': 20.0e3, 'C32': 50.0e3}



        aberrations = {"C10": -13.,
                       'C12': 1.0, "phi12": 0.,
                       'C23': 2., 'phi23': 0.,
                       'C21': 2., 'phi21': 0.,
                       "C30": 4., #um
                       'C34': .02, 'phi34': 0., #um
                       'C45': .05, 'phi45': 0.} #um

        #aberrations = {'C10': -7., 'C30': 5.}
        #aberrations = {'C10': -13., 'C30': 2.}



        env_params = {
            'source_size': 0.1e-3,  # rad
            'Cc_um': 1.e3 ,  # um
            'deltaE': .35 ,  # eV
            'instab_defocus_nm': .01,  # nm
            'use_detector': False,
        }

        amp_contrast = .1
        ctf_layer = CTFLayer(H, W, self.pixel_size_nm, self.voltage_kV, self.wavelength_nm,
                            amp_contrast,
                             aberrations=aberrations,
                             envelope_params=env_params,
                             device=self.device,
                             train_method=self.train_method)

        self.model = nn.Sequential(
            potential_layer,
            multislice,
            ctf_layer)

        self.tunable_parameters = ParticlesTunableParameters(
            potential_parameters=self.model[0].get_tunable_parameters(),
            ctf_parameters=self.model[2].get_tunable_parameters(),
            normalisation_parameters = nn.Parameter(torch.tensor([0.0,0.0]), requires_grad=True)
        )

        for name, value in vars(self.tunable_parameters).items():
            if isinstance(value, (nn.Parameter, nn.ParameterList, nn.ParameterDict)):
                setattr(self, name, value)

    def get_tunable_parameters(self):
        return self.tunable_parameters


    def loss_old(self, target_image, brightness=None, constrast=None):

        intensities = torch.pow(self.intensities, self.contrast)
        potential = self.simulation.calc_potential(self.positions, torch.abs(intensities))
        simulated_image = self.simulation.simulate(potential, self.aberrations, blur=self.blur)



        if self.train_methods.only_train_small_area >= 0:
            # train_small_area_square = [(500, 600), (1000, 1000)]
            x_min, y_min, x_max, y_max = self.train_small_area_square[self.train_methods.only_train_small_area]
            simulated_image = simulated_image[x_min:x_max,y_min:y_max]
            target_image = target_image[x_min:x_max, y_min:y_max]

        minimum_sim_image = torch.min(simulated_image)
        maximum_sim_image = torch.max(simulated_image)
        brightness_sim_image = torch.mean(simulated_image)
        contrast_sim_image = torch.std(simulated_image)
        # Normalisierung der Simulation
        simulated_image = (simulated_image - brightness_sim_image) / contrast_sim_image
        # simulated_image = (simulated_image - minimum_sim_image) / (maximum_sim_image - minimum_sim_image)

        brightness_tgt_image = torch.mean(target_image)
        contrast_tgt_image = torch.std(target_image)
        # Normalisierung des Zielbildes
        target_image = (target_image - brightness_tgt_image) / contrast_tgt_image

        return simulated_image, F.mse_loss(simulated_image, target_image)

    def clamp_parameters(self):
        self.model[1].aberr_params['C10'] = torch.clamp(self.model[1].aberr_params['C10'], -100, 0)
        self.model[1].aberr_params['C12'] = torch.clamp(self.model[1].aberr_params['C12'], 0, 10)
        self.model[1].aberr_params['C23'] = torch.clamp(self.model[1].aberr_params['C23'], -100, 100)
        self.model[1].aberr_params['C21'] = torch.clamp(self.model[1].aberr_params['C21'], -100, 100)
        self.model[1].aberr_params['C30'] = torch.clamp(self.model[1].aberr_params['C30'], -1000, 1000)
        self.model[1].aberr_params['C34'] = torch.clamp(self.model[1].aberr_params['C34'], -1000, 1000)
        self.model[1].aberr_params['C45'] = torch.clamp(self.model[1].aberr_params['C45'], -1000, 1000)

        #self.intensities = torch.clamp(self.intensities, 0.00, 1000.0)



    def calc_energy(self):
        pos = self.positions
        q = torch.abs(self.intensities)
        epsilon = 1e-3

        # paarweise Differenzen
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # shape: (1, N, N, 2)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + epsilon ** 2)  # shape: (1, N, N)

        # Energie: nur i<j
        mask = 1 - torch.eye(1000).unsqueeze(0)
        energy = torch.sum(q.unsqueeze(1) * q.unsqueeze(0) / dist * mask) / 2  # Faktor 1/2 für i<j

        return energy

    def calc_error(self, simulation, image, imageHeight, imageWidth):
        error = torch.sum((torch.mean(simulation) - simulation) * (torch.mean(image) - image)) / \
                     (torch.std(simulation) * torch.std(image) * (imageHeight * imageWidth - 1))

        loss_fft = FFTBraggLoss()
        #loss, P_diff = loss_fn(simulation.unsqueeze(0).unsqueeze(0),  # → (1, 1, H, W)
        #               image.unsqueeze(0).unsqueeze(0))

        error = torch.corrcoef(torch.stack([simulation.flatten(), image.flatten()]))[0, 1]
        diff_image = (simulation - image)

        ncc_loss = NCCLoss()
        ssim_loss = SSIMLoss(data_range=1.0)
        local_ncc_loss_100 = LocalNCCLoss(window_size=100)
        local_ncc_loss_50 = LocalNCCLoss(window_size=50)
        local_ncc_loss_20 = LocalNCCLoss(window_size=20)
        loss_l1 = nn.L1Loss()
        loss_MSE = nn.MSELoss()
        tt = image.unsqueeze(0).unsqueeze(0)
        sim = simulation.unsqueeze(0).unsqueeze(0)
        a_mse = float(self.train_method.loss_mse)
        a_ncc = float(self.train_method.loss_ncc)
        a_local_ncc_100 = float(self.train_method.loss_local_ncc_100)
        a_local_ncc_50 = float(self.train_method.loss_local_ncc_50)
        a_local_ncc_20 = float(self.train_method.loss_local_ncc_20)
        a_ssim = float(self.train_method.loss_ssim)
        a_l1 = float(self.train_method.loss_l1)
        a_fft = float(self.train_method.loss_fft)

        P_diff = None
        total_loss = 0.0
        if a_mse > 0:
             total_loss += a_mse * loss_MSE(sim, tt)
        if a_ncc > 0:
             total_loss += a_ncc * ncc_loss(sim, tt)
        if a_local_ncc_100 > 0:
             total_loss += a_local_ncc_100 * local_ncc_loss_100(sim, tt)
        if a_local_ncc_50 > 0:
             total_loss += a_local_ncc_50 * local_ncc_loss_50(sim, tt)
        if a_local_ncc_20 > 0:
             total_loss += a_local_ncc_20 * local_ncc_loss_20(sim, tt)
        if a_ssim > 0:
             total_loss += a_ssim * ssim_loss(sim, tt)
        if a_l1 > 0:
             total_loss += a_l1 * loss_l1(sim, tt)
        if a_fft > 0:
            _loss_fft, P_diff = loss_fft(sim, tt)
            total_loss += a_fft * _loss_fft
        return  total_loss, P_diff

    def normalize_images(
            self,
            simulated_image: torch.Tensor,
            target_image: torch.Tensor,
            method: str = "z",  # "z" oder "minmax"
    ):
        """
        Berechnet die Normalisierungsstatistik auf einem Referenz-Ausschnitt
        und wendet sie anschließend auf das gesamte Bild an.

        Regeln:
        - train_area_idx >= 0:
            Statistik basiert auf train_small_area_squares[train_area_idx]
        - train_area_idx < 0:
            Statistik basiert auf train_small_area_squares[0]
        """

        eps = 1e-8
        train_area_idx = int(self.train_methods.only_train_small_area)

        if train_area_idx >= 0:
            ref_idx = train_area_idx
        else:
            ref_idx = 0

        x_min, y_min, x_max, y_max = self.train_methods.train_small_area_squares[ref_idx]

        # Referenzbereiche nur für Statistik
        sim_ref = simulated_image[x_min:x_max, y_min:y_max]
        tgt_ref = target_image[x_min:x_max, y_min:y_max]

        if method == "z":
            sim_mean = self.tunable_parameters.normalisation_parameters[0] + torch.mean(sim_ref)
            sim_std = self.tunable_parameters.normalisation_parameters[1] + torch.std(sim_ref)
            tgt_mean = torch.mean(tgt_ref)
            tgt_std = torch.std(tgt_ref)
            print(tgt_mean, tgt_std)
            simulated_image = (simulated_image - sim_mean) / (sim_std + eps)
            target_image = (target_image - tgt_mean) / (tgt_std + eps)

        elif method == "minmax":
            sim_min = self.tunable_parameters.normalisation_parameters[0]#torch.min(sim_ref)
            sim_max = self.tunable_parameters.normalisation_parameters[1]#torch.max(sim_ref)
            tgt_min = torch.min(tgt_ref)
            tgt_max = torch.max(tgt_ref)

            simulated_image = (simulated_image - sim_min) / (sim_max - sim_min + eps)
            target_image = (target_image - tgt_min) / (tgt_max - tgt_min + eps)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return simulated_image, target_image

    def loss_multislice(self, target_image, brightness=None, contrast=None):
        """Berechnet den Unterschied zwischen Simulation und Zielbild."""
        #intensities = torch.pow(self.intensities, self.contrast)

        simulated_image, aux_loss = self.model(None)

        simulated_image, target_image= self.normalize_images(simulated_image, target_image, method="z")

        train_area_idx = int(self.train_methods.only_train_small_area)

        if train_area_idx >= 0:
            # train_small_area_square = [(500, 600), (1000, 1000)]
            x_min, y_min, x_max, y_max = self.train_methods.train_small_area_squares[train_area_idx]
            simulated_image = simulated_image[x_min:x_max,y_min:y_max]
            target_image = target_image[x_min:x_max, y_min:y_max]


        rec_error, P_diff = self.calc_error(simulated_image, target_image, target_image.shape[0], target_image.shape[1])

        return simulated_image, target_image, P_diff, rec_error + aux_loss
