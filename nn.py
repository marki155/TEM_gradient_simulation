import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import math

from multislice import DifferentiableMultislice_Free, DifferentiableMultislice
from potential import DifferentiablePotential
from simulation import Simulation
from torch.autograd import gradcheck
from abtem.waves import PlaneWave

from ctf import CTFLayer
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

class NN_particles(nn.Module):
    def __init__(self, image_shape, fov, voltage_kV, train_method, device='cpu', train_small_area_square=None):
        super().__init__()
        self.shape = image_shape
        self.fov = fov
        self.pixel_size_nm = fov / image_shape[0]
        self.voltage_kV = voltage_kV
        self.device = device
        self.train_method = train_method
        self.wavelength_nm = relativistic_wavelength_nm(voltage_kV)
        self.train_small_area_square = train_small_area_square


    def build_multislice(self, init_positions, init_intensities, dz, max_shifts=None, max_rotations=None):
        H = self.shape[0]
        W = self.shape[1]

        potential_layer = DifferentiablePotential(self.train_method, init_positions, init_intensities,self.shape, self.pixel_size_nm,
                                                  self.wavelength_nm, device=self.device,
                                                  max_shifts=max_shifts, max_rotations=max_rotations,
                                                  train_small_area_square=self.train_small_area_square)


        multislice = DifferentiableMultislice(
            dz=dz,
            shape=self.shape,
            pixel_size_nm=self.pixel_size_nm,
            wavelength_nm=self.wavelength_nm,
            device=self.device,
            train_method=self.train_method
        )

        aberrations = {'C10': -57., 'C12': 1.0, 'C23': 3.0e3,
                       'C21': 50.0e3, 'C30': 5.0e3, 'C34': 20.0e3, 'C32': 50.0e3}

        aberrations = {'C10': -57., 'C30': 2.}

        aberrations = {"C10": -15.,
                       'C12': 1.0, "phi12": 0.,
                       'C23': 5., 'phi23': 0.,
                       'C21': 5., 'phi21': 0.,
                       "C30": 4., #um
                       'C34': .02, 'phi34': 0., #um
                       'C45': .05, 'phi45': 0.} #um

        #aberrations = {'C10': -7., 'C30': 5.}



        env_params = {
            'source_size': 0.1e-3,  # rad
            'Cc_um': 1.e3 ,  # um
            'deltaE': .35 ,  # eV
            'instab_defocus_nm': .01,  # nm
            'use_detector': False,
        }

        amp_contrast = .5
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


    def loss(self, target_image, brightness=None, constrast=None):

        intensities = torch.pow(self.intensities, self.contrast)
        potential = self.simulation.calc_potential(self.positions, torch.abs(intensities))
        simulated_image = self.simulation.simulate(potential, self.aberrations, blur=self.blur)

        minimum_sim_image = torch.min(simulated_image)
        maximum_sim_image = torch.max(simulated_image)
        brightness_sim_image = torch.mean(simulated_image)
        contrast_sim_image = torch.std(simulated_image)
        # Normalisierung der Simulation
        simulated_image = (simulated_image - brightness_sim_image) / contrast_sim_image
        #simulated_image = (simulated_image - minimum_sim_image) / (maximum_sim_image - minimum_sim_image)

        if self.train_methods.only_train_small_area:
            # train_small_area_square = [(500, 600), (1000, 1000)]
            x_min, y_min = self.train_small_area_square[0]
            x_max, y_max = self.train_small_area_square[1]
            simulated_image = simulated_image[x_min:x_max,y_min:y_max]
            target_image = target_image[x_min:x_max, y_min:y_max]

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



        #error = torch.corrcoef(torch.stack([simulation.flatten(), image.flatten()]))[0, 1]
        return - error + F.mse_loss(simulation, image)

    def loss_multislice(self, target_image, brightness=None, contrast=None):
        """Berechnet den Unterschied zwischen Simulation und Zielbild."""
        #intensities = torch.pow(self.intensities, self.contrast)

        simulated_image, aux_loss = self.model(None)

        #simulated_image = simulated_image[:target_image.shape[0], :target_image.shape[1]]
        minimum_sim_image = torch.min(simulated_image)
        maximum_sim_image = torch.max(simulated_image)
        simulated_image = (simulated_image - minimum_sim_image) / (maximum_sim_image - minimum_sim_image)
        #simulated_image = (simulated_image - brightness_sim_image) / contrast_sim_image

        rec_error = self.calc_error(simulated_image, target_image, target_image.shape[0], target_image.shape[1])

        return simulated_image, rec_error + aux_loss
