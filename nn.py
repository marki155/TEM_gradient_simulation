from dataclasses import dataclass, asdict, fields, is_dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json


from loss_functions import *
from multislice import DifferentiableMultislice
from potential import DifferentiablePotential, TunablePotentialParameters

from ctf import CTFLayer, TunableCTFParameters


@dataclass
class ParticlesTunableParameters:
    potential_parameters: TunablePotentialParameters
    ctf_parameters: TunableCTFParameters
    normalisation_parameters: nn.Parameter

@dataclass
class ParticlesConfig:
    init_positions: List[np.ndarray]
    init_intensities: List[np.ndarray]
    shape: tuple
    pixel_size_nm: float
    wavelength_nm: float



def is_train_param_instance(value) -> bool:
    return is_dataclass(value) and hasattr(value, "enabled") and hasattr(value, "lr")

def _flatten_params(obj):
    out = []

    if obj is None:
        return out

    if isinstance(obj, nn.Parameter):
        return [obj]

    if isinstance(obj, nn.ParameterList):
        for p in obj:
            out.extend(_flatten_params(p))
        return out

    if isinstance(obj, nn.ParameterDict):
        for p in obj.values():
            out.extend(_flatten_params(p))
        return out

    if isinstance(obj, (list, tuple)):
        for item in obj:
            out.extend(_flatten_params(item))
        return out

    if is_dataclass(obj):
        for f in fields(obj):
            out.extend(_flatten_params(getattr(obj, f.name)))
        return out

    return out


def _safe_index_paramlist(param_list, idx: int):
    if param_list is None:
        return []
    if idx < 0 or idx >= len(param_list):
        return []
    return _flatten_params(param_list[idx])


def _safe_paramlist_slice(param_list, start: int, stop=None):
    if param_list is None:
        return []
    items = list(param_list)[start:stop]
    return _flatten_params(items)


def _collect_strain_params(displacement):
    return _flatten_params(displacement)

def get_trainmethod_param_registry(params: ParticlesTunableParameters):
    pot = params.potential_parameters
    ctf = params.ctf_parameters

    registry = {
        # Graphen geometry
        "graphen_shift": lambda: [pot.shift_Gr],
        "graphen_rotation": lambda: [pot.rotation_Gr],
        "graphen_scale": lambda: [pot.scale_Gr],

        # WS2 geometry
        "WS2_shift": lambda: [pot.shift_WS2],
        "WS2_rotation": lambda: [pot.rotation_WS2],
        "WS2_scale": lambda: [pot.scale_WS2],

        # Strain / free positions
        "Gr_strain_field": lambda: _collect_strain_params(pot.strain_fields[0]),
        "WS2_strain_field": lambda: _collect_strain_params(pot.strain_fields[1]),
        "graphen_free_positions": lambda: pot.position_shifts[0],
        "WS2_free_positions": lambda: pot.position_shifts[1:],

        # Intensities
        "train_graphen_intensities": lambda: _safe_index_paramlist(pot.intensities, 0),
        "train_S1_intensities": lambda: _safe_index_paramlist(pot.intensities, 1),
        "train_W_intensities": lambda: _safe_index_paramlist(pot.intensities, 2),
        "train_S2_intensities": lambda: _safe_index_paramlist(pot.intensities, 3),

        # Occupancies
        "train_graphen_occupancies": lambda: _safe_index_paramlist(pot.occupancies, 0),
        "train_S1_occupancies": lambda: _safe_index_paramlist(pot.occupancies, 1),
        "train_W_occupancies": lambda: _safe_index_paramlist(pot.occupancies, 2),
        "train_S2_occupancies": lambda: _safe_index_paramlist(pot.occupancies, 3),

        # CTF
        "train_amplitude_contrast": lambda: [ctf.amp_contrast],
        "train_ctf_aberrations": lambda: list(ctf.aberrations.values()),
        "train_env_parameters": lambda: [ctf.deltaE, ctf.Cc_um],

        # Normalisation
        "train_normalisation_parameters": lambda: [params.normalisation_parameters],
    }

    return registry




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
class ParticlesConfig:
    init_positions: List[np.ndarray]
    init_intensities: List[np.ndarray]
    z_positions_nm: List[float]
    shape: tuple
    pixel_size_nm: float
    wavelength_nm: float
    dz: float
    voltage_kV: float
    init_aberrations: dict
    init_ctf_env_params: dict
    init_amp_contrast: float
    ctf_alpha_max_mrad: float
    padding: int


class NN_particles(nn.Module):
    def __init__(self, image_shape, pixel_size_nm, voltage_kV, train_method, device='cpu', train_methods=None):
        super().__init__()
        self.shape = image_shape
        self.pixel_size_nm = pixel_size_nm
        self.voltage_kV = voltage_kV
        self.device = device
        self.train_method = train_method
        self.wavelength_nm = relativistic_wavelength_nm(voltage_kV)
        self.train_methods = train_methods

    def build_optimizer(self,
            optimizer_cls=torch.optim.Adam,
            **optimizer_kwargs,
    ):
        registry = get_trainmethod_param_registry(self.tunable_parameters)

        param_groups = []
        seen = set()

        for f in fields(self.train_method):
            name = f.name
            value = getattr(self.train_method, name)

            if not is_train_param_instance(value):
                continue



            if name not in registry:
                raise KeyError(f"Kein Registry-Eintrag für TrainMethod-Feld '{name}' gefunden.")

            params = registry[name]()
            #params = [p for p in params if p is not None and getattr(p, "requires_grad", True)]

            #unique_params = []
            #for p in params:
            #    pid = id(p)
            #    if pid in seen:
            #        continue
            #    seen.add(pid)
            #    unique_params.append(p)#

            #if not unique_params:
            #    continue

            param_groups.append({
                "name": name,
                "params": params,
                "lr": float(value.lr),
            })

        if not param_groups:
            raise ValueError("Keine aktiven Parametergruppen aus TrainMethod gefunden.")

        return optimizer_cls(param_groups, **optimizer_kwargs)



    def save_parameters(self):
        # Speichern
        p = self.tunable_parameters

        with open("parameters.json", "w") as f:
            json.dump(asdict(p), f)

    def load_parameters(self):
        with open("parameters.json", "r") as f:
            data = json.load(f)

        p_loaded = ParticlesTunableParameters(**data)
        print(p_loaded)

    def build_multislice(self, init_positions, init_intensities, dz, max_shifts=None, max_rotations=None):



        aberrations = {"C10": -13.,
                       'C12': 1.0, "phi12": 0.,
                       'C23': 2., 'phi23': 0.,
                       'C21': 2., 'phi21': 0.,
                       "C30": 4.,  # um
                       'C34': .02, 'phi34': 0.,  # um
                       'C45': .05, 'phi45': 0.}  # um

        env_params = {
            'source_size': 0.1e-3,  # rad
            'Cc_um': 1.e3,  # um
            'deltaE': .35,  # eV
            'instab_defocus_nm': .01,  # nm
            'use_detector': False,
        }

        amp_contrast = .1

        z_positions_nm = [
            0.00,  # Graphen
            0.33,  # S1
            0.488,  # W
            0.646,  # S2
        ]

        config = ParticlesConfig(init_positions=init_positions, init_intensities=init_intensities,
                                           shape=self.shape, pixel_size_nm=self.pixel_size_nm,
                                            z_positions_nm=z_positions_nm,
                                           wavelength_nm=self.wavelength_nm, dz=dz, voltage_kV=self.voltage_kV,
                                           init_aberrations=aberrations, init_ctf_env_params=env_params,
                                           init_amp_contrast=amp_contrast, ctf_alpha_max_mrad=50.,
                                            padding=100)
        potential_layer = DifferentiablePotential(config, self.train_method, device=self.device,
                                                  max_shifts=max_shifts, max_rotations=max_rotations)


        multislice = DifferentiableMultislice(
            config,
            train_method=self.train_method,
            device=self.device)



        ctf_layer = CTFLayer(config,
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

    def calc_error(self, simulation, image, mask=None):
        #error = torch.sum((torch.mean(simulation) - simulation) * (torch.mean(image) - image)) / \
        #             (torch.std(simulation) * torch.std(image) * (imageHeight * imageWidth - 1))


        #loss, P_diff = loss_fn(simulation.unsqueeze(0).unsqueeze(0),  # → (1, 1, H, W)
        #               image.unsqueeze(0).unsqueeze(0))

        #error = torch.corrcoef(torch.stack([simulation.flatten(), image.flatten()]))[0, 1]
        #diff_image = (simulation - image)

        loss_l1 = MaskedL1Loss()
        loss_mse = MaskedMSELoss()
        ncc_loss = NCCLoss()

        #local_l1_100 = LocalL1Loss(win_size=100)
        #local_mse_100 = LocalMSELoss(win_size=100)

        local_ncc_loss_100 = LocalNCCLoss(win_size=100)
        loss_var = GlobalDifferenceVarianceLoss(reduction="mean")
        local_loss_var = LocalDifferenceVarianceLoss(win_size=100, reduction="mean")

        loss_ssim_global = GlobalSSIMLoss(data_range=1.0)


        tt = image.unsqueeze(0).unsqueeze(0)
        sim = simulation.unsqueeze(0).unsqueeze(0)
        a_var = float(self.train_methods.loss_var)
        a_var_local_100 = float(self.train_methods.loss_local_var_100)
        a_mse = float(self.train_method.loss_mse)
        a_ncc = float(self.train_method.loss_ncc)
        a_local_ncc_100 = float(self.train_method.loss_local_ncc_100)
        a_ssim = float(self.train_method.loss_ssim)
        a_l1 = float(self.train_method.loss_l1)


        P_diff = None
        total_loss = 0.0
        if a_mse > 0:
             total_loss += a_mse * loss_mse(sim, tt, mask)
        if a_ncc > 0:
             total_loss += a_ncc * ncc_loss(sim, tt, mask)
        if a_local_ncc_100 > 0:
             total_loss += a_local_ncc_100 * local_ncc_loss_100(sim, tt, mask)
        if a_ssim > 0:
             total_loss += a_ssim * loss_ssim_global(sim, tt, mask)
        if a_l1 > 0:
             total_loss += a_l1 * loss_l1(sim, tt, mask)
        if a_var > 0:
             total_loss += a_var * loss_var(sim, tt, mask)
        if a_var_local_100 > 0:
             total_loss += a_var_local_100 * local_loss_var(sim, tt, mask)
        return  total_loss, P_diff

    def get_current_target_and_mask_img(self, target_image):
        train_area_idx = int(self.train_methods.only_train_small_area)
        if self.train_methods.target_gr is not None and train_area_idx == 0:
            ret_target = self.train_methods.target_gr
            mask_img = None
        else:
            if train_area_idx >= 0:
                mask_img = self.train_methods.area_masks[train_area_idx]
                ret_target = target_image
            else:
                mask_img = None
                ret_target = target_image
        return ret_target, mask_img

    def get_current_target_and_sim_ref(self, target_image, simulated_image):
        train_area_idx = int(self.train_methods.only_train_small_area)

        if train_area_idx >= 0:
            ref_idx = train_area_idx
        else:
            ref_idx = 0

        mask_img = self.train_methods.area_masks[ref_idx]
        sim_ref = simulated_image[mask_img > 0]
        tgt_ref = target_image[mask_img > 0]

        return tgt_ref, sim_ref

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
        tgt_ref, sim_ref = self.get_current_target_and_sim_ref(target_image, simulated_image)

        if self.train_methods.train_normalisation_parameters:
            norm_params = self.tunable_parameters.normalisation_parameters
        else:
            norm_params = self.tunable_parameters.normalisation_parameters.detach()


        if method == "z":
            sim_mean = norm_params[0] + torch.mean(sim_ref)
            sim_std = norm_params[1] + torch.std(sim_ref)
            tgt_mean = torch.mean(tgt_ref)
            tgt_std = torch.std(tgt_ref)
            simulated_image = (simulated_image - sim_mean) / (sim_std + eps)
            target_image = (target_image - tgt_mean) / (tgt_std + eps)

        elif method == "minmax":
            sim_min = norm_params[0]#torch.min(sim_ref)
            sim_max = norm_params[1]#torch.max(sim_ref)
            tgt_min = torch.min(tgt_ref)
            tgt_max = torch.max(tgt_ref)

            simulated_image = (simulated_image - sim_min) / (sim_max - sim_min + eps)
            target_image = (target_image - tgt_min) / (tgt_max - tgt_min + eps)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return simulated_image, target_image



    def loss_multislice(self, target_image):
        """Berechnet den Unterschied zwischen Simulation und Zielbild."""
        #intensities = torch.pow(self.intensities, self.contrast)
        target_image, mask_img = self.get_current_target_and_mask_img(target_image)

        simulated_image, aux_loss = self.model(None)
        simulated_image, target_image = self.normalize_images(simulated_image, target_image, method="z")

        rec_error, P_diff = self.calc_error(simulated_image, target_image, mask=mask_img)

        if mask_img is not None:
            simulated_image_masked = simulated_image * mask_img
            target_image_masked = target_image * mask_img
        else:
            simulated_image_masked = simulated_image
            target_image_masked = target_image

        return simulated_image_masked, target_image_masked, P_diff, rec_error + aux_loss
