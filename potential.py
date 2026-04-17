from dataclasses import dataclass, field
from typing import List, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt

from displacement_field import SplineDisplacementField2D, StrainTunableParameters

from torch.nn.utils.rnn import pad_sequence

from kirkland_supersampled import KirklandPrecalcPotential
from lobato_potential import projected_atomic_potential_kernel_lobato
from physics_helper import relativistic_wavelength_nm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)

PROJECTED_POTENTIAL_PARAMS = {
    "C": [
        (18.2, 0.020),
        (7.6,  0.055),
        (2.1,  0.120),
    ],
    "S": [
        (55.0, 0.018),
        (22.0, 0.050),
        (6.5,  0.120),
    ],
    "W": [
        (210.0, 0.015),
        (85.0,  0.040),
        (25.0,  0.100),
    ],
}


PROJECTED_POTENTIAL_PARAMS = {
    "C": [
        (0.020, 0.020),
        (0.030, 0.055),
        (0.020, 0.120),
    ],
    "S": [
        (0.050, 0.018),
        (0.070, 0.050),
        (0.050, 0.120),
    ],
    "W": [
        (0.120, 0.015),
        (0.180, 0.040),
        (0.120, 0.100),
    ],
}

PROJECTED_POTENTIAL_PARAMS = {
    "C": [
        (0.0008, 0.020),
        (0.0012, 0.050),
        (0.0008, 0.120),
    ],
    "S": [
        (0.0015, 0.018),
        (0.0020, 0.050),
        (0.0015, 0.120),
    ],
    "W": [
        (0.0008, 0.015),
        (0.0012, 0.040),
        (0.0008, 0.100),
    ],
}

DOYLE_TURNER_PARAMS = {
    'C': [(0.489918, 0.206593), (0.262003, 0.0774039), (0.196767, 0.495519)],
    'W': [(0.489918, 0.35384), (0.262003, 0.32576), (0.196767, 0.1488)],
    'S': [(0.489918, 0.02051), (0.262003, 0.01164)]
}

DOYLE_TURNER_PARAMS = {
    "C": [(0.489918, 20.6593), (0.262003, 7.74039), (0.196767, 49.5519)],
    "W": [(0.489918, 35.384), (0.262003, 32.576), (0.196767, 14.880)],
    "S": [(0.489918, 2.051), (0.262003, 1.164)]
}

DOYLE_TURNER_PARAMS_NM = {
    "C": [
        (0.489918, 0.00206593),
        (0.262003, 0.000774039),
        (0.196767, 0.00495519),
    ],
    "W": [
        (0.489918, 0.00353840),
        (0.262003, 0.00325760),
        (0.196767, 0.00148800),
    ],
    "S": [
        (0.489918, 0.00020510),
        (0.262003, 0.00011640),
    ],
}


DOYLE_TURNER_PARAMS = {
    "C": [
        (2980.009564, 19109.271662),
        (4253.567068, 51003.137574),
        (499.001460,   7967.084532),
    ],
    "W": [
        (1739.908195, 11157.138143),
        (1010.690938, 12118.865915),
        (1661.725166, 26531.194627),
    ],
    "S": [
        (30017.021739, 192483.752337),
        (28285.453609, 339161.663268),
    ],
}

@dataclass
class TunablePotentialParameters:
    strain_fields: List[StrainTunableParameters]

    position_shifts: nn.ParameterList
    intensities: nn.ParameterList
    occupancies: nn.ParameterList

    scale_Gr: nn.Parameter = nn.Parameter(torch.tensor([1.12, 1.14], dtype=torch.float32))
    rotation_Gr: nn.Parameter = nn.Parameter(torch.tensor(0.025, dtype=torch.float32))
    shift_Gr: nn.Parameter = nn.Parameter(torch.tensor([-9.5, -11.0], dtype=torch.float32))

    scale_WS2: nn.Parameter = nn.Parameter(torch.tensor([1.05, 1.09 ], dtype=torch.float32))
    rotation_WS2: nn.Parameter = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    shift_WS2: nn.Parameter = nn.Parameter(torch.tensor([15.9, -5.0 ], dtype=torch.float32))

    graphene_potential_factor: nn.Parameter = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    ws2_potential_factor: nn.Parameter = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))







class DifferentiablePotential(nn.Module):
    def __init__(self, config, train_methods, device,
                 max_shifts=None, max_rotations=None):
        super().__init__()
        #self.slices_V = [v.to(device) for v in slices_V]  # Liste komplexer Slices (H,W)
        self.pixel_size_nm = config.pixel_size_nm
        self.wavelength_nm = relativistic_wavelength_nm(config.voltage_kV)
        self.device = device
        self.max_shifts = max_shifts
        self.max_rotations = max_rotations

        self.H, self.W = config.shape

        self._IDX = {"Gr": 0, "S1": 1, "W": 2, "S2": 3}
        self.atom_names_by_layer = ["C", "S", "W", "S"]
        self.N_l = len(config.init_positions)

        self.train_methods = train_methods
        self.config = config



        self.strain_fields = [SplineDisplacementField2D(config=config, control_shape=config.control_points_graphen,
                                                         device=device, dtype=torch.float32, init_scale=0.0),
                         SplineDisplacementField2D(config=config, control_shape=config.control_points_ws2,
                                                         device=device, dtype=torch.float32, init_scale=0.0)]


        #self.positions_method = "graphen_shift"
        #self.intensity_classes = "free" # "free", "vacancies", "classes"
        strain_parameters = [field.get_tunable_parameters() for field in self.strain_fields]
        position_shifts_ = self.initialize_position_shifts_parameters(config.init_positions)
        intensities_, occupancies_ = self.initialize_intensities_parameters(config.init_intensities)

        self.tunable_parameters = TunablePotentialParameters(strain_fields=strain_parameters, position_shifts=position_shifts_,
                                                             intensities=intensities_, occupancies=occupancies_,)

        self.potential_pad = config.padding
        self.kirkland_precalc_pot = KirklandPrecalcPotential(self.H + 2 * self.potential_pad,
                                                             self.W + 2 * self.potential_pad,
                                                             self.pixel_size_nm, self.device)



        self.apply_trainable_selection()
        #self.zero_grads_for_frozen()

        for name, value in vars(self.tunable_parameters).items():
            if isinstance(value, (nn.Parameter, nn.ParameterList, nn.ParameterDict)):
                setattr(self, name, value)

    def get_tunable_parameters(self):
        return self.tunable_parameters


    def apply_trainable_selection(self):
            self.tunable_parameters.intensities[0].requires_grad = self.train_methods.train_graphen_intensities.enabled
            self.tunable_parameters.intensities[1].requires_grad = self.train_methods.train_S1_intensities.enabled
            self.tunable_parameters.intensities[2].requires_grad = self.train_methods.train_W_intensities.enabled
            self.tunable_parameters.intensities[3].requires_grad = self.train_methods.train_S2_intensities.enabled

            self.tunable_parameters.occupancies[0].requires_grad = self.train_methods.train_graphen_occupancies.enabled
            self.tunable_parameters.occupancies[1].requires_grad = self.train_methods.train_S1_occupancies.enabled
            self.tunable_parameters.occupancies[2].requires_grad = self.train_methods.train_W_occupancies.enabled
            self.tunable_parameters.occupancies[3].requires_grad = self.train_methods.train_S2_occupancies.enabled

            self.tunable_parameters.shift_Gr.requires_grad = self.train_methods.graphen_shift.enabled
            self.tunable_parameters.rotation_Gr.requires_grad = self.train_methods.graphen_rotation.enabled
            self.tunable_parameters.position_shifts[self._IDX["Gr"]].requires_grad = self.train_methods.graphen_free_positions.enabled

            self.tunable_parameters.shift_WS2.requires_grad = self.train_methods.WS2_shift.enabled
            self.tunable_parameters.rotation_WS2.requires_grad = self.train_methods.WS2_rotation.enabled
            for i in range(self.N_l):
                if i != self._IDX["Gr"]:
                    self.tunable_parameters.position_shifts[i].requires_grad = self.train_methods.WS2_free_positions.enabled



            for p in self.strain_fields[0].parameters():
                p.requires_grad = self.train_methods.Gr_strain_field.enabled

            for p in self.strain_fields[1].parameters():
                p.requires_grad = self.train_methods.WS2_strain_field.enabled



    def zero_grads_for_frozen(self):
        if not self.train_methods.train_graphen_intensities.enabled:
            self.intensities[0].grad = None
        if not self.train_methods.train_S1_intensities.enabled:
            self.intensities[1].grad = None
        if not self.train_methods.train_W_intensities.enabled:
            self.intensities[2].grad = None
        if not self.train_methods.train_S2_intensities.enabled:
            self.intensities[3].grad = None

        if not  self.train_methods.train_graphen_occupancies.enabled:
            self.occupancies[0].grad = None
        if not self.train_methods.train_S1_occupancies.enabled:
            self.occupancies[1].grad = None
        if not self.train_methods.train_W_occupancies.enabled:
            self.occupancies[2].grad = None
        if not self.train_methods.train_S2_occupancies.enabled:
            self.occupancies[3].grad = None

        if not self.train_methods.graphen_shift.enabled:
            self.tunable_parameters.shift_Gr.grad = None
        if not self.train_methods.graphen_rotation.enabled:
            self.tunable_parameters.rotation_Gr.grad = None
        if not self.train_methods.graphen_free_positions.enabled:
            self.position_shifts[self._IDX["Gr"]].grad = None

        if not self.train_methods.WS2_shift.enabled:
            self.tunable_parameters.shift_WS2.grad = None
        if not self.train_methods.WS2_rotation.enabled:
            self.tunable_parameters.rotation_WS2.grad = None
        for i in range(self.N_l):
            if i != self._IDX["Gr"] and not self.train_methods.WS2_free_positions.enabled:
                self.position_shifts[i].grad = None

    def get_init_positions(self, i):
        return getattr(self, f"init_positions_{i}")


    def initialize_position_shifts_parameters(self, init_positions):
        positions_ = [torch.tensor(pos, dtype=torch.float32, device=self.device) for pos in init_positions]
        for i, t in enumerate(positions_):
            self.register_buffer(f"init_positions_{i}", t)

        position_shifts_ = nn.ParameterList(
            [torch.tensor(pos*0.0, dtype=torch.float32, device=self.device) for pos in init_positions])

        position_shifts_[self._IDX["Gr"]].requires_grad = self.train_methods.graphen_free_positions.enabled
        for i in range(self.N_l):
            if i != self._IDX["Gr"]:
                position_shifts_[i].requires_grad = self.train_methods.WS2_free_positions.enabled

        return position_shifts_






    def initialize_intensities_parameters(self, init_intensities):
        intensities_ = nn.ParameterList(
            [torch.tensor(I, dtype=torch.float32, device=self.device) for I in init_intensities])
        occupancies_ = nn.ParameterList(
            [torch.ones(init_intensities[0].shape, dtype=torch.float32, device=self.device)*100,
             torch.ones(init_intensities[1].shape, dtype=torch.float32, device=self.device)*2,
             torch.ones(init_intensities[2].shape, dtype=torch.float32, device=self.device)*2,
             torch.ones(init_intensities[3].shape, dtype=torch.float32, device=self.device)*2])


        return intensities_, occupancies_


        if False:

            if self.train_methods.intensity_classes == "free":
                for i in range(len(init_intensities)):
                    self.intensities_[i].requires_grad = True
                    self.occupancies_[i].requires_grad = True
            elif self.train_methods.intensity_classes == "free_ws2":
                for n in self._IDX:
                    if n == "Gr":
                        self.intensities_[self._IDX[n]].requires_grad = False
                    else:
                        self.intensities_[self._IDX[n]].requires_grad = True

            elif self.train_methods.intensity_classes == "vacancies":
                raise NotImplementedError("Vacancies not implemented yet")
                self.vacancies_ = nn.ParameterList(
                    [torch.ones(init_intensities[i].shape, dtype=torch.float32, device=self.device)
                     for i in range(len(init_intensities))])

            elif self.train_methods.intensity_classes == "vacancies_WS2":
                raise NotImplementedError("Vacancies not implemented yet")
                self.vacancies_ = nn.ParameterList(
                    [torch.ones(init_intensities[i].shape, dtype=torch.float32, device=self.device)
                     for i in range(len(init_intensities))])

                self.vacancies_[self._IDX["Gr"]].requires_grad = False


            elif self.train_methods.intensity_classes == "classes": ## eher über loss funktion (abweichung von 16, 32, 74)
                raise NotImplementedError("Classes not implemented yet")
                self.classes = torch.nn.Parameter(torch.tensor([0.0, 16., 32., 74.], dtype=torch.float32, device=self.device))
                K = len(self.classes)
                N = self.positions.shape[0]
                self.intensities_ = torch.nn.Parameter(torch.rand(N, K))
            elif self.train_methods.intensity_classes == "fixed":
                for i in range(len(init_intensities)):
                    self.intensities_[i].requires_grad = False
            else:

                raise ValueError("Unknown intensity class type")


    def initialize_intensities_parameters_old(self, init_intensities):
        if self.train_methods.intensity_classes == "free":
            self.intensities_ = nn.ParameterList(
                [torch.tensor(I, dtype=torch.float32, device=self.device) for I in init_intensities])
        elif self.train_methods.intensity_classes == "free_ws2":
            self.register_buffer("init_intensities_Gr", torch.tensor(init_intensities[self._IDX["Gr"]],
                                                                   dtype=torch.float32, device=self.device))
            self.intensities_ = nn.ParameterList(
                [torch.tensor(I, dtype=torch.float32, device=self.device) for I in init_intensities[1:]])
        elif self.train_methods.intensity_classes == "vacancies":
            self.register_buffer("init_intensities_Gr", torch.tensor(init_intensities[self._IDX["Gr"]],
                                                                     dtype=torch.float32, device=self.device))
            #self.init_intensities = [torch.tensor(I, dtype=torch.float32, device=self.device) for I in init_intensities]
            # self.init_intensities = init_intensities
            # self.register_buffer("init_intensities", torch.tensor(init_intensities, dtype=torch.float32))
            self.vacanciesS1 = nn.Parameter(
                torch.ones(init_intensities[1].shape, dtype=torch.float32, device=self.device))
            self.vacanciesW = nn.Parameter(
                torch.ones(init_intensities[2].shape, dtype=torch.float32, device=self.device))
            self.vacanciesS2 = nn.Parameter(
                torch.ones(init_intensities[3].shape, dtype=torch.float32, device=self.device))
            #vacancies_list = [self.vacanciesS1, self.vacanciesW, self.vacanciesS2]

            #self.vacancies_tensor = pad_sequence(vacancies_list, batch_first=True)


        elif self.intensity_classes == "classes": ## eher über loss funktion (abweichung von 16, 32, 74)
            self.classes = torch.nn.Parameter(torch.tensor([0.0, 16., 32., 74.], dtype=torch.float32, device=self.device))
            K = len(self.classes)
            N = self.positions.shape[0]
            self.intensities_ = torch.nn.Parameter(torch.rand(N, K))
        elif self.intensity_classes == "fixed":
            self.intensities_ = [torch.tensor(I, dtype=torch.float32, device=self.device) for I in init_intensities]
        else:

            raise ValueError("Unknown intensity class type")


    def get_positions(self):
        ret_positions = []
        rotations_ = (self.tunable_parameters.rotation_Gr, self.tunable_parameters.rotation_WS2)
        shifts_ = (self.tunable_parameters.shift_Gr, self.tunable_parameters.shift_WS2)
        scales_ = (self.tunable_parameters.scale_Gr, self.tunable_parameters.scale_WS2)


        for i in range(self.N_l):
            if i == self._IDX["Gr"]:
                idx_s_r = 0
            else:
                idx_s_r = 1



            # Rotation um den Mittelpunkt der Graphen-Positionen

            #center = torch.tensor((self.H/2, self.W/2), device=self.device)
            area_mask = self.train_methods.area_masks[idx_s_r]
            #center = torch.tensor([(x_min + x_max) / 2, (y_min + y_max) / 2], device=self.device)
            center = torch.tensor((self.H/2, self.W/2), device=self.device)


            if ((i == self._IDX["Gr"] and self.train_methods.graphen_rotation.enabled) or
                    (i != self._IDX["Gr"] and self.train_methods.WS2_rotation.enabled)):
                theta = rotations_[idx_s_r]
            else:
                theta = rotations_[idx_s_r].detach()

            if self.max_rotations is not None:
                theta = torch.clamp(theta, - self.max_rotations[idx_s_r], self.max_rotations[idx_s_r])
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            if ((i == self._IDX["Gr"] and self.train_methods.graphen_shift.enabled) or
                    (i != self._IDX["Gr"] and self.train_methods.WS2_shift.enabled)):
                shifts = shifts_[idx_s_r]
            else:
                shifts = shifts_[idx_s_r].detach()
            shift = torch.clamp(shifts, -self.max_shifts[idx_s_r],
                                self.max_shifts[idx_s_r]) if self.max_shifts is not None else shifts

            if ((i == self._IDX["Gr"] and self.train_methods.graphen_free_positions.enabled) or
                    (i != self._IDX["Gr"] and self.train_methods.WS2_free_positions.enabled)):
                position_shifts = self.tunable_parameters.position_shifts[i]
            else:
                position_shifts = self.tunable_parameters.position_shifts[i].detach()

            positions = self.get_init_positions(i) + position_shifts
            centered_positions = positions - center

            rotated_positions = torch.zeros_like(centered_positions)
            rotated_positions[:, 0] = cos_theta * centered_positions[:, 0] - sin_theta * centered_positions[:, 1]
            rotated_positions[:, 1] = sin_theta * centered_positions[:, 0] + cos_theta * centered_positions[:, 1]

            if ((i == self._IDX["Gr"] and self.train_methods.graphen_scale.enabled) or
                    (i != self._IDX["Gr"] and self.train_methods.WS2_scale.enabled)):
                scales = torch.clamp(scales_[idx_s_r], min=0.80, max=1.2)
            else:
                scales = scales_[idx_s_r].detach()

            rotated_positions = rotated_positions * scales
            new_positions = rotated_positions + center + shift

            if i == self._IDX["Gr"]:
                new_positions = self.strain_fields[0](new_positions)
            else:
                new_positions = self.strain_fields[1](new_positions)

            ret_positions.append(new_positions)
        return ret_positions

    def ws2_elastic_loss(self, lambda_=1.0, mu=1.0, sample_layers=("S1", "W", "S2")):
        """
        Elastische Energie des WS2-Verschiebungsfeldes.
        Bewertet an den aktuellen WS2-Atompositionen.
        """
        coords = []

        for name in sample_layers:
            idx = self._IDX[name]
            pos = self.get_positions()[idx].detach()
            coords.append(pos)

        coords = torch.cat(coords, dim=0)

        # Für autograd auf den Koordinaten:
        coords = coords.clone().detach().requires_grad_(True)

        W = self.ws2_displacement.elastic_energy_density(coords, lambda_=lambda_, mu=mu)
        return W.mean()


    def occupy_layers_in_area(self, mask, layers):
        pad = 100
        for l in layers:
            pos = self.get_positions()[l]#self.tunable_parameters.positions[l]
            xs, ys = pos[:,0], pos[:,1]
            for i in range(pos.shape[0]):
                x, y = int(xs[i]), int(ys[i])
                if x >= 0 and x < self.H and y >= 0 and y < self.W:
                    if mask[x, y]:
                        self.tunable_parameters.occupancies[l][i] = -100


    def get_positions_old(self):
        ret_positions = []
        for i in range(self.N_l):
            if i == self._IDX["Gr"]:
                idx_s_r = 0
            else:
                idx_s_r = 1

            # Rotation um den Mittelpunkt der Graphen-Positionen
            if ((i == self._IDX["Gr"] and self.train_methods.graphen_rotation.enabled) or
                    (i != self._IDX["Gr"] and self.train_methods.WS2_rotation.enabled)):
                center = torch.tensor((self.H/2, self.W/2), device=self.device)
                theta = self.rotations_[idx_s_r]
                if self.max_rotations is not None:
                    if i == self._IDX["Gr"]:
                        theta = torch.clamp(theta, 0.0, self.max_rotations[0])
                    else:
                        theta = torch.clamp(theta, 0.0, self.max_rotations[0])
                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)
            else:
                center = torch.zeros(2, device=self.device)

            if ((i == self._IDX["Gr"] and self.train_methods.graphen_shift.enabled) or
                    (i != self._IDX["Gr"] and self.train_methods.WS2_shift.enabled)):
                shift = torch.clamp(self.shifts_[idx_s_r], -self.max_shifts[0],
                                    self.max_shifts[0]) if self.max_shifts is not None else self.shifts_[idx_s_r]
                shifted_positions = self.positions_[i] - center + shift
            else:
                shifted_positions = self.positions_[i] - center

            if ((i == self._IDX["Gr"] and self.train_methods.graphen_rotation.enabled) or
                    (i != self._IDX["Gr"] and self.train_methods.WS2_rotation.enabled)):
                rotated_positions = torch.zeros_like(shifted_positions)
                rotated_positions[:, 0] = cos_theta * shifted_positions[:, 0] - sin_theta * shifted_positions[:, 1]
                rotated_positions[:, 1] = sin_theta * shifted_positions[:, 0] + cos_theta * shifted_positions[:, 1]
            else:
                center = torch.zeros(2, device=self.device)
                rotated_positions = shifted_positions

            new_positions = rotated_positions + center

            ret_positions.append(new_positions)
        return ret_positions

    def get_intensities(self):
        intensities = []
        vacancies = []
        for i in range(self.N_l):
            if ((i == self._IDX["Gr"] and self.train_methods.train_graphen_intensities.enabled) or
                    (i == self._IDX["S1"] and self.train_methods.train_S1_intensities.enabled) or
                    (i == self._IDX["W"] and self.train_methods.train_W_intensities.enabled) or
                    (i == self._IDX["S2"] and self.train_methods.train_S2_intensities.enabled)):
                intensities.append(self.tunable_parameters.intensities[i])
            else:
                intensities.append(self.tunable_parameters.intensities[i].detach())

            if ((i == self._IDX["Gr"] and self.train_methods.train_graphen_occupancies.enabled) or
                    (i == self._IDX["S1"] and self.train_methods.train_S1_occupancies.enabled) or
                    (i == self._IDX["W"] and self.train_methods.train_W_occupancies.enabled) or
                    (i == self._IDX["S2"] and self.train_methods.train_S2_occupancies.enabled)):
                vacancies.append(self.tunable_parameters.occupancies[i].sigmoid())
            else:
                vacancies.append(self.tunable_parameters.occupancies[i].sigmoid().detach())


        return [intensities[i] * vacancies[i] for i in range(self.N_l)]

    def get_intensities_raw(self):
        return self.tunable_parameters.intensities #* torch.sigmoid(self.occupancies_)


        if False:
            if self.train_methods.intensity_classes == "classes":
                raise NotImplementedError("Classes not implemented yet")
                # Softmax → Gewichtung über Klassen
                weights = F.softmax(self.intensities_, dim=1)  # shape (N, K)
                q = torch.sum(weights * self.classes, dim=1)  # gewichteter Mittelwert
                ret_intensities = q.unsqueeze(1)  # shape (N, 1)
                return ret_intensities
            elif (self.train_methods.intensity_classes == "free" or self.train_methods.intensity_classes == "fixed" or
                  self.train_methods.intensity_classes == "vacancies_WS2"):
                return self.intensities_
            elif self.train_methods.intensity_classes == "vacancies":
                raise NotImplementedError("Vacancies not implemented yet")
                return [self.init_intensities_Gr,
                        self.vacanciesS1.sigmoid() * 16.,
                        self.vacanciesW.sigmoid() * 74.,
                        self.vacanciesS2.sigmoid() * 16.
                        ]


    def get_pos_and_ints(self, pos_idx, filter_by_area=False):

        intensities = self.get_intensities_raw()[pos_idx].detach().clone().cpu().numpy()
        occ = self.occupancies[pos_idx].sigmoid().detach().cpu().numpy()
        occ_neg_mask = occ < 0.5
        intensities[occ_neg_mask] = None # TODO
        positions = self.get_positions()[pos_idx].cpu().detach().cpu().numpy()
        if filter_by_area:
            idx = self.train_method.only_train_small_area
            #x_min, y_min, x_max, y_max = self.train_methods.train_small_area_squares[idx]
            #mask = ((positions[:, 0] >= x_min) & (positions[:, 0] <= x_max) &
            #        (positions[:, 1] >= y_min) & (positions[:, 1] <= y_max))
            mask_img = self.train_methods.area_masks[idx]
            mask = mask_img[positions[:, 0].astype(int), positions[:, 1].astype(int)]
            positions = positions[mask]
            intensities = intensities[mask]
            #pos_idx = mask.nonzero(as_tuple=True)[0]

        return (positions[:,0], positions[:,1]), intensities

    def get_intensities_old(self):
        if self.train_methods.intensity_classes == "classes":
            # Softmax → Gewichtung über Klassen
            weights = F.softmax(self.intensities_, dim=1)  # shape (N, K)
            q = torch.sum(weights * self.classes, dim=1)  # gewichteter Mittelwert
            ret_intensities = q.unsqueeze(1)  # shape (N, 1)
            return ret_intensities
        elif self.train_methods.intensity_classes == "free":
            return self.intensities_
        elif self.intensity_classes == "free_ws2":
            ret_intensities = [self.init_intensities_Gr]
            for i in range(len(self.intensities_)):
                ret_intensities.append(self.intensities_[i])
            return ret_intensities
        elif self.train_methods.intensity_classes == "vacancies":
            raise NotImplementedError("Vacancies not implemented yet")
            return [self.init_intensities_Gr,
                    self.vacanciesS1.sigmoid() * 16.,
                    self.vacanciesW.sigmoid() * 74.,
                    self.vacanciesS2.sigmoid() * 16.
                    ]
        elif self.train_methods.intensity_classes == "fixed":
            return self.intensities_


    def aux_loss(self):

        entropies = []
        ps = [self.tunable_parameters.occupancies[i].sigmoid() for i in range(self.N_l)
              ]
        for i in range(1, len(ps)):
            p = ps[i]
            entropy = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).mean()
            entropies.append(entropy)
        entropy = torch.mean(torch.stack(entropies))

        if self.train_methods.position_shifts_regularization_a > 0.0:
            pos_shift_reg_stack = []
            for i in range(self.N_l):
                if (i == self._IDX["Gr"] and self.train_methods.graphen_free_positions.enabled) or \
                        (i != self._IDX["Gr"] and self.train_methods.WS2_free_positions.enabled):
                    pos_shift_reg_stack.append(torch.abs(self.tunable_parameters.position_shifts[i]).mean())
            if len(pos_shift_reg_stack) > 0:
                pos_shift_reg = torch.mean(torch.stack(pos_shift_reg_stack))
            else:
                pos_shift_reg = 0.0
        else:
            pos_shift_reg = 0.0

        if self.train_methods.strain_energy_a > 0.0 and self.train_methods.WS2_strain_field.enabled:
            strain_energy = self.ws2_elastic_loss(lambda_=self.train_methods.strain_energy_lambda, mu=self.train_methods.strain_energy_mu)
        else:
            strain_energy = 0.0

        return (self.train_methods.occ_entropy_a * entropy +
                self.train_methods.position_shifts_regularization_a * pos_shift_reg +
                self.train_methods.strain_energy_a * strain_energy)
        if self.train_methods.intensity_classes == "classes":
            weights = F.softmax(self.intensities_, dim=1)
            entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1).mean()
        elif self.train_methods.intensity_classes == "free":
            entropy = 0.0
        elif self.train_methods.intensity_classes == "vacancies":
            entropies = []
            ps = [self.vacanciesS1.sigmoid(),
                    self.vacanciesW.sigmoid(),
                    self.vacanciesS2.sigmoid()
                    ]
            for i in range(1, len(ps)):
               p = ps[i]
               entropy = -(p * torch.log(p + 1e-8) + (1-p) * torch.log(1-p + 1e-8)).mean()
               entropies.append(entropy)
            entropy = torch.mean(torch.stack(entropies))
        else:
            entropy = 0.0
        return entropy


    def forward(self, _):

        potential_list = []
        positions = self.get_positions()
        occupancies = self.tunable_parameters.occupancies

        for i in range(self.N_l):
            if int(self.train_methods.only_train_small_area) == 0 and i > 0:
                potential_i = torch.zeros((self.H+self.potential_pad*2, self.W+self.potential_pad*2), dtype=torch.float32, device=self.device)
            else:
                #potential_i = self.projected_slice_potential_lobato(positions[i], occupancies[i].sigmoid(), atom_type=self.atom_names_by_layer[i])
                potential_i = self.projected_slice_potential_kirkland(positions[i], occupancies[i].sigmoid(), atom_type=self.atom_names_by_layer[i])

            if i == self._IDX["Gr"]:
                potential_i = potential_i * self.tunable_parameters.graphene_potential_factor
            else:
                potential_i = potential_i * self.tunable_parameters.ws2_potential_factor
            potential_list.append(potential_i)

        potential_list = potential_list[::-1]

        return potential_list, self.aux_loss()

    def get_potential_images(self):
        pot_list, _ = self(None)
        pad = self.potential_pad
        pot_images = [p.detach().cpu().numpy()[pad:self.H+pad, pad:self.W+pad] for p in pot_list[::-1]]
        return pot_images



    def rasterize_atoms_bilinear(self, H, W, positions_xy, weights):
        rho = torch.zeros((H, W), dtype=torch.float32, device=self.device)

        x = positions_xy[:, 0]
        y = positions_xy[:, 1]

        mask = (x >= 0.0) & (x < H - 1) & (y >= 0.0) & (y < W - 1)
        x = x[mask]
        y = y[mask]
        w = weights[mask]

        if x.numel() == 0:
            return rho

        ix = torch.floor(x).long()
        iy = torch.floor(y).long()

        fx = x - ix
        fy = y - iy

        w00 = (1 - fx) * (1 - fy) * w
        w10 = fx * (1 - fy) * w
        w01 = (1 - fx) * fy * w
        w11 = fx * fy * w

        #rho.index_put_((iy, ix), w00, accumulate=True)
        #rho.index_put_((iy + 1, ix), w01, accumulate=True)
        #rho.index_put_((iy, ix + 1), w10, accumulate=True)
        #rho.index_put_((iy + 1, ix + 1), w11, accumulate=True)

        rho.index_put_((ix, iy), w00, accumulate=True)
        rho.index_put_((ix, iy + 1), w01, accumulate=True)
        rho.index_put_((ix + 1, iy), w10, accumulate=True)
        rho.index_put_((ix + 1, iy + 1), w11, accumulate=True)

        return rho




    def projected_slice_potential_lobato(self, positions_xy, occupancies, atom_type):
        pad = max(self.H // 2, self.W // 2)
        Hp = self.H + 2 * pad
        Wp = self.W + 2 * pad

        positions_pad = positions_xy + torch.tensor([pad, pad], device=self.device, dtype=positions_xy.dtype)

        rho = self.rasterize_atoms_bilinear(
            H=Hp,
            W=Wp,
            positions_xy=positions_pad,
            weights=occupancies
        )

        kernel = projected_atomic_potential_kernel_lobato(
            H=Hp,
            W=Wp,
            pixel_size_nm=self.pixel_size_nm,
            atom_type=atom_type,
            device=self.device,
        )

        Vp = torch.fft.ifft2(torch.fft.fft2(rho) * torch.fft.fft2(kernel)).real
        V = Vp[pad:pad + self.H, pad:pad + self.W]


        # globaler Offset ist für TEM-Intensität nicht relevant
        V = V - V.median()
        return V

    def projected_slice_potential_kirkland(self, positions_xy, occupancies, atom_type):
        pad = self.potential_pad#max(self.H // 2, self.W // 2)
        Hp = self.H + 2 * pad
        Wp = self.W + 2 * pad

        positions_pad = positions_xy + torch.tensor([pad, pad], device=self.device, dtype=positions_xy.dtype)

        rho = self.rasterize_atoms_bilinear(
            H=Hp,
            W=Wp,
            positions_xy=positions_pad,
            weights=occupancies
        )


        Vp = self.kirkland_precalc_pot.projected_slice_potential_kirkland(rho, atom_type, self.pixel_size_nm, self.device)
        V = Vp#[pad:pad + self.H, pad:pad + self.W]
        V = V - V.median()

        return V

