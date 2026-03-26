from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from displacement_field import SplineDisplacementField2D, StrainTunableParameters

from torch.nn.utils.rnn import pad_sequence


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)

@dataclass
class TunablePotentialParameters:
    ws2_displacement: StrainTunableParameters

    positions: nn.ParameterList
    intensities: nn.ParameterList
    occupancies: nn.ParameterList

    scale_Gr: nn.Parameter = nn.Parameter(torch.tensor([1.12, 1.14], dtype=torch.float32))
    rotation_Gr: nn.Parameter = nn.Parameter(torch.tensor(0.025, dtype=torch.float32))
    shift_Gr: nn.Parameter = nn.Parameter(torch.tensor([-9.5, -11.0], dtype=torch.float32))

    scale_WS2: nn.Parameter = nn.Parameter(torch.tensor([1.05, 1.09 ], dtype=torch.float32))
    rotation_WS2: nn.Parameter = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    shift_WS2: nn.Parameter = nn.Parameter(torch.tensor([15.9, -5.0 ], dtype=torch.float32))

    blur_sigma: nn.Parameter = nn.Parameter(torch.tensor(0.013797, dtype=torch.float32))




class DifferentiablePotential(nn.Module):
    def __init__(self, train_methods, init_positions, init_intensities, shape,
                 pixel_size_nm, wavelength_nm, device, init_blur_sigma=1.0,
                 max_shifts=None, max_rotations=None):
        super().__init__()
        #self.slices_V = [v.to(device) for v in slices_V]  # Liste komplexer Slices (H,W)
        self.pixel_size_nm = pixel_size_nm
        self.wavelength_nm = wavelength_nm
        self.device = device
        self.max_shifts = max_shifts
        self.max_rotations = max_rotations

        self.H, self.W = shape

        self._IDX = {"Gr": 0, "S1": 1, "W": 2, "S2": 3}
        self.N_l = len(init_positions)

        self.train_methods = train_methods

        self.ws2_displacement = SplineDisplacementField2D(
            image_shape=shape,
            control_shape=(8,8),
            init_scale=0.0,
            device=device,
            dtype=torch.float32,
        )
        #self.positions_method = "graphen_shift"
        #self.intensity_classes = "free" # "free", "vacancies", "classes"
        strain_parameters = self.ws2_displacement.get_tunable_parameters()
        positions_ = self.initialize_positions_parameters(init_positions)
        intensities_, occupancies_ = self.initialize_intensities_parameters(init_intensities)

        self.tunable_parameters = TunablePotentialParameters(ws2_displacement=strain_parameters, positions=positions_,
                                                             intensities=intensities_, occupancies=occupancies_,)



        self.apply_trainable_selection()
        #self.zero_grads_for_frozen()

        for name, value in vars(self.tunable_parameters).items():
            if isinstance(value, (nn.Parameter, nn.ParameterList, nn.ParameterDict)):
                setattr(self, name, value)

    def get_tunable_parameters(self):
        return self.tunable_parameters


    def apply_trainable_selection(self):
            self.tunable_parameters.intensities[0].requires_grad = self.train_methods.train_graphen_intensities
            self.tunable_parameters.intensities[1].requires_grad = self.train_methods.train_S1_intensities
            self.tunable_parameters.intensities[2].requires_grad = self.train_methods.train_W_intensities
            self.tunable_parameters.intensities[3].requires_grad = self.train_methods.train_S2_intensities

            self.tunable_parameters.occupancies[0].requires_grad = self.train_methods.train_graphen_occupancies
            self.tunable_parameters.occupancies[1].requires_grad = self.train_methods.train_S1_occupancies
            self.tunable_parameters.occupancies[2].requires_grad = self.train_methods.train_W_occupancies
            self.tunable_parameters.occupancies[3].requires_grad = self.train_methods.train_S2_occupancies

            self.tunable_parameters.shift_Gr.requires_grad = self.train_methods.graphen_shift
            self.tunable_parameters.rotation_Gr.requires_grad = self.train_methods.graphen_rotation
            self.tunable_parameters.positions[self._IDX["Gr"]].requires_grad = self.train_methods.graphen_free_positions

            self.tunable_parameters.shift_WS2.requires_grad = self.train_methods.WS2_shift
            self.tunable_parameters.rotation_WS2.requires_grad = self.train_methods.WS2_rotation
            for i in range(self.N_l):
                if i != self._IDX["Gr"]:
                    self.tunable_parameters.positions[i].requires_grad = self.train_methods.WS2_free_positions

            self.tunable_parameters.blur_sigma.requires_grad = self.train_methods.train_blur

            for p in self.ws2_displacement.parameters():
                p.requires_grad = getattr(self.train_methods, "WS2_strain_field", False)



    def zero_grads_for_frozen(self):
        if not self.train_methods.train_graphen_intensities:
            self.intensities[0].grad = None
        if not self.train_methods.train_S1_intensities:
            self.intensities[1].grad = None
        if not self.train_methods.train_W_intensities:
            self.intensities[2].grad = None
        if not self.train_methods.train_S2_intensities:
            self.intensities[3].grad = None

        if not  self.train_methods.train_graphen_occupancies:
            self.occupancies[0].grad = None
        if not self.train_methods.train_S1_occupancies:
            self.occupancies[1].grad = None
        if not self.train_methods.train_W_occupancies:
            self.occupancies[2].grad = None
        if not self.train_methods.train_S2_occupancies:
            self.occupancies[3].grad = None

        if not self.train_methods.graphen_shift:
            self.tunable_parameters.shift_Gr.grad = None
        if not self.train_methods.graphen_rotation:
            self.tunable_parameters.rotation_Gr.grad = None
        if not self.train_methods.graphen_free_positions:
            self.positions[self._IDX["Gr"]].grad = None

        if not self.train_methods.WS2_shift:
            self.tunable_parameters.shift_WS2.grad = None
        if not self.train_methods.WS2_rotation:
            self.tunable_parameters.rotation_WS2.grad = None
        for i in range(self.N_l):
            if i != self._IDX["Gr"] and not self.train_methods.WS2_free_positions:
                self.positions[i].grad = None


    def initialize_positions_parameters(self, init_positions):
        positions_ = nn.ParameterList(
            [torch.tensor(pos, dtype=torch.float32, device=self.device) for pos in init_positions])

        positions_[self._IDX["Gr"]].requires_grad = self.train_methods.graphen_free_positions
        for i in range(self.N_l):
            if i != self._IDX["Gr"]:
                positions_[i].requires_grad = self.train_methods.WS2_free_positions

        return positions_






    def initialize_intensities_parameters(self, init_intensities):
        intensities_ = nn.ParameterList(
            [torch.tensor(I, dtype=torch.float32, device=self.device) for I in init_intensities])
        occupancies_ = nn.ParameterList(
            [torch.ones(I.shape, dtype=torch.float32, device=self.device)*0.1 for I in init_intensities])


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
            x_min, y_min, x_max, y_max = self.train_methods.train_small_area_squares[idx_s_r]
            center = torch.tensor([(x_min + x_max) / 2, (y_min + y_max) / 2], device=self.device)


            if ((i == self._IDX["Gr"] and self.train_methods.graphen_rotation) or
                    (i != self._IDX["Gr"] and self.train_methods.WS2_rotation)):
                theta = rotations_[idx_s_r]
            else:
                theta = rotations_[idx_s_r].detach()

            if self.max_rotations is not None:
                theta = torch.clamp(theta, - self.max_rotations[idx_s_r], self.max_rotations[idx_s_r])
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            if ((i == self._IDX["Gr"] and self.train_methods.graphen_shift) or
                    (i != self._IDX["Gr"] and self.train_methods.WS2_shift)):
                shifts = shifts_[idx_s_r]
            else:
                shifts = shifts_[idx_s_r].detach()
            shift = torch.clamp(shifts, -self.max_shifts[idx_s_r],
                                self.max_shifts[idx_s_r]) if self.max_shifts is not None else shifts

            if ((i == self._IDX["Gr"] and self.train_methods.graphen_free_positions) or
                    (i != self._IDX["Gr"] and self.train_methods.WS2_free_positions)):
                positions = self.tunable_parameters.positions[i]
            else:
                positions = self.tunable_parameters.positions[i].detach()


            shifted_positions = positions - center + shift

            rotated_positions = torch.zeros_like(shifted_positions)
            rotated_positions[:, 0] = cos_theta * shifted_positions[:, 0] - sin_theta * shifted_positions[:, 1]
            rotated_positions[:, 1] = sin_theta * shifted_positions[:, 0] + cos_theta * shifted_positions[:, 1]

            if ((i == self._IDX["Gr"] and self.train_methods.graphen_scale) or
                    (i != self._IDX["Gr"] and self.train_methods.WS2_scale)):

                scales = torch.clamp(scales_[idx_s_r], min=0.80, max=1.2)
            else:
                scales = scales_[idx_s_r].detach()
            rotated_positions = rotated_positions * scales

            new_positions = rotated_positions + center

            if i != self._IDX["Gr"]:
                new_positions = self.ws2_displacement(new_positions)

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


    def occupy_layers_in_area(self, area, layers):
        x0, y0, x1, y1 = area
        pad = 100
        x0 -= pad
        x1 += pad
        y0 -= pad
        y1 += pad
        for l in layers:
            pos = self.tunable_parameters.positions[l]
            xs, ys = pos[:,0], pos[:,1]
            for i in range(pos.shape[0]):
                if (xs[i] >= x0 and xs[i] <= x1) and \
                        (ys[i] >= y0 and ys[i] <= y1):
                    self.tunable_parameters.occupancies[l][i] = -100


    def get_positions_old(self):
        ret_positions = []
        for i in range(self.N_l):
            if i == self._IDX["Gr"]:
                idx_s_r = 0
            else:
                idx_s_r = 1

            # Rotation um den Mittelpunkt der Graphen-Positionen
            if ((i == self._IDX["Gr"] and self.train_methods.graphen_rotation) or
                    (i != self._IDX["Gr"] and self.train_methods.WS2_rotation)):
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

            if ((i == self._IDX["Gr"] and self.train_methods.graphen_shift) or
                    (i != self._IDX["Gr"] and self.train_methods.WS2_shift)):
                shift = torch.clamp(self.shifts_[idx_s_r], -self.max_shifts[0],
                                    self.max_shifts[0]) if self.max_shifts is not None else self.shifts_[idx_s_r]
                shifted_positions = self.positions_[i] - center + shift
            else:
                shifted_positions = self.positions_[i] - center

            if ((i == self._IDX["Gr"] and self.train_methods.graphen_rotation) or
                    (i != self._IDX["Gr"] and self.train_methods.WS2_rotation)):
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
            if ((i == self._IDX["Gr"] and self.train_methods.train_graphen_intensities) or
                    (i == self._IDX["S1"] and self.train_methods.train_S1_intensities) or
                    (i == self._IDX["W"] and self.train_methods.train_W_intensities) or
                    (i == self._IDX["S2"] and self.train_methods.train_S2_intensities)):
                intensities.append(self.tunable_parameters.intensities[i])
            else:
                intensities.append(self.tunable_parameters.intensities[i].detach())

            if ((i == self._IDX["Gr"] and self.train_methods.train_graphen_occupancies) or
                    (i == self._IDX["S1"] and self.train_methods.train_S1_occupancies) or
                    (i == self._IDX["W"] and self.train_methods.train_W_occupancies) or
                    (i == self._IDX["S2"] and self.train_methods.train_S2_occupancies)):
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
            x_min, y_min, x_max, y_max = self.train_methods.train_small_area_squares[idx]
            mask = ((positions[:, 0] >= x_min) & (positions[:, 0] <= x_max) &
                    (positions[:, 1] >= y_min) & (positions[:, 1] <= y_max))
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

        return entropy
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
        intensities = self.get_intensities()

        for i in range(self.N_l):
            rho_i = self.create_rho(positions[i], intensities[i])
            # rho_i = self.simulation.calc_rho(self.positions[i], torch.abs(self.intensities[i]).unsqueeze(1))

            potential_i = self.potential_2d_fft_nm(rho_i, device=self.device)
            potential_list.append(potential_i)

        return potential_list, self.aux_loss()

    def create_rho(self, positions_xy, intensities, multiplier=1.0):
        """
        positions_xy: (N, 2) mit [:,0]=x und [:,1]=y
        intensities:  (N,)
        """
        intensities = intensities.unsqueeze(1)
        pad = 2

        # Bildtensor immer als (H, W)
        rho = torch.zeros((self.H + 2 * pad, self.W + 2 * pad), device=self.device)

        x = positions_xy[:, 0]
        y = positions_xy[:, 1]


        mask = (x >= 0.0) & (x < self.H - 1) & (y >= 0.0) & (y < self.W - 1)
        x = x[mask]
        y = y[mask]
        intensities = intensities[mask]

        ix = torch.floor(x).to(torch.int64)
        iy = torch.floor(y).to(torch.int64)

        fx = x - ix
        fy = y - iy

        # bilineare Gewichte
        w00 = ((1 - fx) * (1 - fy) * intensities[:, 0]) * multiplier
        w10 = (fx * (1 - fy) * intensities[:, 0]) * multiplier
        w01 = ((1 - fx) * fy * intensities[:, 0]) * multiplier
        w11 = (fx * fy * intensities[:, 0]) * multiplier





        idx00 = (pad + ix, pad + iy)
        idx10 = (pad + ix, pad + iy + 1)
        idx11 = (pad + ix + 1, pad + iy + 1)
        idx01 = (pad + ix + 1, pad + iy)

        # In-place Akkumulation wie np.add.at
        rho.index_put_(idx00, w00, accumulate=True)
        rho.index_put_(idx10, w10, accumulate=True)
        rho.index_put_(idx11, w11, accumulate=True)
        rho.index_put_(idx01, w01, accumulate=True)

        return torch.abs(rho[pad:-pad, pad:-pad])


    def create_rho_old(self, pxpositions, intensities, multiplier=1.0):
        # pxpositions: (N, 2) Tensor
        # intensities: (N,) Tensor

        intensities = intensities.unsqueeze(1)  # (N, 1) Tensor

        pad = 2

        rho = torch.zeros((self.H + pad * 2, self.W + pad * 2), device=self.device)

        # potential.fill_(self.background)  # Torch-Äquivalent

        px, py = pxpositions[:, 0], pxpositions[:, 1] # TODO warum vertauscht

        pos_mask = (px >= 0.0) & (px <= self.H) & (py >= 0.0) & (py <= self.W)
        px = px[pos_mask]
        py = py[pos_mask]
        intensities = intensities[pos_mask]
        # offset = int(self.canvas_size / self.kernelsize)
        offset = pad

        # Integer-Anteile
        # ix = torch.clamp(px.floor().to(torch.int64), min=0, max=self.shape[0] + self.canvas_size - 2)
        # iy = torch.clamp(py.floor().to(torch.int64), min=0, max=self.shape[1] + self.canvas_size - 2)

        ix = px.floor().to(torch.int64)
        iy = py.floor().to(torch.int64)

        # Bruchteile
        fx = px - ix
        fy = py - iy

        # Pixelgewichte
        w00 = torch.matmul((1 - fx), (1 - fy)) * intensities * multiplier
        w10 = torch.matmul(fx, (1 - fy)) * intensities* multiplier
        w11 = torch.matmul(fx, fy) * intensities* multiplier
        w01 = torch.matmul((1 - fx), fy) * intensities* multiplier

        # Indizes für alle vier Nachbarn
        idx00 = (offset + ix, offset + iy)
        idx10 = (offset + ix, offset + iy + 1)
        idx11 = (offset + ix + 1, offset + iy + 1)
        idx01 = (offset + ix + 1, offset + iy)

        # In-place Akkumulation wie np.add.at
        rho.index_put_(idx00, w00[:idx00[0].shape[0], 0], accumulate=True)
        rho.index_put_(idx10, w10[:idx10[0].shape[0], 0], accumulate=True)
        rho.index_put_(idx11, w11[:idx11[0].shape[0], 0], accumulate=True)
        rho.index_put_(idx01, w01[:idx01[0].shape[0], 0], accumulate=True)

        return torch.abs(rho[pad:-pad, pad:-pad])  # Entfernen des Pads

    def potential_2d_nm(self, rho, sigma=0.1, epsilon0_nm=8.854e-21, device=None):
        """
        Berechnet geglättetes 2D-Potential aus Delta-Peaks (nm-Einheiten)

        rho: 2D-Tensor [ny,nx] in Vielfachen von e
        spacing: Pixelgröße in nm
        sigma: Glättungs-Gauss in nm
        epsilon0_nm: Permittivität in F/nm
        """
        e = 1.602e-19
        rho = rho.to(device, dtype=torch.float32)
        ny, nx = (self.W, self.H)
        spacing = self.pixel_size_nm

        # Gaussian-Kernel
        s_pix = max(1, int(6 * sigma / spacing))
        if s_pix % 2 == 0: s_pix += 1
        coords = torch.arange(s_pix, device=device) - s_pix // 2
        Y, X = torch.meshgrid(coords, coords, indexing='ij')
        R2 = (X * spacing) ** 2 + (Y * spacing) ** 2
        G_kernel = torch.exp(-R2 / (2 * sigma ** 2))
        G_kernel /= G_kernel.sum()

        # Faltung
        rho_input = rho.unsqueeze(0).unsqueeze(0)
        kernel = G_kernel.unsqueeze(0).unsqueeze(0)
        V = F.conv2d(rho_input, kernel, padding=s_pix // 2)
        V = V[0, 0] / (4 * torch.pi * epsilon0_nm)  # Volt

        return V

    def solve_potential(self,iterations=100):
        V = torch.zeros_like(self.potential)  # initial guess
        for _ in range(iterations):
            V = 0.25 * (F.pad(V, (1, 1, 1, 1), mode='replicate')[:, :-2, 1:-1] +
                        F.pad(V, (1, 1, 1, 1), mode='replicate')[:, 2:, 1:-1] +
                        F.pad(V, (1, 1, 1, 1), mode='replicate')[:, 1:-1, :-2] +
                        F.pad(V, (1, 1, 1, 1), mode='replicate')[:, 1:-1, 2:] +
                        self.potential)  # simple Jacobi iteration
        return V

    def potential_2d_fft_nm(self, rho, epsilon0_nm=8.854e-21, device=None):
        """
        Berechnet 2D Potential mit FFT-Faltung (viel schneller für große Raster)

        rho: 2D-Tensor [ny,nx] in Vielfachen von e
        spacing: Pixelgröße in nm
        sigma: Glättungs-Gauss in nm
        """

        sigma = self.blur_sigma * 1.e-3

        e = 1.602e-19
        ny, nx = (self.W, self.H)
        spacing = self.pixel_size_nm

        # Frequenzraum-Koordinaten
        y = torch.fft.fftfreq(ny, d=spacing, device=device)
        x = torch.fft.fftfreq(nx, d=spacing, device=device)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        R2 = X ** 2 + Y ** 2

        # FFT von 1/sqrt(r^2 + sigma^2) in Fourierraum:
        # Für Gaussian-Kernel einfacher: F(G) = exp(-2 pi^2 sigma^2 (fx^2 + fy^2))
        G_fft = torch.exp(-2 * torch.pi ** 2 * sigma ** 2 * R2)

        # FFT des Rasters
        rho_fft = torch.fft.fft2(rho * e)

        # Faltung im Fourierraum
        V_fft = rho_fft * G_fft
        V = torch.fft.ifft2(V_fft).real / (4 * torch.pi * epsilon0_nm)

        return V