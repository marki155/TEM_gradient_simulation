from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json


from loss_functions import *
from multislice import DifferentiableMultislice
from potential import DifferentiablePotential, TunablePotentialParameters

from ctf import DEFAULT_PROBE_ABERRATIONS, CTFLayer, TunableCTFParameters
from stem import DifferentiableHAADF

from layers import (
    LayerConfig,
    LayerGroupConfig,
    legacy_layers_and_groups,
    resolve_layers_and_groups,
)


@dataclass
class ParticlesTunableParameters:
    potential_parameters: TunablePotentialParameters
    ctf_parameters: TunableCTFParameters
    normalisation_parameters: nn.Parameter




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

def get_trainmethod_param_registry(params: ParticlesTunableParameters,
                                   layer_names, group_names):
    """Bildet die Schluessel aus :meth:`TrainMethod.iter_train_params` auf Parameter ab.

    Schichten und Gruppen kommen aus der Config, es ist nichts fest verdrahtet.
    """

    pot = params.potential_parameters
    ctf = params.ctf_parameters

    registry = {
        # CTF
        "train_amplitude_contrast": lambda: [ctf.amp_contrast],
        "train_ctf_aberrations": lambda: list(ctf.aberrations.values()),
        "train_env_parameters": lambda: [ctf.deltaE, ctf.Cc_um],

        # Normalisation
        "train_normalisation_parameters": lambda: [params.normalisation_parameters],
    }

    for gname in group_names:
        registry[f"groups.{gname}.shift"] = lambda g=gname: [pot.shifts[g]]
        registry[f"groups.{gname}.rotation"] = lambda g=gname: [pot.rotations[g]]
        registry[f"groups.{gname}.scale"] = lambda g=gname: [pot.scales[g]]
        registry[f"groups.{gname}.contrast"] = lambda g=gname: [pot.contrast_factors[g]]
        registry[f"groups.{gname}.strain_field"] = (
            lambda g=gname: _collect_strain_params(pot.strain_fields[g]))

    for i, lname in enumerate(layer_names):
        registry[f"layers.{lname}.free_positions"] = (
            lambda idx=i: _safe_index_paramlist(pot.position_shifts, idx))
        registry[f"layers.{lname}.intensities"] = (
            lambda idx=i: _safe_index_paramlist(pot.intensities, idx))
        registry[f"layers.{lname}.occupancies"] = (
            lambda idx=i: _safe_index_paramlist(pot.occupancies, idx))

    return registry


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
    """Beschreibt das Modell - inkl. der frei definierbaren Schichten.

    ``layers``/``groups`` ersetzen die alte feste Gr/S1/W/S2-Struktur.  Alte
    configs ohne diese Felder werden in ``__post_init__`` automatisch migriert
    (aus ``z_positions_nm``, ``control_points_*``, ``max_shifts``,
    ``max_rotations``).
    """

    init_positions: List[np.ndarray]
    init_intensities: List[np.ndarray]
    shape: tuple
    pixel_size_nm: float
    voltage_kV: float
    init_aberrations: dict
    init_ctf_env_params: dict
    init_amp_contrast: float
    ctf_alpha_max_mrad: float
    padding: int

    #: Schichten (eine je Multislice-Slice) und ihre Struktur-Gruppen
    layers: List[LayerConfig] = field(default_factory=list)
    groups: List[LayerGroupConfig] = field(default_factory=list)

    #: "+z" = Strahl laeuft von der kleinsten zur groessten z-Position
    beam_direction: str = "+z"

    # --- Abbildung -----------------------------------------------------------
    #: "tem" = Multislice + CTF (wie bisher), "stem_haadf" = Sonde x Objekt
    imaging_mode: str = "tem"
    #: Konvergenzwinkel der Sonde (Halbwinkel, mrad)
    stem_convergence_mrad: float = 25.0
    #: innerer und aeusserer Detektorwinkel des HAADF-Rings (mrad)
    stem_detector_mrad: Tuple[float, float] = (60.0, 200.0)
    #: endliche Quellgroesse, Gauss-Sigma in Angstroem (verschmiert die Sonde)
    stem_source_size_A: float = 0.5
    #: eigene Aberrationen der STEM-Sonde; None -> init_aberrations
    init_probe_aberrations: Optional[dict] = None

    #: Schreibweise der Aberrationen: "cartesian" (a/b, besser zu optimieren)
    #: oder "polar" (C/phi wie frueher). Die Physik ist identisch.
    ctf_parametrization: str = "cartesian"

    # --- Legacy-Felder (nur noch fuer die Migration alter configs) -----------
    z_positions_nm: Optional[List[float]] = None
    control_points_graphen: Optional[List[int]] = None
    control_points_ws2: Optional[List[int]] = None
    max_shifts: Optional[Tuple[float, float]] = None
    max_rotations: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        if not self.layers:
            legacy_layers, legacy_groups = legacy_layers_and_groups(
                z_positions_nm=self.z_positions_nm or [],
                control_points_graphen=self.control_points_graphen or (10, 10),
                control_points_ws2=self.control_points_ws2 or (10, 10),
                max_shifts=self.max_shifts,
                max_rotations=self.max_rotations,
                n_layers=len(self.init_positions),
            )
            self.layers = legacy_layers
            if not self.groups:
                self.groups = legacy_groups

        self.layers, self.groups = resolve_layers_and_groups(self.layers, self.groups)

        # z_positions_nm bleibt als abgeleitete Sicht erhalten
        self.z_positions_nm = [float(l.z_nm) for l in self.layers]

    # -- Bequeme Sichten ------------------------------------------------------
    @property
    def layer_names(self) -> List[str]:
        return [l.name for l in self.layers]

    @property
    def group_names(self) -> List[str]:
        return [g.name for g in self.groups]

    def group(self, name: str) -> LayerGroupConfig:
        for g in self.groups:
            if g.name == name:
                return g
        raise KeyError(f"Unbekannte Gruppe {name!r}, vorhanden: {self.group_names}")

    def layer_indices(self, group_name: str) -> List[int]:
        return [i for i, l in enumerate(self.layers) if l.group == group_name]

    @classmethod
    def from_dict(cls, data: dict) -> "ParticlesConfig":
        data = dict(data)
        data["init_positions"] = [np.array(x) for x in data["init_positions"]]
        data["init_intensities"] = [np.array(x) for x in data["init_intensities"]]
        data["shape"] = tuple(data["shape"])
        known = {f.name for f in fields(cls)}
        unknown = [k for k in data if k not in known]
        for k in unknown:
            data.pop(k)
        if unknown:
            print(f"[config] ignorierte unbekannte Felder: {unknown}")
        return cls(**data)


#: Registry fuer die (De)Serialisierung einer ParticlesConfig
CONFIG_CLASS_REGISTRY = {
    "ParticlesConfig": ParticlesConfig,
    "LayerConfig": LayerConfig,
    "LayerGroupConfig": LayerGroupConfig,
}


class NN_particles(nn.Module):
    def __init__(self, config, pixel_size_nm, voltage_kV, train_method, device='cpu', train_methods=None):
        super().__init__()
        self.config = config
        self.shape = config.shape
        self.pixel_size_nm = pixel_size_nm
        self.voltage_kV = voltage_kV
        self.device = device
        self.train_method = train_method
        self.train_methods = train_methods

    def build_optimizer(self,
            optimizer_cls=torch.optim.Adam,
            **optimizer_kwargs,
    ):
        registry = get_trainmethod_param_registry(
            self.tunable_parameters,
            layer_names=self.config.layer_names,
            group_names=self.config.group_names,
        )

        param_groups = []

        for name, train_param in self.train_method.iter_train_params():
            if name not in registry:
                # z.B. Reste einer alten train_method, die es in dieser Config
                # nicht mehr gibt (andere Schichten/Gruppen)
                print(f"[optimizer] kein Parameter fuer '{name}' in dieser Config - uebersprungen")
                continue

            params = registry[name]()
            if not params:
                continue

            param_groups.append({
                "name": name,
                "params": params,
                "lr": float(train_param.lr),
            })

        if not param_groups:
            raise ValueError("Keine aktiven Parametergruppen aus TrainMethod gefunden.")

        return optimizer_cls(param_groups, **optimizer_kwargs)

    # -------------------------
    # SAVE
    # -------------------------
    def save_particles_config(self, path: str):
        def serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        data = asdict(self.config)

        # numpy Arrays konvertieren
        data["init_positions"] = [serialize(x) for x in data["init_positions"]]
        data["init_intensities"] = [serialize(x) for x in data["init_intensities"]]
        data["shape"] = serialize(data["shape"])

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # -------------------------
    # LOAD
    # -------------------------
    def load_particles_config(self, path: str) -> ParticlesConfig:
        with open(path, "r") as f:
            data = json.load(f)

        self.config = ParticlesConfig.from_dict(data)

        return self.config

    def build(self):
        """Modell passend zu ``config.imaging_mode`` aufbauen."""

        mode = getattr(self.config, "imaging_mode", "tem")
        if mode == "tem":
            return self.build_multislice()
        if mode == "stem_haadf":
            return self.build_stem()
        raise ValueError(f"Unbekannter imaging_mode: {mode!r}")

    def build_stem(self):
        """STEM-HAADF: Elementdichten -> Querschnitt -> Faltung mit der Sonde.

        Die Sonde entsteht aus derselben Aberrationsfunktion wie die CTF, der
        CTFLayer dient hier also als Sondenformer.
        """

        potential_layer = DifferentiablePotential(self.config, self.train_method,
                                                  device=self.device,
                                                  output_mode="element_density")
        # Ohne eigene Vorgabe wuerde die Sonde die TEM-Aberrationen erben - bei
        # C10 = -13 nm waere sie um vier Viertelwellen defokussiert und ueber
        # tausende Pixel verschmiert. Der Sondensatz startet stattdessen bei 0.
        probe_aberrations = getattr(self.config, "init_probe_aberrations", None)
        if not probe_aberrations:
            probe_aberrations = dict(DEFAULT_PROBE_ABERRATIONS)
        probe_layer = CTFLayer(self.config, device=self.device, train_method=self.train_method,
                               aberrations=probe_aberrations)
        haadf_layer = DifferentiableHAADF(self.config, self.train_method,
                                          probe_layer, device=self.device)

        self.model = nn.Sequential(potential_layer, haadf_layer)
        self.potential_layer = potential_layer
        self.imaging_layer = haadf_layer
        self.probe_layer = probe_layer

        self.tunable_parameters = ParticlesTunableParameters(
            potential_parameters=potential_layer.get_tunable_parameters(),
            ctf_parameters=probe_layer.get_tunable_parameters(),
            normalisation_parameters=nn.Parameter(
                torch.tensor([0.0, 0.0], device=self.device), requires_grad=True)
        )

        for name, value in vars(self.tunable_parameters).items():
            if isinstance(value, (nn.Parameter, nn.ParameterList, nn.ParameterDict)):
                setattr(self, name, value)

        return self.model

    def build_multislice(self):
        potential_layer = DifferentiablePotential(self.config, self.train_method, device=self.device)


        multislice = DifferentiableMultislice(
            self.config,
            train_method=self.train_method,
            device=self.device)



        ctf_layer = CTFLayer(self.config,
                             device=self.device,
                             train_method=self.train_method)

        self.model = nn.Sequential(
            potential_layer,
            multislice,
            ctf_layer)

        self.potential_layer = potential_layer
        self.imaging_layer = ctf_layer

        self.tunable_parameters = ParticlesTunableParameters(
            potential_parameters=self.model[0].get_tunable_parameters(),
            ctf_parameters=self.model[2].get_tunable_parameters(),
            normalisation_parameters = nn.Parameter(
                torch.tensor([0.0, 0.0], device=self.device), requires_grad=True)
        )

        for name, value in vars(self.tunable_parameters).items():
            if isinstance(value, (nn.Parameter, nn.ParameterList, nn.ParameterDict)):
                setattr(self, name, value)

    def get_tunable_parameters(self):
        return self.tunable_parameters


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

    def _union_mask(self):
        """Maske ueber alle Gruppen (fuer das Main-Training)."""

        masks = [m for m in self.train_methods.masks.values() if m is not None]
        if not masks:
            return None
        union = masks[0] > 0
        for m in masks[1:]:
            union = torch.logical_or(union, m > 0)
        return union.float()

    def get_current_target_and_mask_img(self, target_image):
        """Zielbild + Maske der aktiven Gruppe.

        Hat die aktive Gruppe ein eigenes Zielbild (``targets[gruppe]``), wird
        dieses ohne Maske benutzt - sonst das globale Zielbild mit der
        Gruppenmaske.  Ohne aktive Gruppe: globales Ziel, keine Maske.
        """

        active = self.train_methods.active_group

        group_target = self.train_methods.target_for(active)
        if group_target is not None:
            return group_target, None

        if active is not None:
            return target_image, self.train_methods.mask_for(active)

        return target_image, None

    def get_current_target_and_sim_ref(self, target_image, simulated_image):
        active = self.train_methods.active_group

        if active is not None:
            mask_img = self.train_methods.mask_for(active)
        else:  # Main Training
            mask_img = self._union_mask()

        if mask_img is None:
            sim_ref = simulated_image.flatten()
            tgt_ref = target_image.flatten()
        else:
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

        if self.train_methods.train_normalisation_parameters.enabled:
            norm_params = self.tunable_parameters.normalisation_parameters
        else:
            norm_params = self.tunable_parameters.normalisation_parameters.detach()


        if method == "z":
            # Die Korrekturen wirken RELATIV zur Streuung des Bildes. Additiv
            # ginge das nur, solange das Bild von der Groessenordnung 1 ist:
            # ein HAADF-Bild steht in Querschnittseinheiten (std ~ 1e-6), da
            # wuerde ein Offset von 1e-4 den Nenner ins Negative kippen und den
            # Loss sprengen. Bei norm_params = 0 ist es die uebliche
            # z-Normierung, unveraendert.
            sim_mean_raw = torch.mean(sim_ref)
            sim_std_raw = torch.std(sim_ref)
            sim_mean = sim_mean_raw + norm_params[0] * sim_std_raw
            sim_std = sim_std_raw * torch.clamp(1.0 + norm_params[1], min=0.1)
            tgt_mean = torch.mean(tgt_ref)
            tgt_std = torch.std(tgt_ref)
            # geklemmt statt "+ eps": ein additives eps waere bei einer
            # Streuung von 7e-7 schon ein Fehler von gut einem Prozent.
            simulated_image = (simulated_image - sim_mean) / torch.clamp(sim_std, min=eps)
            target_image = (target_image - tgt_mean) / torch.clamp(tgt_std, min=eps)

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
