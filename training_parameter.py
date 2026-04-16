from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class TrainState:
    running: bool = False
    paused: bool = False
    step: int = 0
    loss: float = 0.0

@dataclass
class TrainMethodOLD:

    loss_mse: float = 0.0
    loss_ncc: float = .5
    loss_local_ncc_100: float = 0.
    loss_local_ncc_50: float = 0.
    loss_local_ncc_20: float = 0.
    loss_ssim: float = 0.0
    loss_l1: float = 0.1
    loss_fft: float = 0.0



    train_amplitude_contrast: bool = False

    learning_rate: float = 0.00001

    train_graphen_intensities: bool = False
    train_S1_intensities: bool = False
    train_W_intensities: bool = False
    train_S2_intensities: bool = False

    train_graphen_occupancies: bool = False
    train_S1_occupancies: bool = False
    train_W_occupancies: bool = False
    train_S2_occupancies: bool = False

    WS2_strain_field: bool = False

    graphen_shift: bool = True
    graphen_rotation: bool = True
    graphen_free_positions: bool = False
    graphen_scale: bool = True

    WS2_shift: bool = False
    WS2_rotation: bool = False
    WS2_free_positions: bool = False
    WS2_scale: bool = False


    train_ctf_aberrations: bool = True
    train_aberrations = {"C10": True,
                   'C12': True,
                   'C23': True,
                   'C21': True,
                   "C30": True,
                   'C34': True,
                   'C45': True}  # um
    train_env_parameters: bool = False
    train_env_params = {
        'source_size': False,
        'Cc': True,
        'deltaE': True,
        'instab_defocus_nm': False
    }

    train_normalisation_parameters: bool = True

    only_train_small_area: int = 1 # TODO



    #train_small_area_squares = [(50, 50, 350, 350), (500, 600, 1000, 1000)]
    train_small_area_squares = [(50, 50, 350, 350), (600, 600, 1000, 1000)]

@dataclass
class TrainParam:
    enabled: bool = False
    lr: float = 1e-4




@dataclass
class TrainMethod:
    area_masks: List
    target_gr: Optional[torch.Tensor] | None

    loss_var: float = 0.5
    loss_local_var_100: float = 0.0
    loss_mse: float = 0.0
    loss_ncc: float = 0.5
    loss_local_ncc_100: float = 0.0
    loss_ssim: float = 0.0
    loss_l1: float = 0.1

    occ_entropy_a: float = 1.0
    position_shifts_regularization_a: float = 1.0
    strain_energy_a: float = 0.0

    train_amplitude_contrast: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-5))

    train_graphen_intensities: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))
    train_S1_intensities: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))
    train_W_intensities: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))
    train_S2_intensities: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))

    train_graphen_occupancies: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))
    train_S1_occupancies: TrainParam = field(default_factory=lambda: TrainParam(False, .1))
    train_W_occupancies: TrainParam = field(default_factory=lambda: TrainParam(False, .1))
    train_S2_occupancies: TrainParam = field(default_factory=lambda: TrainParam(False, .1))

    Gr_strain_field: TrainParam = field(default_factory=lambda: TrainParam(False, .1))
    WS2_strain_field: TrainParam = field(default_factory=lambda: TrainParam(False, .1))

    graphen_shift: TrainParam = field(default_factory=lambda: TrainParam(True, 1e-5))
    graphen_rotation: TrainParam = field(default_factory=lambda: TrainParam(True, 1e-5))
    graphen_free_positions: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))
    graphen_scale: TrainParam = field(default_factory=lambda: TrainParam(True, 1e-5))

    WS2_shift: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-5))
    WS2_rotation: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-5))
    WS2_free_positions: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))
    WS2_scale: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-5))

    train_ctf_aberrations: TrainParam = field(default_factory=lambda: TrainParam(True, 1e-5))
    train_aberrations = {"C10": True,
                         'C12': True,
                         'C23': True,
                         'C21': True,
                         "C30": True,
                         'C34': True,
                         'C45': True}  # um
    train_env_parameters: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-5))
    train_env_params = {
        'source_size': False,
        'Cc': True,
        'deltaE': True,
        'instab_defocus_nm': False
    }
    train_normalisation_parameters: TrainParam = field(default_factory=lambda: TrainParam(True, 1e-5))

    only_train_small_area: int = 1
    #train_small_area_squares = [(50, 50, 350, 350), (600, 600, 1000, 1000)]
