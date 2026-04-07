from dataclasses import dataclass



@dataclass
class TrainState:
    running: bool = False
    paused: bool = False
    step: int = 0
    loss: float = 0.0

@dataclass
class TrainMethod:

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