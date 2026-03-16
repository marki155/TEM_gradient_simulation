from dataclasses import dataclass



@dataclass
class TrainState:
    running: bool = False
    paused: bool = False
    step: int = 0
    loss: float = 0.0

@dataclass
class TrainMethod:
    intensity_classes: str = "free"   # "free", "vacancies", "classes"
    train_blur: bool = False

    learning_rate: float = 0.01

    train_graphen_intensities: bool = False
    train_S1_intensities: bool = False
    train_W_intensities: bool = False
    train_S2_intensities: bool = False

    train_graphen_occupancies: bool = False
    train_S1_occupancies: bool = False
    train_W_occupancies: bool = False
    train_S2_occupancies: bool = False

    graphen_shift: bool = True
    graphen_rotation: bool = False
    graphen_free_positions: bool = False
    graphen_scale: bool = False

    WS2_shift: bool = True
    WS2_rotation: bool = True
    WS2_free_positions: bool = False
    WS2_scale: bool = False


    train_ctf_aberrations: bool = False
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

    train_multislice_sigma: bool = False

    only_train_small_area: int = 0 # TODO


    #train_amp_contrast: bool = False # TODO

    train_small_area_squares = [(500, 600, 1000, 1000), (0, 0, 300, 300)]