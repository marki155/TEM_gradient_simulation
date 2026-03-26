import copy
import numpy as np
import torch

from nn import NN_particles
from training_parameter import TrainMethod
from gui_qt import launch_gui_qt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)

VOLTAGE_KV = 80


def make_train_method(image_shape):
    tm = TrainMethod()

    H, W = image_shape

    # Wichtig: in deinem Code werden zwei Trainingsbereiche erwartet
    # Index 0 -> Graphen
    # Index 1 -> WS2
    tm.train_small_area_squares = [
        (0, 0, H, W),
        (0, 0, H, W),
    ]
    tm.only_train_small_area = -1

    # Nur Positionen trainieren
    tm.graphen_free_positions = True
    tm.WS2_free_positions = True

    tm.graphen_shift = False
    tm.graphen_rotation = False
    tm.graphen_scale = False

    tm.WS2_shift = False
    tm.WS2_rotation = False
    tm.WS2_scale = False
    tm.WS2_strain_field = False

    # Intensitäten/Belegungen fixieren
    tm.train_graphen_intensities = False
    tm.train_S1_intensities = False
    tm.train_W_intensities = False
    tm.train_S2_intensities = False

    tm.train_graphen_occupancies = False
    tm.train_S1_occupancies = False
    tm.train_W_occupancies = False
    tm.train_S2_occupancies = False

    tm.train_blur = False

    # Robuste einfache Loss-Konfiguration
    tm.loss_mse = 1.0
    tm.loss_ncc = 0.0
    tm.loss_local_ncc_100 = 0.0
    tm.loss_local_ncc_50 = 0.0
    tm.loss_local_ncc_20 = 0.0
    tm.loss_ssim = 0.0
    tm.loss_l1 = 0.0
    tm.loss_fft = 0.0

    return tm


def make_test_particles():
    """
    7 Partikel insgesamt:
      Gr: 1
      S1: 2
      W : 2
      S2: 2
    """
    init_positions = [
        np.array([[29.5, 29.5],
            [41.5, 25.5],
        ], dtype=np.float32),  # Gr

        np.array([[29.5, 29.5],
            [41.5, 25.5],

        ], dtype=np.float32),  # S1

        np.array([
            [10,20],
            [30,40],
        ], dtype=np.float32),  # W

        np.array([[29.5, 29.5],
            [41.5, 25.5],

        ], dtype=np.float32),  # S2
    ]

    init_intensities = [
        np.array([0.0,0.0], dtype=np.float32),          # C
        np.array([0.0,0.0], dtype=np.float32),   # S
        np.array([74.0, 74.0], dtype=np.float32),   # W
        np.array([0.0,0.0], dtype=np.float32),   # S
    ]

    return init_positions, init_intensities


@torch.no_grad()
def create_target_image(image_shape, fov, train_method, init_positions, init_intensities):
    """
    Erzeugt mit einem Referenzmodell ein synthetisches Zielbild.
    """
    ref_model = NN_particles(
        image_shape=image_shape,
        fov=fov,
        voltage_kV=VOLTAGE_KV,
        train_method=train_method,
        device=DEVICE,
        train_methods=train_method
    )

    ref_model.build_multislice(
        init_positions=copy.deepcopy(init_positions),
        init_intensities=copy.deepcopy(init_intensities),
        dz=0.3,
        max_shifts=(0.0, 0.0),
        max_rotations=(0.0, 0.0)
    )

    simulated_image, _ = ref_model.model(None)

    # So normalisieren, dass es zur restlichen Pipeline passt
    min_val = torch.min(simulated_image)
    max_val = torch.max(simulated_image)
    target = (simulated_image - min_val) / (max_val - min_val + 1e-8)

    return target.detach()


@torch.no_grad()
def perturb_positions(model, sigma_px=1.5):
    pot_layer = model.model[0]
    for p in pot_layer.tunable_parameters.positions:
        p.add_(torch.randn_like(p) * sigma_px)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    image_shape = (64, 64)
    fov = 4.0

    init_positions, init_intensities = make_test_particles()
    train_method = make_train_method(image_shape)

    # 1) Synthetisches Zielbild aus idealen Positionen erzeugen
    target = create_target_image(
        image_shape=image_shape,
        fov=fov,
        train_method=train_method,
        init_positions=init_positions,
        init_intensities=init_intensities
    )

    brightness = torch.mean(target)
    contrast = torch.std(target)

    # 2) Trainingsmodell mit denselben Partikeln aufbauen
    model = NN_particles(
        image_shape=image_shape,
        fov=fov,
        voltage_kV=VOLTAGE_KV,
        train_method=train_method,
        device=DEVICE,
        train_methods=train_method
    )

    model.build_multislice(
        init_positions=copy.deepcopy(init_positions),
        init_intensities=copy.deepcopy(init_intensities),
        dz=0.3,
        max_shifts=(0.0, 0.0),
        max_rotations=(0.0, 0.0)
    )

    # 3) Partikel leicht verschieben -> das ist der Startzustand fürs Training
    perturb_positions(model, sigma_px=1.5)

    ctf_layer = model.model[2]
    pot_layer = model.model[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    # 4) GUI starten: dort sollte man sehen, wie die Partikel zurücklaufen
    launch_gui_qt(
        model=model,
        optimizer=optimizer,
        target=target,
        brightness=brightness,
        contrast=contrast,
        pot_layer=pot_layer,
        ctf_layer=ctf_layer,
        train_method=train_method,
        plot_every=10,
        heavy_every=100,
        max_steps=5000,
    )