"""Zahl der Strain-Kontrollpunkte aendern - im Betrieb und beim Laden.

Das Kontrollgitter wird in normierten Koordinaten abgetastet, seine Groesse
ist also reine Aufloesung. Umrechnen heisst: die neuen Knoten bekommen den
Wert, den das bisherige Feld dort hat.

    python3 test_strain_grid.py
"""

import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import torch

from displacement_field import SplineDisplacementField2D, resample_control_grid
from layers import graphene_only
from nn import NN_particles, ParticlesConfig
from training_parameter import TrainMethod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)

FAILS = []


def check(name, condition, detail=""):
    print(f"  {name:<48} {'ok' if condition else 'FEHLER'} {detail}")
    if not condition:
        FAILS.append(name)


def smooth_control(shape, device=DEVICE):
    """Ein glattes Feld - so sieht ein gefittetes Strainfeld aus."""

    ys = torch.linspace(-1, 1, shape[0], device=device)
    xs = torch.linspace(-1, 1, shape[1], device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    control = torch.zeros((1, 2, shape[0], shape[1]), device=device)
    control[0, 0] = 1.5 * torch.sin(1.2 * xx) * torch.cos(0.8 * yy)
    control[0, 1] = 1.5 * torch.cos(1.0 * xx + 0.5 * yy)
    return control


def build_model(control_points=(10, 10), n_atoms=150, shape=(96, 96)):
    rng = np.random.default_rng(0)
    layers, groups = graphene_only()
    groups[0].control_points = list(control_points)
    config = ParticlesConfig(
        init_positions=[rng.uniform(8, shape[0] - 8, size=(n_atoms, 2)) for _ in layers],
        init_intensities=[np.full(n_atoms, 6.0) for _ in layers],
        shape=shape, pixel_size_nm=0.012, voltage_kV=80, padding=8,
        init_aberrations={"C10": -13.0, "C12": 1.0, "phi12": 0.0, "C30": 4.0},
        init_ctf_env_params={"source_size": 1e-4, "Cc_um": 1e3, "deltaE": 0.35,
                             "instab_defocus_nm": 0.01, "use_detector": False},
        init_amp_contrast=0.1, ctf_alpha_max_mrad=50.0,
        layers=layers, groups=groups,
    )
    train_method = TrainMethod().ensure_for_config(config)
    train_method.set_active_group("Gr")
    train_method.group("Gr").strain_field.enabled = True
    model = NN_particles(config=config, pixel_size_nm=config.pixel_size_nm,
                         voltage_kV=config.voltage_kV, train_method=train_method,
                         device=DEVICE, train_methods=train_method)
    model.build_multislice()
    return model, config, train_method


def test_field_survives():
    """Verfeinern darf das Feld praktisch nicht veraendern."""

    class Cfg:
        shape = (256, 256)
        padding = 16

    field = SplineDisplacementField2D(config=Cfg(), control_shape=(10, 10), device=DEVICE)
    with torch.no_grad():
        field.tunable_parameters.control.copy_(smooth_control((10, 10)))

    probe = field.make_coordinate_grid(H=128, W=128, pixel_centers=False)
    with torch.no_grad():
        before = field.sample(probe).clone()
    amplitude = float(before.abs().max())

    # an den neuen Knoten muss es exakt stimmen
    nodes = field.make_coordinate_grid(H=20, W=20, pixel_centers=False)
    with torch.no_grad():
        at_nodes = field.sample(nodes).clone()
    deviation = field.resample_control((20, 20))
    with torch.no_grad():
        at_nodes_after = field.sample(nodes)
    check("neue Knoten treffen das alte Feld exakt",
          float((at_nodes_after - at_nodes).abs().max()) < 1e-4,
          f"{float((at_nodes_after - at_nodes).abs().max()):.2e} px")
    check("Verfeinern aendert das Feld kaum",
          deviation / amplitude < 0.03, f"{deviation:.4f} px von {amplitude:.3f} px")
    check("Form uebernommen", tuple(field.tunable_parameters.control.shape) == (1, 2, 20, 20),
          str(tuple(field.tunable_parameters.control.shape)))

    field.resample_control((10, 10))
    with torch.no_grad():
        back = float((field.sample(probe) - before).abs().max())
    check("hin und zurueck bleibt beim Feld", back / amplitude < 0.02,
          f"{back:.4f} px = {back / amplitude * 100:.2f} %")

    coarse = SplineDisplacementField2D(config=Cfg(), control_shape=(10, 10), device=DEVICE)
    with torch.no_grad():
        coarse.tunable_parameters.control.copy_(smooth_control((10, 10)))
    lost = coarse.resample_control((4, 4))
    check("Vergroebern verliert sichtbar mehr", lost > deviation,
          f"{lost:.4f} px gegen {deviation:.4f} px beim Verfeinern")


def test_runtime_change():
    """Im laufenden Modell umstellen und weitertrainieren."""

    model, config, train_method = build_model((8, 8))
    potential = model.model[0]
    with torch.no_grad():
        potential.tunable_parameters.strain_fields["Gr"].control.copy_(smooth_control((8, 8)))
        target = model.model(None)[0].detach().clone()

    parameter = potential.tunable_parameters.strain_fields["Gr"].control
    before_image = model.model(None)[0].detach().clone()

    deviation = potential.set_control_points("Gr", (16, 16))
    check("Kontrollgitter umgestellt", potential.control_points("Gr") == [16, 16],
          str(potential.control_points("Gr")))
    check("Config zieht mit", config.group("Gr").control_points == [16, 16],
          str(config.group("Gr").control_points))
    check("Parameter behaelt seine Identitaet",
          potential.tunable_parameters.strain_fields["Gr"].control is parameter,
          f"Form jetzt {tuple(parameter.shape)}")

    after_image = model.model(None)[0].detach()
    relative = float((after_image - before_image).abs().max() / before_image.abs().max())
    check("simuliertes Bild bleibt praktisch gleich", relative < 0.05,
          f"{relative * 100:.2f} % Abweichung, Feld {deviation:.4f} px")

    optimizer = model.build_optimizer()
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        *_, loss = model.loss_multislice(target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    check("Training laeuft nach dem Umstellen weiter",
          parameter.grad is not None and np.all(np.isfinite(losses)),
          f"Loss {losses[0]:.5f} -> {losses[-1]:.5f}")


def test_save_and_load_across_sizes():
    """Mit 10x10 gespeichert, in ein 20x20-Modell geladen."""

    from parameter_frame import migrate_parameters_dict, serialize, update_from_serialized

    small, _, _ = build_model((10, 10))
    with torch.no_grad():
        small.model[0].tunable_parameters.strain_fields["Gr"].control.copy_(
            smooth_control((10, 10)))
    saved = json.loads(json.dumps(serialize(small.get_tunable_parameters())))

    big, _, _ = build_model((20, 20))
    target_params = big.get_tunable_parameters()

    try:
        update_from_serialized(target_params, saved)
        loaded_raw = True
    except Exception:
        loaded_raw = False
    check("ohne Umrechnen scheitert das Laden (wie erwartet)", not loaded_raw)

    migrated = migrate_parameters_dict(saved, target_params, big.config)
    update_from_serialized(target_params, migrated)
    control = target_params.potential_parameters.strain_fields["Gr"].control
    check("geladen und auf 20x20 gebracht", tuple(control.shape) == (1, 2, 20, 20),
          str(tuple(control.shape)))

    probe = small.model[0].strain_fields["Gr"].make_coordinate_grid(
        H=64, W=64, pixel_centers=False)
    with torch.no_grad():
        u_small = small.model[0].strain_fields["Gr"].sample(probe)
        u_big = big.model[0].strain_fields["Gr"].sample(probe)
    amplitude = float(u_small.abs().max())
    check("das geladene Feld ist dasselbe",
          float((u_big - u_small).abs().max()) / amplitude < 0.03,
          f"{float((u_big - u_small).abs().max()):.4f} px von {amplitude:.3f} px")


def test_gui_button():
    """Der Knopf in der GUI stellt um und baut den Optimizer neu."""

    from PySide6 import QtWidgets
    from gui_qt import LiveTrainWindow

    model, config, train_method = build_model((8, 8))
    with torch.no_grad():
        target = model.model(None)[0].detach().clone()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = LiveTrainWindow(model=model, target=target, pot_layer=model.model[0],
                             ctf_layer=model.imaging_layer, train_method=train_method,
                             plot_every=5, heavy_every=10, max_steps=10)
    window.resize(1200, 800)
    window.show()
    for _ in range(3):
        app.processEvents()

    spins = window.control_point_spins["Gr"]
    check("Eingabefelder je Gruppe da", len(spins) == 2 and spins[0].value() == 8,
          f"{[s.value() for s in spins]}")

    old_optimizer = window.optimizer
    spins[0].setValue(14)
    spins[1].setValue(14)
    window.apply_control_points_btn.click()
    for _ in range(3):
        app.processEvents()

    check("Knopf stellt das Gitter um", window.pot_layer.control_points("Gr") == [14, 14],
          str(window.pot_layer.control_points("Gr")))
    check("Optimizer neu gebaut", window.optimizer is not old_optimizer)

    editors = [e for e in window.parameter_frame._tensor_editors
               if "control" in getattr(e, "label_text", "")]
    check("Parameter-Editor zeigt die neue Form",
          bool(editors) and tuple(editors[0].tensor.shape) == (1, 2, 14, 14),
          str(tuple(editors[0].tensor.shape)) if editors else "kein Editor")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("Feld ueberlebt die Umrechnung")
    test_field_survives()
    print("im laufenden Modell umstellen")
    test_runtime_change()
    print("speichern und in anderer Groesse laden")
    test_save_and_load_across_sizes()
    print("GUI-Knopf")
    test_gui_button()
    print("\n" + ("alle Tests durch" if not FAILS else "FEHLER: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)
