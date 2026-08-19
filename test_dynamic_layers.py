"""Headless-Smoketest fuer das dynamische Schichtsystem.

Prueft ohne GUI:
  * Modell mit genau einer Graphen-Schicht
  * Modell mit mehreren Gruppen/Schichten
  * Isolation (``isolate_when_active``)
  * Optimizer-Gruppen und Gradienten folgen den TrainMethod-Schaltern
  * Migration alter config.json / train_method.json / Parameter-Dateien

Aufruf:  python3 test_dynamic_layers.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multislice

multislice.CHECK_PHASE = False

from layers import LayerConfig, LayerGroupConfig, graphene_only
from nn import CONFIG_CLASS_REGISTRY, NN_particles, ParticlesConfig
from ctf import (
    cartesian_to_polar_aberrations,
    polar_to_cartesian_aberrations,
    zernike_phase,
)
from potential import is_legacy_potential_parameters, migrate_serialized_potential_parameters
from training_parameter import TrainMethod, load_train_method

#: potential.py setzt beim Import das Default-Device; hier dasselbe benutzen,
#: sonst liegen Modell und Testdaten auf verschiedenen Geraeten
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ABERRATIONS = {"C10": -13., 'C12': 1.0, "phi12": 0., 'C23': 2., 'phi23': 0.,
               'C21': 2., 'phi21': 0., "C30": 4., 'C34': .02, 'phi34': 0.,
               'C45': .05, 'phi45': 0.}
ENV_PARAMS = {'source_size': 0.1e-3, 'Cc_um': 1.e3, 'deltaE': .35,
              'instab_defocus_nm': .01, 'use_detector': False}


def make_config(layers, groups, shape=(96, 96), n_atoms=40):
    rng = np.random.default_rng(0)
    return ParticlesConfig(
        init_positions=[rng.uniform(10, shape[0] - 10, size=(n_atoms, 2)) for _ in layers],
        init_intensities=[np.full(n_atoms, 6.0) for _ in layers],
        shape=shape, pixel_size_nm=0.0052, voltage_kV=80,
        init_aberrations=ABERRATIONS, init_ctf_env_params=ENV_PARAMS,
        init_amp_contrast=0.1, ctf_alpha_max_mrad=50., padding=8,
        layers=layers, groups=groups,
    )


def build_model(config, train_method):
    model = NN_particles(config=config, pixel_size_nm=config.pixel_size_nm,
                         voltage_kV=config.voltage_kV, train_method=train_method,
                         device=DEVICE, train_methods=train_method)
    model.build_multislice()
    return model


def step_and_report(model, train_method, target=None):
    """Ein Trainingsschritt; liefert die Namen der Parameter mit Gradient."""

    config = model.config
    pot_layer = model.model[0]
    if target is None:
        target = torch.rand(config.shape, dtype=torch.float32)

    optimizer = model.build_optimizer()
    _sim, _tgt, _p_diff, loss = model.loss_multislice(target)
    loss.backward()
    pot_layer.zero_grads_for_frozen()

    with_grad = sorted(
        group["name"] for group in optimizer.param_groups
        if any(p.grad is not None for p in group["params"])
    )
    optimizer.step()
    return float(loss), with_grad


def test_single_graphene_layer():
    layers, groups = graphene_only(max_shift=20.0, max_rotation=0.35)
    config = make_config(layers, groups)

    tm = TrainMethod().ensure_for_config(config)
    tm.set_active_group("Gr")
    tm.group("Gr").shift.enabled = True
    tm.group("Gr").rotation.enabled = True

    model = build_model(config, tm)
    loss, with_grad = step_and_report(model, tm)

    assert config.layer_names == ["Gr"], config.layer_names
    assert len(model.model[0].get_potential_images()) == 1
    assert "groups.Gr.shift" in with_grad and "groups.Gr.rotation" in with_grad
    assert "groups.Gr.scale" not in with_grad
    print(f"  nur Graphen: loss={loss:.4f}, Gradienten fuer {with_grad}")


def test_multiple_groups_and_isolation():
    layers = [
        LayerConfig("Gr", "C", 0.0, "Gr", init_occupancy=100.0, occ_entropy=False),
        LayerConfig("S1", "S", 0.33, "WS2"),
        LayerConfig("W", "W", 0.488, "WS2"),
        LayerConfig("S2", "S", 0.646, "WS2"),
        LayerConfig("Gr2", "C", 1.0, "Gr2", init_occupancy=100.0, occ_entropy=False),
    ]
    groups = [
        LayerGroupConfig("Gr", max_shift=20., max_rotation=0.35, isolate_when_active=True),
        LayerGroupConfig("WS2", max_shift=20., max_rotation=0.09),
        LayerGroupConfig("Gr2", max_shift=20., max_rotation=0.35, isolate_when_active=True),
    ]
    config = make_config(layers, groups)

    tm = TrainMethod().ensure_for_config(config)
    tm.set_active_group("Gr2")
    tm.group("Gr2").shift.enabled = True
    tm.layer("S1").occupancies.enabled = True

    model = build_model(config, tm)
    pot_layer = model.model[0]

    maxima = [float(np.abs(img).max()) for img in pot_layer.get_potential_images()]
    for layer, value in zip(config.layers, maxima):
        if layer.group == "Gr2":
            assert value > 0, f"{layer.name} sollte simuliert werden"
        else:
            assert value == 0, f"{layer.name} sollte bei isolierter Gruppe Null sein"

    tm.set_active_group(None)
    assert all(np.abs(img).max() > 0 for img in pot_layer.get_potential_images())

    tm.set_active_group("Gr2")
    pot_layer.apply_trainable_selection()
    loss, with_grad = step_and_report(model, tm)
    assert with_grad == sorted(["groups.Gr2.shift", "layers.S1.occupancies",
                                "train_ctf_aberrations",
                                "train_normalisation_parameters"]), with_grad
    print(f"  5 Schichten / 3 Gruppen: loss={loss:.4f}, Gradienten fuer {with_grad}")


def test_atom_weights_use_intensities():
    """Gewicht = (I / I_start) * sigmoid(occ): Start unveraendert, I wirkt."""

    layers, groups = graphene_only(max_shift=20.0, max_rotation=0.35)
    config = make_config(layers, groups)
    tm = TrainMethod().ensure_for_config(config)

    model = build_model(config, tm)
    pot_layer = model.model[0]

    # Startzustand: relative Intensitaet ist exakt 1, Gewicht also die Occupancy
    weights = pot_layer.get_atom_weights()[0]
    occupancy = pot_layer.tunable_parameters.occupancies[0].sigmoid()
    assert torch.allclose(weights, occupancy), "Startgewicht muss die Occupancy sein"

    potential_before = pot_layer.get_potential_images()[0].copy()

    # doppelte Intensitaet -> doppeltes Potential
    with torch.no_grad():
        pot_layer.tunable_parameters.intensities[0].mul_(2.0)
    potential_after = pot_layer.get_potential_images()[0]
    ratio = np.abs(potential_after).max() / max(np.abs(potential_before).max(), 1e-12)
    assert abs(ratio - 2.0) < 1e-3, ratio
    with torch.no_grad():
        pot_layer.tunable_parameters.intensities[0].div_(2.0)

    # freigegeben -> Gradient
    tm.layer("Gr").intensities.enabled = True
    pot_layer.apply_trainable_selection()
    _loss, with_grad = step_and_report(model, tm)
    assert "layers.Gr.intensities" in with_grad, with_grad
    print("  Intensitaeten wirken aufs Potential und bekommen Gradienten")


def test_element_mixture():
    """Intensitaet als kontinuierliche Ordnungszahl: I=7 ist wirklich Stickstoff."""

    elements = ["B", "C", "N", "O"]
    group = LayerGroupConfig("Gr", max_shift=20., max_rotation=0.35, isolate_when_active=True)

    def build(layer_elements, element="C"):
        layer = LayerConfig("Gr", element, 0.0, "Gr", init_occupancy=100.0,
                            occ_entropy=False, elements=list(layer_elements))
        config = make_config([layer], [group])
        tm = TrainMethod().ensure_for_config(config)
        return build_model(config, tm), tm

    (mix_model, mix_tm) = build(elements)
    mix_pot = mix_model.model[0]
    assert mix_pot.layer_elements[0] == elements, mix_pot.layer_elements[0]

    def potential(intensity):
        with torch.no_grad():
            mix_pot.tunable_parameters.intensities[0].fill_(intensity)
        return mix_pot.get_potential_images()[0]

    # reine Elemente als Referenz
    reference = {}
    for symbol in elements:
        model, _tm = build([], element=symbol)
        pot = model.model[0]
        with torch.no_grad():
            pot.tunable_parameters.intensities[0].fill_(float(pot.get_init_intensities(0)[0]))
        reference[symbol] = pot.get_potential_images()[0]

    def deviation(a, b):
        # bis auf einen konstanten Offset vergleichen: das Potential wird um
        # seinen Median zentriert, und der Median ist nicht linear. Ein
        # konstanter Potentialoffset ist eine globale Phase und damit im Bild
        # ohnehin nicht sichtbar.
        difference = a - b
        return np.abs(difference - difference.mean()).max() / np.abs(b).max()

    for intensity, symbol in ((5.0, "B"), (6.0, "C"), (7.0, "N"), (8.0, "O")):
        assert deviation(potential(intensity), reference[symbol]) < 1e-5, symbol

    # dazwischen linear gemischt
    for intensity, share in ((6.5, 0.5), (6.3, 0.3)):
        expected = (1 - share) * reference["C"] + share * reference["N"]
        assert deviation(potential(intensity), expected) < 1e-5, intensity

    # Gradient auch genau auf einem Stuetzpunkt (Knick der Zeltfunktion)
    with torch.no_grad():
        mix_pot.tunable_parameters.intensities[0].fill_(6.0)
    mix_tm.layer("Gr").intensities.enabled = True
    mix_pot.apply_trainable_selection()
    _loss, with_grad = step_and_report(mix_model, mix_tm)
    assert "layers.Gr.intensities" in with_grad, with_grad

    print("  Elementmischung: I=5/6/7/8 exakt B/C/N/O, dazwischen linear, Gradient laeuft")


def test_isolated_group_skips_empty_slices():
    """Uebersprungene Schichten aendern die Welle nicht (nur die Rechenzeit)."""

    layers = [
        LayerConfig("Gr", "C", 0.0, "Gr", init_occupancy=100.0, occ_entropy=False),
        LayerConfig("S1", "S", 0.33, "WS2"),
        LayerConfig("W", "W", 0.488, "WS2"),
        LayerConfig("S2", "S", 0.646, "WS2"),
    ]
    groups = [
        LayerGroupConfig("Gr", max_shift=20., max_rotation=0.35, isolate_when_active=True),
        LayerGroupConfig("WS2", max_shift=20., max_rotation=0.09),
    ]
    config = make_config(layers, groups)
    tm = TrainMethod().ensure_for_config(config)
    tm.set_active_group("Gr")

    model = build_model(config, tm)
    potentials, aux = model.model[0](None)
    assert [p is None for p in potentials] == [False, True, True, True], \
        "isolierte Gruppe: nur die eigene Schicht darf ein Potential liefern"

    wave_skipped = model.model[1]((potentials, aux))[0]

    # Vergleich mit dem alten Weg: explizite Nullbilder statt None
    zeros = torch.zeros_like(potentials[0])
    filled = [p if p is not None else zeros for p in potentials]
    wave_zeros = model.model[1]((filled, aux))[0]

    difference = float((wave_skipped - wave_zeros).abs().max())
    scale = float(wave_zeros.abs().max())
    assert difference < 1e-4 * max(scale, 1.0), (difference, scale)
    print(f"  isolierte Gruppe: Welle identisch (max. Abweichung {difference:.2e})")


def test_ctf_aberration_switches():
    """CTF: Betraege UND Winkel werden mitoptimiert, Startwerte aus der Config."""

    aberrations = dict(ABERRATIONS)
    aberrations.update({"phi12": 0.3, "phi21": 0.1, "phi23": 0.2, "phi34": 0.4, "phi45": 0.5})

    layers, groups = graphene_only(max_shift=20.0, max_rotation=0.35)
    config = make_config(layers, groups)
    config.init_aberrations = aberrations
    config.init_amp_contrast = 0.1

    tm = TrainMethod().ensure_for_config(config)
    tm.train_ctf_aberrations.enabled = True

    model = build_model(config, tm)
    ctf = model.model[2]

    # Startwerte kommen aus der Config (nicht aus den Defaults von make_param_dict).
    # Gespeichert wird kartesisch, zum Vergleich zurueckrechnen.
    stored = cartesian_to_polar_aberrations(
        {k: float(v) for k, v in ctf.tunable_parameters.aberrations.items()})
    for key, value in aberrations.items():
        assert abs(stored[key] - value) < 1e-5, (key, stored[key], value)
    assert abs(float(ctf.tunable_parameters.amp_contrast) - 0.1) < 1e-6

    step_and_report(model, tm)
    # die Winkel-Freiheitsgrade (kartesisch: die b-Komponenten) muessen laufen
    angles = {k: v for k, v in ctf.tunable_parameters.aberrations.items()
              if k.startswith("phi") or (k[0] == "b" and k[1:].isdigit())}
    assert angles, "Testkonfiguration ohne Winkel"
    for key, param in angles.items():
        assert param.grad is not None and float(param.grad) != 0.0, f"{key} bekommt keinen Gradienten"

    # einzelne Ordnung abschalten -> Betrag und Winkel zusammen
    tm.train_aberrations["C12"] = False
    for key in ("C12", "phi12", "a12", "b12"):
        assert not ctf.aberration_is_trainable(key), key
    assert ctf.aberration_is_trainable("b23")
    tm.train_aberrations["C12"] = True

    # CTF komplett aus
    tm.train_ctf_aberrations.enabled = False
    for key in ("C10", "phi12", "a12", "b23"):
        assert not ctf.aberration_is_trainable(key), key
    print("  CTF: Winkel bekommen Gradienten, Startwerte aus der Config uebernommen")


def test_ctf_parametrizations_are_equivalent():
    """polar (C, phi) und kartesisch (a, b) beschreiben dieselbe Phase."""

    polar = {"C10": -13.0, "C12": 1.0, "phi12": 0.3, "C21": 2.0, "phi21": 0.1,
             "C23": 2.0, "phi23": 0.2, "C30": 4.0, "C34": 0.02, "phi34": 0.4,
             "C45": 0.05, "phi45": 0.5}
    cartesian = polar_to_cartesian_aberrations(polar)

    # Hin- und Rueckrechnung
    back = cartesian_to_polar_aberrations(cartesian)
    for key, value in polar.items():
        assert abs(back[key] - value) < 1e-6, (key, back[key], value)

    torch.manual_seed(0)
    alpha = torch.rand(64, 64) * 0.05
    phi = (torch.rand(64, 64) - 0.5) * 2 * np.pi

    phase_polar = zernike_phase(alpha, phi, {k: torch.tensor(v) for k, v in polar.items()})
    phase_cart = zernike_phase(alpha, phi, {k: torch.tensor(v) for k, v in cartesian.items()})
    difference = float((phase_polar - phase_cart).abs().max())
    assert difference < 1e-6, difference

    # beide Schreibweisen im vollen Modell trainierbar
    for parametrization, angle_keys in (("polar", ["phi12", "phi23"]),
                                        ("cartesian", ["a12", "b12", "a23", "b23"])):
        layers, groups = graphene_only(max_shift=20.0, max_rotation=0.35)
        config = make_config(layers, groups)
        config.init_aberrations = polar
        config.ctf_parametrization = parametrization

        tm = TrainMethod().ensure_for_config(config)
        tm.train_ctf_aberrations.enabled = True

        model = build_model(config, tm)
        step_and_report(model, tm)
        aberrations = model.model[2].tunable_parameters.aberrations
        for key in angle_keys:
            assert key in aberrations, (parametrization, key)
            assert aberrations[key].grad is not None, (parametrization, key)

    print(f"  polar == kartesisch (max. Phasendifferenz {difference:.2e}), beide trainierbar")


def test_legacy_migration(dataset="platelet_11"):
    config_file = os.path.join(dataset, "config", "config.json")
    if not os.path.exists(config_file):
        print(f"  uebersprungen (kein {config_file})")
        return

    from mains.main import deserialize

    with open(config_file) as f:
        config = deserialize(json.load(f), CONFIG_CLASS_REGISTRY)

    assert config.layer_names == ["Gr", "S1", "W", "S2"], config.layer_names
    assert config.group_names == ["Gr", "WS2"], config.group_names
    assert config.group("Gr").isolate_when_active

    tm_file = os.path.join(dataset, "config", "train_method.json")
    if os.path.exists(tm_file):
        tm = load_train_method(tm_file, config)
        assert set(tm.groups) == {"Gr", "WS2"}
        assert set(tm.layers) == {"Gr", "S1", "W", "S2"}
        print(f"  train_method migriert: aktive Gruppe={tm.active_group}, Masken={list(tm.masks)}")

    param_file = os.path.join(dataset, "best.json")
    if os.path.exists(param_file):
        with open(param_file) as f:
            pot_json = json.load(f)["potential_parameters"]
        assert is_legacy_potential_parameters(pot_json)
        migrated = migrate_serialized_potential_parameters(pot_json, config)
        assert set(migrated["scales"]) == {"Gr", "WS2"}
        assert set(migrated["strain_fields"]) == {"Gr", "WS2"}
        print(f"  Parameterdatei migriert: {sorted(migrated)}")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("test_single_graphene_layer")
    test_single_graphene_layer()
    print("test_multiple_groups_and_isolation")
    test_multiple_groups_and_isolation()
    print("test_atom_weights_use_intensities")
    test_atom_weights_use_intensities()
    print("test_element_mixture")
    test_element_mixture()
    print("test_isolated_group_skips_empty_slices")
    test_isolated_group_skips_empty_slices()
    print("test_ctf_aberration_switches")
    test_ctf_aberration_switches()
    print("test_ctf_parametrizations_are_equivalent")
    test_ctf_parametrizations_are_equivalent()
    print("test_legacy_migration")
    test_legacy_migration()
    print("\nalle Tests durch")
