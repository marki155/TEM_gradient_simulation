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
from potential import is_legacy_potential_parameters, migrate_serialized_potential_parameters
from training_parameter import TrainMethod, load_train_method

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
                         device="cpu", train_methods=train_method)
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
                                "train_ctf_aberrations", "train_ctf_angles",
                                "train_normalisation_parameters"]), with_grad
    print(f"  5 Schichten / 3 Gruppen: loss={loss:.4f}, Gradienten fuer {with_grad}")


def test_ctf_aberration_switches():
    """CTF: Betraege und Winkel folgen ihren Schaltern, Startwerte aus der Config."""

    aberrations = dict(ABERRATIONS)
    aberrations.update({"phi12": 0.3, "phi21": 0.1, "phi23": 0.2, "phi34": 0.4, "phi45": 0.5})

    layers, groups = graphene_only(max_shift=20.0, max_rotation=0.35)
    config = make_config(layers, groups)
    config.init_aberrations = aberrations
    config.init_amp_contrast = 0.1

    tm = TrainMethod().ensure_for_config(config)
    tm.train_ctf_aberrations.enabled = True
    tm.train_ctf_angles.enabled = True

    model = build_model(config, tm)
    ctf = model.model[2]

    # Startwerte kommen aus der Config (nicht aus den Defaults von make_param_dict)
    for key, value in aberrations.items():
        assert abs(float(ctf.tunable_parameters.aberrations[key]) - value) < 1e-6, key
    assert abs(float(ctf.tunable_parameters.amp_contrast) - 0.1) < 1e-6

    _loss, with_grad = step_and_report(model, tm)
    assert "train_ctf_angles" in with_grad, with_grad
    assert "train_ctf_aberrations" in with_grad, with_grad

    # Winkel abschalten -> nur noch die Betraege bekommen Gradienten
    tm.train_ctf_angles.enabled = False
    for p in ctf.tunable_parameters.aberrations.values():
        p.grad = None
    _loss, with_grad = step_and_report(model, tm)
    assert "train_ctf_angles" not in with_grad, with_grad
    assert "train_ctf_aberrations" in with_grad, with_grad

    # einzelne Ordnung abschalten (gilt fuer C12 und phi12 zusammen)
    tm.train_aberrations["C12"] = False
    assert not ctf.aberration_is_trainable("C12")
    assert not ctf.aberration_is_trainable("phi12")
    tm.train_aberrations["C12"] = True

    # alles aus -> auch die Winkel sind aus
    tm.train_ctf_aberrations.enabled = False
    tm.train_ctf_angles.enabled = True
    assert not ctf.aberration_is_trainable("phi12")
    print("  CTF-Winkel schaltbar, Startwerte aus der Config uebernommen")


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
    print("test_ctf_aberration_switches")
    test_ctf_aberration_switches()
    print("test_legacy_migration")
    test_legacy_migration()
    print("\nalle Tests durch")
