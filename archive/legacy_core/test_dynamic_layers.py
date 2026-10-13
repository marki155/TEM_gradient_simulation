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
    build_ctf_envelope_on_k,
    cartesian_to_polar_aberrations,
    polar_to_cartesian_aberrations,
    zernike_phase,
)
from potential import is_legacy_potential_parameters, migrate_serialized_potential_parameters
from stem import haadf_cross_section
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


def test_debye_waller():
    """Thermische Verschmierung: Spitze runter, Flaechenintegral unveraendert."""

    group = LayerGroupConfig("Gr", max_shift=20., max_rotation=0.35)

    def potential(u_rms):
        layer = LayerConfig("Gr", "C", 0.0, "Gr", init_occupancy=100.0,
                            occ_entropy=False, debye_waller_u=u_rms)
        config = make_config([layer], [group], n_atoms=1)
        # genau auf eine Pixelmitte setzen, sonst verteilt schon die bilineare
        # Rasterung die Spitze auf vier Pixel
        config.init_positions = [np.array([[48.0, 48.0]], dtype=np.float64)]
        tm = TrainMethod().ensure_for_config(config)
        model = build_model(config, tm)
        return model.model[0].get_potential_images()[0]

    static = potential(0.0)
    smeared = potential(0.06)

    assert smeared.max() < 0.6 * static.max(), (static.max(), smeared.max())

    # eine normierte Faltung erhaelt das Flaechenintegral
    relative = abs(smeared.sum() - static.sum()) / abs(static.sum())
    assert relative < 1e-3, relative

    # u = 0 aendert nichts
    assert np.abs(potential(0.0) - static).max() == 0.0

    print(f"  Debye-Waller: Spitze {static.max():.1f} -> {smeared.max():.1f}, "
          f"Integral gleich (Abweichung {relative:.1e})")


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


def test_contrast_factor():
    """Der Kontrastfaktor ist ein eigener, trainierbarer Gruppenparameter."""

    layers, groups = graphene_only(max_shift=20.0, max_rotation=0.35)
    config = make_config(layers, groups)
    tm = TrainMethod().ensure_for_config(config)

    model = build_model(config, tm)
    pot_layer = model.model[0]
    contrast = pot_layer.tunable_parameters.contrast_factors["Gr"]

    assert abs(float(contrast) - 1.0) < 1e-9, "Start ist das physikalische Potential"

    before = np.abs(pot_layer.get_potential_images()[0]).max()
    with torch.no_grad():
        contrast.fill_(0.25)
    after = np.abs(pot_layer.get_potential_images()[0]).max()
    assert abs(after / before - 0.25) < 1e-3, (before, after)
    with torch.no_grad():
        contrast.fill_(1.0)

    tm.group("Gr").contrast.enabled = True
    pot_layer.apply_trainable_selection()
    _loss, with_grad = step_and_report(model, tm)
    assert "groups.Gr.contrast" in with_grad, with_grad

    print("  Kontrastfaktor: skaliert das Potential, eigener Trainingsschalter")


def test_spatial_coherence_envelope():
    """Raeumliche Kohaerenz daempft ueber den Gradienten der Aberrationsfunktion."""

    lam = 0.00418
    k = torch.linspace(0.01, 12.0, 200).reshape(-1, 1)
    zeros = torch.zeros_like(k)
    defocus = torch.tensor(-13.0)
    cs_nm = torch.tensor(4000.0)

    def envelope(alpha_mrad):
        params = {"source_size": alpha_mrad * 1e-3,
                  "use_chromatic": False, "use_instability": False}
        return build_ctf_envelope_on_k(k, zeros, 80, lam, params, cache={},
                                       defocus_nm=defocus, cs_nm=cs_nm)

    # kleine Quelle -> praktisch keine Daempfung
    assert float(envelope(0.1).min()) > 0.99

    # grosse Quelle -> deutliche Daempfung bei hohen Frequenzen
    strong = envelope(5.0)
    assert float(strong.min()) < 0.5, float(strong.min())

    # ohne Quelle gar keine
    assert torch.allclose(envelope(0.0), torch.ones_like(k))

    # und sie haengt am Defokus (frueher gar nicht)
    other = build_ctf_envelope_on_k(
        k, zeros, 80, lam, {"source_size": 5e-3, "use_chromatic": False,
                            "use_instability": False},
        cache={}, defocus_nm=torch.tensor(-60.0), cs_nm=cs_nm)
    assert float(other.min()) < float(strong.min()), "mehr Defokus muss staerker daempfen"

    print(f"  Kohaerenz-Envelope: 0.1 mrad -> {float(envelope(0.1).min()):.4f}, "
          f"5 mrad -> {float(strong.min()):.4f}")


def test_renamed_config_fields():
    """Configs von vor der Umbenennung muessen weiter laden."""

    from layers import coerce_groups

    stored = {"name": "Gr", "control_points": [10, 10], "max_shift": 49.3,
              "max_rotation": 0.349, "init_shift": [-9.5, -11.0], "init_rotation": 0.025,
              "init_scale": [1.12, 1.14], "init_potential_factor": 0.7,   # alter Feldname
              "mask_file": None, "target_file": None, "isolate_when_active": True}

    group = coerce_groups([stored])[0]
    assert abs(group.init_contrast_factor - 0.7) < 1e-12, group.init_contrast_factor

    # echter Tippfehler soll weiterhin auffliegen
    try:
        coerce_groups([{**stored, "init_kontrast": 1.0}])
    except KeyError:
        pass
    else:
        raise AssertionError("unbekannte Felder muessen weiterhin einen Fehler geben")

    print("  alter Feldname init_potential_factor wird uebernommen")


def test_haadf_cross_sections():
    """Der HAADF-Querschnitt muss mit Z steigen, ungefaehr wie Z^1.7."""

    elements = {"B": 5, "C": 6, "N": 7, "O": 8, "Si": 14, "S": 16, "W": 74}
    sigma = {el: haadf_cross_section(el, 60., 200., 80., u_rms_A=0.06)
             for el in elements}

    ordered = ["B", "C", "N", "O"]
    for lighter, heavier in zip(ordered, ordered[1:]):
        assert sigma[heavier] > sigma[lighter], (lighter, heavier, sigma)

    numbers = np.array(list(elements.values()), dtype=float)
    values = np.array([sigma[el] for el in elements])
    exponent = np.polyfit(np.log(numbers), np.log(values), 1)[0]
    assert 1.5 < exponent < 2.0, exponent

    # groesserer Innenwinkel -> staerkere Z-Abhaengigkeit
    inner_100 = np.array([haadf_cross_section(el, 100., 300., 80., 0.06) for el in elements])
    exponent_100 = np.polyfit(np.log(numbers), np.log(inner_100), 1)[0]
    assert exponent_100 > exponent, (exponent, exponent_100)

    print(f"  HAADF-Querschnitte: sigma ~ Z^{exponent:.2f} (60-200 mrad), "
          f"Z^{exponent_100:.2f} (100-300 mrad)")


def test_stem_imaging():
    """STEM-Zweig: normierte Sonde, Z-Kontrast in der richtigen Richtung, Gradienten."""

    layers, groups = graphene_only(max_shift=20.0, max_rotation=0.35)
    layers[0].elements = ["B", "C", "N", "O"]
    layers[0].debye_waller_u = 0.06
    config = make_config(layers, groups, n_atoms=60)
    config.imaging_mode = "stem_haadf"
    config.stem_convergence_mrad = 25.0
    config.stem_detector_mrad = (60.0, 200.0)
    config.stem_source_size_A = 0.4
    config.init_aberrations = {"C10": 0.0, "C12": 0.0, "phi12": 0.0, "C30": 0.001}

    tm = TrainMethod().ensure_for_config(config)
    model = NN_particles(config=config, pixel_size_nm=config.pixel_size_nm,
                         voltage_kV=config.voltage_kV, train_method=tm,
                         device=DEVICE, train_methods=tm)
    model.build()
    haadf = model.imaging_layer

    probe = haadf.probe_intensity()
    assert abs(float(probe.sum()) - 1.0) < 1e-4, float(probe.sum())
    assert float(probe.min()) >= 0.0

    image, _aux = model.model(None)
    image_np = image.detach().cpu().numpy()
    assert image_np.shape == tuple(config.shape)
    assert np.isfinite(image_np).all() and image_np.min() >= 0.0

    # Substitution C -> N muss heller werden
    intensities = model.potential_layer.tunable_parameters.intensities[0]
    with torch.no_grad():
        intensities[:5] = 7.0
    brighter = model.model(None)[0].detach().cpu().numpy()
    with torch.no_grad():
        intensities[:5] = 6.0
    difference = brighter - image_np
    assert difference.max() > 0 and difference.max() > abs(difference.min()), \
        "Stickstoff muss im HAADF heller werden"

    # fittbar?
    tm.set_active_group("Gr")
    tm.group("Gr").shift.enabled = True
    tm.layer("Gr").intensities.enabled = True
    model.potential_layer.apply_trainable_selection()
    _loss, with_grad = step_and_report(model, tm)
    assert "groups.Gr.shift" in with_grad and "layers.Gr.intensities" in with_grad, with_grad

    print("  STEM: Sonde normiert, N heller als C, Positionen und Elemente fittbar")


def test_normalisation_is_scale_invariant():
    """Die Normalisierungsparameter wirken relativ - sonst sprengt STEM den Loss."""

    layers, groups = graphene_only()
    config = make_config(layers, groups, n_atoms=40)
    tm = TrainMethod().ensure_for_config(config)
    tm.train_normalisation_parameters.enabled = True
    model = NN_particles(config=config, pixel_size_nm=config.pixel_size_nm,
                         voltage_kV=config.voltage_kV, train_method=tm,
                         device=DEVICE, train_methods=tm)
    model.build()

    torch.manual_seed(3)
    sim = torch.rand(tuple(config.shape), device=DEVICE)
    tgt = torch.rand(tuple(config.shape), device=DEVICE)
    norm = model.tunable_parameters.normalisation_parameters

    # (a) bei 0 exakt die uebliche z-Normierung
    with torch.no_grad():
        norm[:] = 0.0
    got, _ = model.normalize_images(sim, tgt)
    want = (sim - sim.mean()) / sim.std()
    assert torch.allclose(got, want, atol=1e-5), float((got - want).abs().max())

    # (b) Korrekturen greifen und sind unabhaengig von der Bildgroessenordnung.
    #     Ein HAADF-Bild hat eine Streuung um 1e-6 - additive Offsets wuerden
    #     dort den Nenner ins Negative kippen.
    with torch.no_grad():
        norm[0], norm[1] = 0.25, -0.4
    got, _ = model.normalize_images(sim, tgt)
    want = (sim - (sim.mean() + 0.25 * sim.std())) / (sim.std() * 0.6)
    assert torch.allclose(got, want, atol=1e-4), float((got - want).abs().max())

    for scale in (1e-6, 1e3):
        scaled, _ = model.normalize_images(sim * scale, tgt)
        deviation = float((scaled - got).abs().max())
        assert deviation < 1e-3, f"Faktor {scale:g} veraendert die Normierung um {deviation}"

    print("  Normierung: bei 0 unveraendert, Korrekturen relativ, invariant von 1e-6 bis 1e3")


def test_stem_probe_defaults():
    """Die Sonde startet bei typischen Korrektor-Restwerten - und erbt nicht die TEM-Aberrationen."""

    from ctf import DEFAULT_PROBE_ABERRATIONS

    layers, groups = graphene_only()
    layers[0].elements = ["B", "C", "N", "O"]
    config = make_config(layers, groups, n_atoms=60)
    config.imaging_mode = "stem_haadf"
    config.stem_convergence_mrad = 25.0
    config.stem_detector_mrad = (60.0, 200.0)
    config.stem_source_size_A = 0.4
    config.init_aberrations = dict(ABERRATIONS)      # TEM: C10 = -13 nm
    config.init_probe_aberrations = None             # nichts vorgegeben

    tm = TrainMethod().ensure_for_config(config)
    model = NN_particles(config=config, pixel_size_nm=config.pixel_size_nm,
                         voltage_kV=config.voltage_kV, train_method=tm,
                         device=DEVICE, train_methods=tm)
    model.build()

    values = {k: float(v) for k, v in model.probe_layer.tunable_parameters.aberrations.items()}
    assert abs(values["C10"] - DEFAULT_PROBE_ABERRATIONS["C10"]) < 1e-6, values
    assert abs(values["C30"] - DEFAULT_PROBE_ABERRATIONS["C30"]) < 1e-6, values
    assert abs(values["C10"]) < 5.0, "kein TEM-Defokus (-13 nm)"
    assert abs(float(config.init_aberrations["C10"]) + 13.0) < 1e-9, "TEM-CTF bleibt unberuehrt"

    # Der Satz ist der, den man am Korrektor einstellt
    for key in ("C10", "C12", "phi12", "C21", "phi21", "C23", "phi23", "C30"):
        assert key in DEFAULT_PROBE_ABERRATIONS, key

    # eine fokussierte Sonde ist deutlich enger als die TEM-geerbte
    focused = model.imaging_layer.probe_intensity().detach().cpu().numpy()
    config.init_probe_aberrations = dict(config.init_aberrations)
    blurred_model = NN_particles(config=config, pixel_size_nm=config.pixel_size_nm,
                                 voltage_kV=config.voltage_kV, train_method=tm,
                                 device=DEVICE, train_methods=tm)
    blurred_model.build()
    blurred = blurred_model.imaging_layer.probe_intensity().detach().cpu().numpy()

    assert focused.max() > 3.0 * blurred.max(), (float(focused.max()), float(blurred.max()))
    print(f"  STEM-Sonde: Korrektor-Startwerte, Spitze {focused.max():.3g} statt "
          f"{blurred.max():.3g} mit den TEM-Werten")


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
    print("test_debye_waller")
    test_debye_waller()
    print("test_isolated_group_skips_empty_slices")
    test_isolated_group_skips_empty_slices()
    print("test_ctf_aberration_switches")
    test_ctf_aberration_switches()
    print("test_ctf_parametrizations_are_equivalent")
    test_ctf_parametrizations_are_equivalent()
    print("test_contrast_factor")
    test_contrast_factor()
    print("test_spatial_coherence_envelope")
    test_spatial_coherence_envelope()
    print("test_normalisation_is_scale_invariant")
    test_normalisation_is_scale_invariant()
    print("test_renamed_config_fields")
    test_renamed_config_fields()
    print("test_haadf_cross_sections")
    test_haadf_cross_sections()
    print("test_stem_imaging")
    test_stem_imaging()
    print("test_stem_probe_defaults")
    test_stem_probe_defaults()
    print("test_legacy_migration")
    test_legacy_migration()
    print("\nalle Tests durch")
