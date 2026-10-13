"""Part I, Section 9: the complete forward model against the legacy core."""

import numpy as np
import pytest
import torch

from temsim.data.schema import Imaging, Stem
from temsim.physics import optics, simulation, structure
from temsim.physics.optics import FourierGrid
from temsim.physics.simulation import OpticsState, Simulator

ABERRATIONS = {"C10": -13., "C12": 1.0, "phi12": 0.2, "C23": 2., "phi23": 0.1, "C21": 2., "phi21": 0.3,
               "C30": 4., "C34": .02, "phi34": 0.1, "C45": .05, "phi45": 0.2}
ENV = {"source_size": 0.1e-3, "Cc_um": 1.e3, "deltaE": .35, "instab_defocus_nm": .01, "use_detector": False}
PROBE = {"C10": 1.0, "C12": 1.0, "phi12": 0.2, "C21": 5.0, "phi21": 0.1, "C23": 3.0, "phi23": 0.4, "C30": 0.5}
SHAPE, PAD, PX = (48, 56), 12, 0.0052 * 4


def legacy_model(legacy, mode, mixed=False):
    layers_mod, nn_mod, tp_mod = legacy("layers"), legacy("nn"), legacy("training_parameter")
    legacy("multislice").CHECK_PHASE = False
    L, G = layers_mod.LayerConfig, layers_mod.LayerGroupConfig
    layers = [
        L("Gr", "C", 0.0, "Gr", init_occupancy=1.5, occ_entropy=False, debye_waller_u=0.06),
        L("S1", "S", 0.33, "WS2", debye_waller_u=0.07),
        L("W", "W", 0.488, "WS2", debye_waller_u=0.055),
        L("S2", "S", 0.646, "WS2", debye_waller_u=0.07),
    ]
    if mixed:
        layers.append(L("BN", "C", 0.9, "BN", elements=["B", "C", "N", "O"], debye_waller_u=0.06))
    groups = [G("Gr", max_shift=20., max_rotation=0.35), G("WS2", max_shift=20., max_rotation=0.09)]
    if mixed:
        groups.append(G("BN", max_shift=20., max_rotation=0.2))
    rng = np.random.default_rng(1)
    n = 30
    positions, intensities = [], []
    for layer in layers:
        positions.append(rng.uniform(6, min(SHAPE) - 6, size=(n, 2)))
        intensities.append(rng.uniform(5.6, 7.4, n) if layer.elements else
                           np.full(n, {"C": 6.0, "S": 16.0, "W": 74.0}[layer.element]))
    config = nn_mod.ParticlesConfig(
        init_positions=positions, init_intensities=intensities, shape=SHAPE, pixel_size_nm=PX, voltage_kV=80,
        init_aberrations=ABERRATIONS, init_ctf_env_params=ENV, init_amp_contrast=0.1, ctf_alpha_max_mrad=50.,
        padding=PAD, layers=layers, groups=groups, imaging_mode=mode, init_probe_aberrations=dict(PROBE),
        stem_convergence_mrad=25.0, stem_detector_mrad=(60.0, 200.0), stem_source_size_A=0.5)
    tm = tp_mod.TrainMethod().ensure_for_config(config)
    model = nn_mod.NN_particles(config=config, pixel_size_nm=PX, voltage_kV=80, train_method=tm,
                                device="cpu", train_methods=tm)
    model.build()
    return model, config


def new_layers(model):
    pot = model.potential_layer
    positions = pot.get_positions()
    weights = pot.get_atom_weights()
    contributes = pot.active_layer_mask()
    out = []
    for i, cfg in enumerate(model.config.layers):
        elements = list(pot.layer_elements[i])
        mixed = len(elements) > 1
        if mixed:
            intensities = pot.tunable_parameters.intensities[i].detach()
            atomic_number = intensities.clone()
            w = torch.sigmoid(pot.tunable_parameters.occupancies[i].detach())
        else:
            atomic_number, w = None, weights[i].detach()
        out.append(structure.AtomLayer(
            z_nm=cfg.z_nm, element=cfg.element, positions=positions[i].detach(), weights=w,
            elements=elements if mixed else (), atomic_number=atomic_number,
            debye_waller_u_A=cfg.debye_waller_u,
            contrast=float(pot.tunable_parameters.contrast_factors[cfg.group].detach()),
            active=bool(contributes[i])))
    return out


def imaging_setup(mode, **kw):
    return Imaging(shape=SHAPE, pixel_size_nm=PX, mode=mode, voltage_kV=80.0, padding=PAD,
                   aberrations=ABERRATIONS, envelope=ENV, amp_contrast=0.1, alpha_max_mrad=50.0,
                   band_limit=0.0, stem=Stem(25.0, (60.0, 200.0), 0.5), **kw)


def tem_simulator(model, setup):
    sim = Simulator(setup)
    ctf = model.imaging_layer
    sim.grid = FourierGrid(sim.padded_shape, PX, sim.wavelength_nm, k_row=ctf.kx, k_col=ctf.ky)
    sim.objective_aperture = ctf.aperture
    return sim


@pytest.mark.parametrize("mixed", [False, True])
def test_tem_matches_legacy(legacy, mixed):
    model, config = legacy_model(legacy, "tem", mixed)
    with torch.no_grad():
        old, _ = model.model(None)
    sim = tem_simulator(model, imaging_setup("tem"))
    state = OpticsState(ABERRATIONS, 0.1, 0.35, 1.0e3)
    layers = new_layers(model)
    with torch.no_grad():
        new = sim.image(layers, state)
    assert new.shape == SHAPE
    torch.testing.assert_close(new, old, rtol=2e-3, atol=2e-3 * float(old.abs().max()))


def test_tem_intensity_option(legacy):
    model, _ = legacy_model(legacy, "tem")
    sim = tem_simulator(model, imaging_setup("tem", tem_signal="intensity"))
    sim_amp = tem_simulator(model, imaging_setup("tem"))
    state = OpticsState(ABERRATIONS, 0.1, 0.35, 1.0e3)
    layers = new_layers(model)
    with torch.no_grad():
        torch.testing.assert_close(sim.image(layers, state), sim_amp.image(layers, state) ** 2,
                                   rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("mixed", [False, True])
def test_stem_matches_legacy(legacy, mixed):
    model, config = legacy_model(legacy, "stem_haadf", mixed)
    with torch.no_grad():
        old, _ = model.model(None)
    setup = imaging_setup("stem_haadf")
    sim = Simulator(setup)
    probe = model.imaging_layer.probe_layer
    sim.grid = FourierGrid(sim.padded_shape, PX, sim.wavelength_nm, k_row=probe.kx, k_col=probe.ky)
    sim.probe_aperture = model.imaging_layer.probe_aperture
    layers = new_layers(model)
    with torch.no_grad():
        new = sim.image(layers, OpticsState(PROBE, 0.1, 0.35, 1.0e3))
    torch.testing.assert_close(new, old, rtol=2e-3, atol=2e-3 * float(old.abs().max()))


def test_inactive_layer_is_skipped_and_gradients_reach_parameters(legacy):
    model, _ = legacy_model(legacy, "tem")
    sim = tem_simulator(model, imaging_setup("tem"))
    layers = new_layers(model)
    for layer in layers:
        layer.positions = layer.positions.clone().requires_grad_(True)
        layer.weights = layer.weights.clone().requires_grad_(True)
    layers[0].active = False
    coeffs = {k: torch.tensor(float(v), requires_grad=True) for k, v in ABERRATIONS.items()}
    de, cc, q = (torch.tensor(v, requires_grad=True) for v in (0.35, 1.0e3, 0.1))
    image = sim.image(layers, OpticsState(coeffs, q, de, cc))
    image.sum().backward()
    assert layers[0].positions.grad is None
    assert all(l.positions.grad is not None and float(l.positions.grad.abs().sum()) > 0 for l in layers[1:])
    for t in (coeffs["C10"], coeffs["C30"], de, cc, q):
        assert t.grad is not None and torch.isfinite(t.grad)
