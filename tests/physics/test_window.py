"""Part I, Section 10: local window update equals the full simulation."""

import numpy as np
import pytest
import torch

from temsim.data.schema import Imaging, Stem
from temsim.physics import structure
from temsim.physics.simulation import OpticsState, Simulator
from temsim.physics.structure import AtomLayer
from temsim.physics.window import LocalImage

SHAPE, PAD, PX = (96, 104), 16, 0.02
ABERR = {"C10": -8.0, "C12": 1.0, "phi12": 0.2, "C30": 1.0, "C21": 5.0, "phi21": 0.3}
PROBE = {"C10": 1.0, "C12": 1.0, "phi12": 0.2, "C30": 0.5}
STATE = OpticsState(ABERR, 0.1, 0.35, 1.0e3)


def setup(mode, **kw):
    return Imaging(shape=SHAPE, pixel_size_nm=PX, mode=mode, voltage_kV=80.0, padding=PAD, aberrations=ABERR,
                   envelope={"source_size": 0.1e-3, "instab_defocus_nm": 0.01}, alpha_max_mrad=50.0,
                   stem=Stem(25.0, (60.0, 200.0), 0.5), **kw)


def make_layers(seed=0, n=60, mixed=False):
    rng = np.random.default_rng(seed)

    def pos():
        return torch.tensor(rng.uniform(10, min(SHAPE) - 10, (n, 2)), dtype=torch.float32)

    layers = [
        AtomLayer(0.0, "C", pos(), torch.tensor(rng.uniform(0.5, 1.0, n), dtype=torch.float32), debye_waller_u_A=0.06),
        AtomLayer(0.33, "S", pos(), torch.tensor(rng.uniform(0.5, 1.0, n), dtype=torch.float32), debye_waller_u_A=0.07,
                  contrast=1.2),
        AtomLayer(0.49, "W", pos(), torch.tensor(rng.uniform(0.5, 1.0, n), dtype=torch.float32), debye_waller_u_A=0.055),
    ]
    if mixed:
        layers.append(AtomLayer(0.8, "C", pos(), torch.ones(n), elements=["B", "C", "N", "O"],
                                atomic_number=torch.tensor(rng.uniform(5.2, 7.8, n), dtype=torch.float32),
                                debye_waller_u_A=0.06))
    return layers


def modified(layer, remove, add_pos, add_w, add_z=None):
    keep = torch.ones(len(layer.positions), dtype=torch.bool)
    keep[list(remove)] = False
    positions = torch.cat([layer.positions[keep], torch.as_tensor(add_pos, dtype=torch.float32).reshape(-1, 2)])
    weights = torch.cat([layer.weights[keep], torch.as_tensor(add_w, dtype=torch.float32)])
    z = layer.atomic_number
    if z is not None:
        z = torch.cat([z[keep], torch.as_tensor(add_z, dtype=torch.float32)])
    return AtomLayer(layer.z_nm, layer.element, positions, weights, layer.elements, z, layer.debye_waller_u_A,
                     layer.contrast, layer.active)


def changes_for(layer, remove, add_pos, add_w, add_z=None):
    """Per-element signed density changes for removing ``remove`` and adding atoms."""

    out = {}

    def put(element, pts, w):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if element in out:
            pts, w = np.concatenate([out[element][0], pts]), np.concatenate([out[element][1], w])
        out[element] = (pts, w)

    def share_weights(z, w):
        nodes = [structure.atomic_number_of(e) for e in layer.elements]
        shares = structure.mixture_shares(torch.as_tensor(z, dtype=torch.float32), nodes)
        return {e: (s * torch.as_tensor(w, dtype=torch.float32)).numpy() for e, s in zip(layer.elements, shares)}

    remove = list(remove)
    if layer.mixed:
        if remove:
            for e, w in share_weights(layer.atomic_number[remove], layer.weights[remove]).items():
                put(e, layer.positions[remove].numpy(), -w)
        if len(add_pos):
            for e, w in share_weights(add_z, add_w).items():
                put(e, add_pos, w)
    else:
        if remove:
            put(layer.element, layer.positions[remove].numpy(), -layer.weights[remove].numpy())
        if len(add_pos):
            put(layer.element, add_pos, add_w)
    return out


def compare(local_image, full_image, change, tol):
    err = float((local_image.image - full_image).abs().max())
    scale = float((full_image - change).abs().max()) if change is not None else 1.0
    return err, scale


@pytest.mark.parametrize("mode", ["tem", "stem_haadf"])
@pytest.mark.parametrize("mixed", [False, True])
def test_local_equals_full_after_commits(mode, mixed):
    sim = Simulator(setup(mode))
    state = OpticsState(PROBE if mode == "stem_haadf" else ABERR, 0.1, 0.35, 1.0e3)
    layers = make_layers(mixed=mixed)
    local = LocalImage(sim, layers, state)
    start = local.image.clone()
    rng = np.random.default_rng(3)
    scale = None
    for step in range(6):
        li = int(rng.integers(len(layers)))
        layer = layers[li]
        remove = rng.choice(len(layer.positions), 2, replace=False)
        add_pos = rng.uniform(15, min(SHAPE) - 15, (2, 2))
        add_w = rng.uniform(0.5, 1.0, 2)
        add_z = rng.uniform(5.2, 7.8, 2) if layer.mixed else None
        before = local.image.clone()
        change = local.propose(li, changes_for(layer, remove, add_pos, add_w, add_z))
        assert change is not None
        local.commit(change)
        layers[li] = modified(layer, remove, add_pos, add_w, add_z)
        full = sim.image(layers, state).detach()
        err = float((local.image - full).abs().max())
        moved = float((full - before).abs().max())
        assert moved > 0
        assert err < 2e-3 * moved + 1e-6 * float(full.abs().max()), (step, li, err, moved)
    drift = float((local.image - sim.image(layers, state).detach()).abs().max())
    assert drift < 0.05 * float((start - local.image).abs().max())
    local.refresh(layers, state)
    torch.testing.assert_close(local.image, sim.image(layers, state).detach(), rtol=1e-5, atol=1e-6)


def test_change_in_upstream_layer_reaches_downstream_waves():
    """After a commit in layer 0, a proposal in layer 2 uses the updated incident wave."""

    sim = Simulator(setup("tem"))
    layers = make_layers()
    local = LocalImage(sim, layers, STATE)
    c0 = local.propose(0, changes_for(layers[0], [0, 1, 2], np.zeros((0, 2)), []))
    local.commit(c0)
    layers[0] = modified(layers[0], [0, 1, 2], np.zeros((0, 2)), [])
    p2 = layers[2].positions[5].numpy()
    c2 = local.propose(2, changes_for(layers[2], [5], np.zeros((0, 2)), []))
    layers[2] = modified(layers[2], [5], np.zeros((0, 2)), [])
    full = sim.image(layers, STATE).detach()
    moved = float((full - local.image).abs().max())
    err = float((c2.new - full[c2.rows, c2.cols]).abs().max())
    assert err < 2e-3 * moved + 1e-6


def test_oversized_change_returns_none():
    sim = Simulator(setup("stem_haadf"))
    layers = make_layers()
    local = LocalImage(sim, layers, OpticsState(PROBE, 0.1, 0.35, 1e3), max_extent_px=10)
    pts = np.array([[20.0, 20.0], [80.0, 80.0]])
    assert local.propose(0, {"C": (pts, np.ones(2))}) is None
