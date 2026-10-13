"""Part I, Section 4: atomic structure."""

import numpy as np
import pytest
import torch

from temsim.data.schema import LatticeSpec
from temsim.physics import structure


@pytest.mark.parametrize("spec,shape,px", [
    (LatticeSpec(type="graphene", a=2.46, rotation_deg=7.0, offset_px=(3.0, -2.0)), (96, 80), 0.02),
    (LatticeSpec(type="mx2", formula="WS2", a=3.18), (64, 64), 0.025),
    (LatticeSpec(type="custom", a=3.0, b=4.0, gamma_deg=90.0,
                 basis=[("Mo", 0.0, 0.0, 0.0), ("S", 0.5, 0.5, 0.2)]), (60, 70), 0.03),
])
def test_lattice_matches_legacy(legacy, spec, shape, px):
    legacy = legacy("lattice")
    legacy_spec = legacy.LatticeSpec(type=spec.type, a=spec.a, b=spec.b, gamma_deg=spec.gamma_deg,
                                     formula=spec.formula, basis=list(spec.basis),
                                     rotation_deg=spec.rotation_deg, offset_px=spec.offset_px)
    old = legacy.build_layer_positions(legacy_spec, shape, px, max_shift_px=5.0, max_rotation_deg=10.0)
    new = structure.build_lattice(spec, shape, px, max_shift_px=5.0, max_rotation_deg=10.0)
    assert len(old) == len(new)
    for o, n in zip(old, new):
        assert n.z_nm == pytest.approx(o["z_nm"])
        np.testing.assert_allclose(n.positions, o["positions"])
        assert n.elements == o["elements"]


def test_graphene_density_is_two_atoms_per_cell():
    spec = LatticeSpec(type="graphene")
    px = 0.02
    layers = structure.build_lattice(spec, (200, 200), px)
    area_cell = 0.5 * np.sqrt(3) * 2.46 ** 2 * 1e-2       # nm^2
    inside = layers[0].positions
    mask = (inside >= 0).all(1) & (inside < 200).all(1)
    assert mask.sum() / (200 * px) ** 2 == pytest.approx(2 / area_cell, rel=0.05)


def test_rasterise_matches_legacy_and_conserves_weight(legacy):
    legacy = legacy("potential")
    g = torch.Generator().manual_seed(0)
    pos = torch.rand(300, 2, generator=g) * torch.tensor([60.0, 70.0]) - 5
    w = torch.rand(300, generator=g)
    new = structure.rasterise(pos, w, (60, 70), padding=10)
    ref = legacy.DifferentiablePotential.rasterize_atoms_bilinear(
        SimpleSelf(), 80, 90, pos + 10, w)
    torch.testing.assert_close(new, ref)
    inside = ((pos + 10 >= 0) & (pos + 10 < torch.tensor([79.0, 89.0]))).all(1)
    assert float(new.sum()) == pytest.approx(float(w[inside].sum()), rel=1e-5)


class SimpleSelf:
    device = "cpu"


def test_rasterise_preserves_centre_of_mass_and_is_differentiable():
    pos = torch.tensor([[10.3, 20.8]], requires_grad=True)
    rho = structure.rasterise(pos, torch.tensor([2.0]), (40, 40))
    yy, xx = torch.meshgrid(torch.arange(40.0), torch.arange(40.0), indexing="ij")
    cy, cx = (rho * yy).sum() / rho.sum(), (rho * xx).sum() / rho.sum()
    assert float(cy) == pytest.approx(10.3, abs=1e-5) and float(cx) == pytest.approx(20.8, abs=1e-5)
    (rho * xx).sum().backward()
    assert float(pos.grad[0, 1]) == pytest.approx(2.0, abs=1e-5)


def test_transform_identity_rotation_and_shift():
    pts = torch.tensor([[10.0, 20.0], [30.0, 5.0]], dtype=torch.float64)
    out = structure.transform(pts, (40, 50), torch.zeros(2, dtype=torch.float64),
                              torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(out, pts)
    quarter = structure.transform(pts, (40, 40), torch.zeros(2, dtype=torch.float64),
                                  torch.tensor(np.pi / 2, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
    # (dy, dx) = (-10, 0) -> (0, -10); centre (20, 20): -> (20, 10)
    torch.testing.assert_close(quarter[0], torch.tensor([20.0, 10.0], dtype=torch.float64), atol=1e-9, rtol=0)


def test_transform_matches_legacy_get_positions_formula():
    pts = torch.rand(20, 2, dtype=torch.float64) * 50
    shift = torch.tensor([1.5, -2.0], dtype=torch.float64)
    theta = torch.tensor(0.1, dtype=torch.float64)
    scale = torch.tensor(1.02, dtype=torch.float64)
    centre = torch.tensor([25.0, 30.0], dtype=torch.float64)
    c = pts - centre
    rot = torch.stack([torch.cos(theta) * c[:, 0] - torch.sin(theta) * c[:, 1],
                       torch.sin(theta) * c[:, 0] + torch.cos(theta) * c[:, 1]], 1)
    ref = rot * scale + centre + shift
    torch.testing.assert_close(structure.transform(pts, (50, 60), shift, theta, scale), ref)


def test_mixture_is_partition_of_unity_and_interpolates():
    z = torch.tensor([6.0, 6.5, 7.0, 12.0, 16.0, 5.0, 20.0])
    shares = structure.mixture_shares(z, [6, 7, 16])
    total = sum(shares)
    torch.testing.assert_close(total, torch.ones_like(total))
    assert [float(s[0]) for s in shares] == [1.0, 0.0, 0.0]
    assert [float(s[1]) for s in shares] == pytest.approx([0.5, 0.5, 0.0])
    assert [float(s[3]) for s in shares] == pytest.approx([0.0, 4 / 9, 5 / 9])
    assert float(shares[0][5]) == 1.0 and float(shares[2][6]) == 1.0      # clamped


def test_element_densities_of_mixed_layer_sum_to_occupancy():
    pos = torch.tensor([[10.2, 10.7], [20.5, 22.1]])
    layer = structure.AtomLayer(z_nm=0.0, element="C", positions=pos, weights=torch.tensor([1.0, 0.5]),
                                elements=("C", "N"), atomic_number=torch.tensor([6.0, 6.5]))
    maps = structure.element_densities(layer, (32, 32), padding=4)
    assert set(maps) == {"C", "N"}
    assert float(maps["C"].sum()) == pytest.approx(1.0 + 0.25, rel=1e-5)
    assert float(maps["N"].sum()) == pytest.approx(0.25, rel=1e-5)
