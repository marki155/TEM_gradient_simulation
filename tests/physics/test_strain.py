"""Part I, Section 5: displacement field."""

from types import SimpleNamespace

import pytest
import torch

from temsim.physics import strain

SHAPE = (60, 80)


def linear_control(gradient, nodes=(6, 7)):
    """Control tensor of u = G r with the 2x2 matrix G (nodes span the image area)."""

    Hc, Wc = nodes
    rows = torch.linspace(0, SHAPE[0] - 1, Hc, dtype=torch.float64)
    cols = torch.linspace(0, SHAPE[1] - 1, Wc, dtype=torch.float64)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    G = torch.as_tensor(gradient, dtype=torch.float64)
    u0 = G[0, 0] * yy + G[0, 1] * xx
    u1 = G[1, 0] * yy + G[1, 1] * xx
    return torch.stack([u0, u1]).unsqueeze(0)


def interior_points(n=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(n, 2, generator=g, dtype=torch.float64)
    return pts * torch.tensor([SHAPE[0] - 21.0, SHAPE[1] - 21.0], dtype=torch.float64) + 10.0


def test_linear_field_is_reproduced_and_has_constant_jacobian():
    G = [[0.01, -0.02], [0.03, 0.005]]
    control = linear_control(G)
    pts = interior_points()
    u = strain.displacement(control, pts, SHAPE)
    expected = pts @ torch.tensor(G, dtype=torch.float64).T
    torch.testing.assert_close(u, expected, atol=1e-9, rtol=0)
    J = strain.jacobian(control, pts, SHAPE)
    torch.testing.assert_close(J, torch.tensor(G, dtype=torch.float64).expand_as(J), atol=1e-9, rtol=0)


def test_rigid_rotation_has_no_strain_energy():
    angle = 1e-3
    G = [[0.0, -angle], [angle, 0.0]]
    J = strain.jacobian(linear_control(G), interior_points(), SHAPE)
    assert float(strain.strain_tensor(J).abs().max()) < 1e-9
    assert float(strain.elastic_energy_density(J, 1.0, 1.0).max()) < 1e-18
    assert float(strain.rotation_tensor(J)[:, 1, 0].mean()) == pytest.approx(angle)


def test_uniaxial_stretch_energy():
    # eps_rr = e: w = 1/2 lambda e^2 + mu e^2
    e = 0.01
    J = strain.jacobian(linear_control([[e, 0.0], [0.0, 0.0]]), interior_points(), SHAPE)
    w = strain.elastic_energy_density(J, lame_lambda=2.0, lame_mu=3.0)
    torch.testing.assert_close(w, torch.full_like(w, 0.5 * 2.0 * e ** 2 + 3.0 * e ** 2), atol=1e-12, rtol=1e-6)


def test_jacobian_matches_finite_differences():
    g = torch.Generator().manual_seed(1)
    control = 0.5 * torch.randn(1, 2, 5, 6, generator=g, dtype=torch.float64)
    pts = interior_points(8)
    J = strain.jacobian(control, pts, SHAPE, create_graph=False)
    h = 1e-4
    for j in range(2):
        d = torch.zeros(2, dtype=torch.float64)
        d[j] = h
        fd = (strain.displacement(control, pts + d, SHAPE) - strain.displacement(control, pts - d, SHAPE)) / (2 * h)
        torch.testing.assert_close(J[:, :, j], fd, atol=1e-6, rtol=1e-5)


def test_energy_gradient_reaches_control():
    control = torch.randn(1, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    J = strain.jacobian(control, interior_points(10), SHAPE)
    strain.elastic_energy_density(J).mean().backward()
    assert control.grad is not None and float(control.grad.abs().sum()) > 0


def test_resample_is_exact_at_old_nodes_and_refinement_is_lossless_for_linear_field():
    control = linear_control([[0.01, 0.0], [0.0, -0.02]], nodes=(4, 5))
    finer = strain.resample_control(control, (9, 13))
    assert finer.shape == (1, 2, 9, 13)
    assert strain.resample_deviation(control, finer, SHAPE) < 1e-9
    # coarser grid of a field that is itself linear is also exact
    assert strain.resample_deviation(control, strain.resample_control(control, (2, 2)), SHAPE) < 1e-9


def test_resample_rejects_bad_input():
    with pytest.raises(ValueError):
        strain.resample_control(torch.zeros(1, 2, 4, 4), (1, 5))
    with pytest.raises(ValueError):
        strain.resample_control(torch.zeros(2, 4, 4), (3, 3))


def test_maps_keys_and_values():
    control = linear_control([[0.01, 0.0], [0.0, 0.0]])
    m = strain.maps(control, SHAPE, 1.0, 1.0, resolution=(10, 12))
    assert set(m) >= {"u_row", "u_col", "eps_rr", "eps_cc", "eps_rc", "rot", "div", "energy"}
    assert m["eps_rr"].shape == (10, 12)
    assert m["div"].mean() == pytest.approx(0.01, rel=1e-5)


def test_from_legacy_control_reproduces_legacy_field_at_nodes():
    legacy = pytest.importorskip("displacement_field")
    padding = 12
    cfg = SimpleNamespace(shape=SHAPE, padding=padding)
    field = legacy.SplineDisplacementField2D(cfg, control_shape=(5, 6), init_scale=0.7)
    new = strain.from_legacy_control(field.tunable_parameters.control.detach(), SHAPE, padding)
    Hc, Wc = new.shape[-2:]
    rows = torch.linspace(0, SHAPE[0] - 1, Hc)
    cols = torch.linspace(0, SHAPE[1] - 1, Wc)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    pts = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
    old_u = field.sample(pts).detach()
    new_u = strain.displacement(new, pts, SHAPE)
    torch.testing.assert_close(new_u, old_u, atol=1e-5, rtol=1e-5)


def test_uniform_strain_is_exact_up_to_the_border():
    G = [[0.02, 0.01], [-0.01, 0.03]]
    control = linear_control(G)
    pts = torch.tensor([[0.0, 0.0], [0.0, 79.0], [59.0, 0.0], [59.0, 79.0], [0.5, 40.0], [58.7, 3.2]],
                       dtype=torch.float64)
    J = strain.jacobian(control, pts, SHAPE)
    torch.testing.assert_close(J, torch.tensor(G, dtype=torch.float64).expand_as(J), atol=1e-9, rtol=0)
    # outside the image the field is constant
    out = torch.tensor([[-10.0, 40.0], [80.0, 40.0]], dtype=torch.float64)
    assert float(strain.jacobian(control, out, SHAPE)[:, 0, 0].abs().max()) == 0.0
