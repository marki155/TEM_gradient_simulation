"""Part II, Section 3: priors against analytic values."""

import numpy as np
import pytest
import torch

from temsim.data.schema import Group, TrainSettings
from temsim.learning import priors
from temsim.learning.parameters import ParameterSet

from .helpers import SHAPE, PX, make_experiment


def setup(**kw):
    exp = make_experiment(**kw)
    ps = ParameterSet(exp)
    s = TrainSettings()
    s.ensure_for(exp)
    return exp, ps, s


def test_entropy_values():
    _, ps, _ = setup()
    with torch.no_grad():
        ps.layers[0].logit.zero_()           # p = 1/2 -> ln 2
        ps.layers[1].logit.fill_(30.0)       # p = 1  -> 0
    assert float(priors.entropy(ps)) == pytest.approx(0.5 * np.log(2.0), rel=1e-4)


def test_entropy_skips_switched_off_layers():
    exp, ps, _ = setup()
    exp.layers[1].occ_entropy = False
    with torch.no_grad():
        ps.layers[0].logit.zero_()
    assert float(priors.entropy(ps)) == pytest.approx(np.log(2.0), rel=1e-4)


def test_position_norm_counts_only_free_layers():
    _, ps, s = setup()
    s.layer("top").free_positions.enabled = True
    with torch.no_grad():
        ps.layers[0].shifts.fill_(0.5)
        ps.layers[1].shifts.fill_(9.0)
    assert float(priors.position_norm(ps, s)) == pytest.approx(0.5)
    s.layer("top").free_positions.enabled = False
    assert float(priors.position_norm(ps, s)) == 0.0


def test_repulsion_pair():
    exp, ps, s = setup(n=(2, 1))
    s.layer("top").free_positions.enabled = True
    s.regularisation.repulsion_distance_A = 2.0
    d0 = 2.0 / (PX * 10)                                    # px
    with torch.no_grad():
        ps.layers[0].base.copy_(torch.tensor([[20.0, 20.0], [20.0, 20.0 + 0.5 * d0]]))
        ps.layers[0].logit.fill_(3.0)
    value = priors.repulsion(ps, s)
    assert float(value) == pytest.approx(0.25 / 2.0, rel=1e-4)           # (1-1/2)^2 / n
    with torch.no_grad():
        ps.layers[0].base[1, 1] = 20.0 + 1.5 * d0
    assert float(priors.repulsion(ps, s)) == 0.0
    # unoccupied atoms do not repel
    with torch.no_grad():
        ps.layers[0].base[1, 1] = 20.0 + 0.5 * d0
        ps.layers[0].logit[1] = -3.0
    assert float(priors.repulsion(ps, s)) == 0.0


def test_repulsion_gradient_pushes_apart():
    exp, ps, s = setup(n=(2, 1))
    s.layer("top").free_positions.enabled = True
    s.regularisation.repulsion_distance_A = 2.0
    d0 = 2.0 / (PX * 10)
    with torch.no_grad():
        ps.layers[0].base.copy_(torch.tensor([[20.0, 20.0], [20.0, 20.0 + 0.5 * d0]]))
    ps.apply_flags(s)
    priors.repulsion(ps, s).backward()
    g = ps.layers[0].shifts.grad
    assert g[0, 1] > 0 and g[1, 1] < 0 and abs(float(g[0, 1] + g[1, 1])) < 1e-6   # descent separates them


def test_attraction_pulls_close_pairs_together():
    exp, ps, s = setup(n=(2, 1))
    s.layer("top").free_positions.enabled = True
    s.regularisation.merge_distance_A = 1.0
    dm = 1.0 / (PX * 10)
    with torch.no_grad():
        ps.layers[0].base.copy_(torch.tensor([[20.0, 20.0], [20.0, 20.0 + 0.5 * dm]]))
        ps.layers[0].logit.fill_(3.0)
    ps.apply_flags(s)
    value = priors.attraction(ps, s)
    assert float(value) == pytest.approx(0.25 / 2.0, rel=1e-4)
    value.backward()
    g = ps.layers[0].shifts.grad
    assert g[0, 1] < 0 and g[1, 1] > 0                       # descent brings them together
    with torch.no_grad():
        ps.layers[0].base[1, 1] = 20.0 + 1.5 * dm
    assert float(priors.attraction(ps, s)) == 0.0


def test_merge_close_fuses_chains_and_keeps_the_rest():
    exp, ps, s = setup(n=(5, 1))
    with torch.no_grad():
        ps.layers[0].base.copy_(torch.tensor([[20.0, 20.0], [20.0, 21.0], [20.0, 22.0], [20.0, 40.0], [30.0, 30.0]]))
        ps.layers[0].logit.copy_(torch.tensor([4.0, 4.0, 4.0, 4.0, -4.0]))
        ps.layers[0].intensity.copy_(torch.tensor([6.0, 6.0, 9.0, 6.0, 6.0]))
    assert ps.merge_close(0, 1.5) == 2
    assert len(ps.layers[0]) == 3
    base = ps.layers[0].base.detach()
    assert float(base[0, 0]) == pytest.approx(20.0) and 20.9 < float(base[0, 1]) < 21.6   # weighted by occ * I
    assert float(base[1, 1]) == 40.0 and float(base[2, 0]) == 30.0
    assert 6.0 < float(ps.layers[0].intensity[0]) < 9.0 and float(ps.layers[0].logit[0]) == pytest.approx(4.0)
    assert ps.merge_close(0, 1.5) == 0
    with pytest.raises(ValueError):
        ps.merge_atoms(0, [[0, 1], [1, 2]])


def strain_experiment(e):
    exp, ps, s = setup(groups=[Group("g1", control_points=[5, 5])])
    s.group("g1").strain_field.enabled = True
    s.regularisation.strain_energy_lambda, s.regularisation.strain_energy_mu = 2.0, 3.0
    H, W = SHAPE
    rows = torch.linspace(0, 1, 5)[:, None].expand(5, 5)
    control = torch.zeros(1, 2, 5, 5)
    control[0, 0] = e * (H - 1) * rows                    # u_row = e * row  -> J00 = e
    with torch.no_grad():
        ps.groups["g1"].strain.copy_(control)
    ps.apply_flags(s)
    return exp, ps, s


def test_uniform_strain_energy_is_analytic():
    e = 0.01
    _, ps, s = strain_experiment(e)
    # eps = diag(e, 0): w = lambda/2 e^2 + mu e^2
    assert float(priors.elastic_energy(ps, s)) == pytest.approx((0.5 * 2.0 + 3.0) * e ** 2, rel=1e-3)


def test_strain_energy_is_trainable():
    _, ps, s = strain_experiment(0.01)
    s.regularisation.strain_energy_a = 1.0
    loss, parts = priors.prior_loss(ps, s)
    loss.backward()
    grad = ps.groups["g1"].strain.grad
    assert grad is not None and torch.isfinite(grad).all() and float(grad.abs().max()) > 0
    assert "strain_energy" in parts


def test_rigid_rotation_costs_nothing():
    exp, ps, s = setup(groups=[Group("g1", control_points=[5, 5])])
    s.group("g1").strain_field.enabled = True
    with torch.no_grad():
        ps.groups["g1"].rotation.fill_(0.1)
    assert float(priors.elastic_energy(ps, s)) == pytest.approx(0.0, abs=1e-12)


def test_zero_weight_terms_are_not_evaluated():
    _, ps, s = setup()
    s.regularisation.occ_entropy_a = 0.0
    s.regularisation.position_shifts_a = 0.0
    s.regularisation.repulsion_a = 0.0
    s.regularisation.attraction_a = 0.0
    value, parts = priors.prior_loss(ps, s)
    assert float(value) == 0.0 and parts == {}
