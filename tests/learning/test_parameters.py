"""Part II, Section 1: parameter set."""

import numpy as np
import pytest
import torch

from temsim.data.schema import Group, TrainSettings
from temsim.learning.parameters import ParameterSet
from temsim.physics.simulation import Simulator

from .helpers import make_experiment


def test_defaults_give_identity_geometry():
    exp = make_experiment()
    ps = ParameterSet(exp)
    for i in range(2):
        np.testing.assert_allclose(ps.positions(i).detach().numpy(), exp.structure.positions[i], atol=1e-4)
        torch.testing.assert_close(ps.weights(i), torch.full((ps.counts()[i],), float(torch.sigmoid(torch.tensor(3.0)))))


def test_apply_flags_follows_switches():
    exp = make_experiment()
    ps = ParameterSet(exp)
    s = TrainSettings()
    s.ensure_for(exp)
    s.aberrations.enabled = True
    s.aberration_flags = {k: False for k in s.aberration_flags}
    s.aberration_flags["C10"] = True
    s.normalisation.enabled = False
    s.group("g1").shift.enabled = True
    s.layer("top").occupancies.enabled = True
    s.envelope.enabled = True
    s.envelope_flags.update(deltaE=True, Cc=False)
    ps.apply_flags(s)
    assert ps.aberrations["C10"].requires_grad and not ps.aberrations["C30"].requires_grad
    assert not ps.normalisation.requires_grad
    assert ps.groups["g1"].shift.requires_grad and not ps.groups["g1"].rotation.requires_grad
    assert ps.layers[0].logit.requires_grad and not ps.layers[1].logit.requires_grad
    assert ps.delta_E.requires_grad and not ps.Cc_um.requires_grad


def test_bounds_clamp_only_the_effective_value():
    exp = make_experiment()
    ps = ParameterSet(exp)
    g = ps.groups["g1"]
    with torch.no_grad():
        g.shift.copy_(torch.tensor([50.0, -50.0]))
        g.rotation.fill_(1.0)
        ps.aberrations["C10"].fill_(-5000.0)
    shift, rotation, scale = ps.rigid("g1")
    assert shift.tolist() == [5.0, -5.0] and float(rotation) == pytest.approx(0.2)
    assert float(ps.optics_state().aberrations["C10"]) == -1000.0
    assert float(ps.aberrations["C10"]) == -5000.0


def test_state_round_trip():
    exp = make_experiment()
    ps = ParameterSet(exp)
    with torch.no_grad():
        ps.layers[0].shifts.add_(0.3)
        ps.layers[1].logit.mul_(0.5)
        ps.groups["g1"].rotation.fill_(0.1)
        ps.groups["g1"].strain.normal_()
        ps.aberrations["C10"].fill_(-12.0)
        ps.normalisation.copy_(torch.tensor([0.1, -0.1]))
    again = ParameterSet(exp, ps.to_state())
    for a, b in zip(ps.all_leaves(), again.all_leaves()):
        torch.testing.assert_close(a, b)


def test_legacy_strain_control_is_converted():
    exp = make_experiment()
    state = ParameterSet(exp).to_state()
    rng = np.random.default_rng(1)
    state.strain_control["g1"] = rng.normal(size=(1, 2, 4, 4)) * 0.01
    plain = ParameterSet(exp, state).groups["g1"].strain
    state.strain_convention = "legacy"
    converted = ParameterSet(exp, state).groups["g1"].strain
    assert converted.shape == plain.shape
    assert not torch.allclose(converted, plain)


def test_append_and_remove_keep_flags_and_values():
    exp = make_experiment()
    ps = ParameterSet(exp)
    s = TrainSettings()
    s.layer("top").occupancies.enabled = True
    s.layer("top").free_positions.enabled = True
    ps.apply_flags(s)
    before = ps.layers[0].logit.detach().clone()
    ps.append_atoms(0, [[10.0, 12.0], [20.0, 22.0]])
    lp = ps.layers[0]
    assert len(lp) == 32 and lp.logit.requires_grad and lp.shifts.requires_grad and not lp.intensity.requires_grad
    torch.testing.assert_close(lp.logit.detach()[:30], before)
    assert float(lp.logit[-1]) == 6.0 and float(lp.intensity[-1]) == 6.0
    keep = np.ones(32, dtype=bool)
    keep[[0, 31]] = False
    ps.remove_atoms(0, keep)
    assert len(ps.layers[0]) == 30 and ps.layers[0].logit.requires_grad
    ps.remove_atoms(0, np.array([1, 2, 3]))
    assert len(ps.layers[0]) == 3
    assert ps.to_structure().counts == [3, 20]


def test_mixed_layer_weight_is_occupancy_only():
    exp = make_experiment(mixed=True)
    ps = ParameterSet(exp)
    torch.testing.assert_close(ps.weights(0), ps.occupancy(0))
    layer = ps.atom_layer(0)
    assert layer.mixed and layer.atomic_number is not None


def test_isolate_when_active_switches_other_groups_off():
    groups = [Group("g1", control_points=[3, 3], isolate_when_active=True), Group("g2", control_points=[3, 3])]
    ps = ParameterSet(make_experiment(groups=groups))
    assert [l.active for l in ps.effective_layers("g1")] == [True, False]
    assert [l.active for l in ps.effective_layers("g2")] == [True, True]
    assert [l.active for l in ps.effective_layers(None)] == [True, True]


def test_gradient_reaches_every_enabled_parameter():
    exp = make_experiment()
    ps = ParameterSet(exp)
    s = TrainSettings()
    s.ensure_for(exp)
    s.group("g1").shift.enabled = s.group("g1").rotation.enabled = s.group("g1").scale.enabled = True
    s.group("g1").strain_field.enabled = s.group("g1").contrast.enabled = True
    for name in ("top", "bottom"):
        s.layer(name).free_positions.enabled = s.layer(name).intensities.enabled = True
        s.layer(name).occupancies.enabled = True
    s.amp_contrast.enabled = True
    ps.apply_flags(s)
    sim = Simulator(exp.imaging)
    image = sim.image(ps.effective_layers(), ps.optics_state())
    (image - image.detach().mean()).pow(2).sum().backward()
    for key in s.params():
        if not s.enabled(key) or key == "normalisation":  # normalisation acts in the objective
            continue
        for t in ps.tensors(key, s):
            assert t.grad is not None and torch.isfinite(t.grad).all(), key
