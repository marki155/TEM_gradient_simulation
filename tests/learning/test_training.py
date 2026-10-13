"""Part II, Section 5: optimiser rebuild and a recovery experiment."""

import numpy as np
import pytest
import torch

from temsim.data.schema import TrainSettings
from temsim.learning.model import Model
from temsim.learning.optimiser import AdamGroups
from temsim.learning.trainer import Trainer

from .helpers import make_experiment


def truth_target(mode="tem", **kw):
    exp = make_experiment(mode=mode, **kw)
    m = Model(exp, np.zeros((64, 72)))
    return exp, m.evaluate()["image"]


def test_lrs_are_resynced_and_state_survives_a_rebuild():
    exp, target = truth_target()
    m = Model(exp, target)
    s = m.settings
    s.group("g1").shift.enabled = True
    s.aberrations.enabled = True
    s.aberration_flags = {k: False for k in s.aberration_flags}
    s.aberration_flags["C10"] = True
    t = Trainer(m)
    t.step()
    t.step()
    first = t.optimiser.optimiser
    state_before = float(first.state[m.params.groups["g1"].shift]["step"])
    s.group("g1").shift.lr = 0.05
    s.amp_contrast.enabled = True
    t.step()
    assert t.optimiser.optimiser is not first
    group = {g["name"]: g for g in t.optimiser.optimiser.param_groups}
    assert group["groups.g1.shift"]["lr"] == 0.05 and "amp_contrast" in group
    assert float(t.optimiser.optimiser.state[m.params.groups["g1"].shift]["step"]) == float(state_before) + 1


def test_nothing_trainable_is_not_an_error():
    exp, target = truth_target()
    m = Model(exp, target)
    r = Trainer(m).step()
    assert np.isfinite(r.loss)


def test_loss_decreases_and_shift_is_recovered():
    exp, _ = truth_target()
    truth = Model(exp, np.zeros((64, 72)))
    with torch.no_grad():
        truth.params.groups["g1"].shift.copy_(torch.tensor([1.5, -1.0]))
    target = truth.evaluate()["image"]
    m = Model(exp, target)
    m.settings.loss.mse = 1.0
    m.settings.group("g1").shift.enabled = True
    m.settings.group("g1").shift.lr = 0.05
    m.settings.regularisation.occ_entropy_a = 0.0
    t = Trainer(m)
    results = t.run(120)
    assert results[-1].loss < 0.2 * results[0].loss
    assert m.params.groups["g1"].shift.detach().numpy() == pytest.approx([1.5, -1.0], abs=0.15)


def test_stem_training_runs():
    exp, target = truth_target(mode="stem_haadf")
    m = Model(exp, target)
    m.settings.layer("top").occupancies.enabled = True
    m.settings.layer("top").occupancies.lr = 0.05
    r = Trainer(m).run(3)
    assert all(np.isfinite(x.loss) for x in r)
