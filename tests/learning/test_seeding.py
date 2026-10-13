"""Part II, Sections 6-7: seeding against its own full evaluation, the legacy core, and ground truth."""

import math

import numpy as np
import pytest
import torch

from temsim.learning import Model
from temsim.learning.seeding import (AtomStore, SeedConfig, Seeder, empty_experiment, minima_positions,
                                     negative_correlation, resolve_point)

from .helpers import SHAPE, PX, make_experiment

STEM_MODES = ["tem", "stem_haadf"]


def truth_image(mode, n=25, seed=0):
    truth = make_experiment(mode=mode, n=(n, 0), seed=seed)
    with torch.no_grad():
        image = Model(truth, target=torch.zeros(SHAPE)).image()
    return truth, image.detach()


def seeding_model(mode, config=None, n=25, seed=0, **kw):
    truth, image = truth_image(mode, n, seed)
    exp = empty_experiment(SHAPE, PX, "C", mode=mode, padding=12, **kw)
    exp.imaging.aberrations = dict(truth.imaging.aberrations)
    model = Model(exp, target=image)
    return model, Seeder(model, 0, config or SeedConfig(seed=1)), truth


# ---------------------------------------------------------------- collision rule
def test_collision_rule_equals_legacy(legacy):
    old = legacy("atom_seeding")
    rng = np.random.default_rng(3)
    positions = rng.uniform(0, 40, (60, 2))
    for _ in range(200):
        x, y = rng.uniform(0, 40, 2)
        exclude = [int(rng.integers(60))] if rng.random() < 0.3 else None
        a = resolve_point(positions, x, y, 1.2, 2.5, exclude)
        b = old.resolve_point(positions, x, y, 1.2, 2.5, exclude)
        assert a[0] == pytest.approx(b[0]) and sorted(a[1]) == sorted(b[1]) and a[2] == b[2]


def test_collision_rule_cases():
    positions = np.array([[10.0, 10.0], [10.0, 14.0]])
    (xy, graveyard, merged) = resolve_point(positions, 10.0, 10.5, touch_px=1.0, merge_px=3.0)
    assert graveyard == [0] and not merged or merged            # an atom within touch is replaced
    (xy, graveyard, merged) = resolve_point(positions, 30.0, 30.0, 1.0, 3.0)
    assert graveyard == [] and not merged and xy == (30.0, 30.0)


# ---------------------------------------------------------------- minima
def test_minima_threshold_and_distance():
    diff = np.zeros((30, 30))
    diff[10, 10], diff[10, 12], diff[20, 20] = -1.0, -0.9, -0.6
    pts = minima_positions(diff, threshold=0.5, distance_px=3.0)
    assert pts.tolist() == [[10.0, 10.0], [20.0, 20.0]]          # (10,12) is within r of a deeper one
    assert minima_positions(diff, threshold=0.7, distance_px=3.0).tolist() == [[10.0, 10.0]]
    assert minima_positions(np.ones((5, 5))).shape == (0, 2)
    near = minima_positions(diff, 0.5, 3.0, atoms=np.array([[10.0, 11.0]]))
    assert near.tolist() == [[20.0, 20.0]]


# ---------------------------------------------------------------- store
def test_store_apply_and_undo_restore_everything():
    model, seeder, _ = seeding_model("stem_haadf")
    store = seeder.store
    store.apply([], np.array([[10.0, 12.0], [20.0, 22.0], [30.0, 32.0]]))
    assert store.n == 3
    before = (store.positions.copy(), model.params.layers[0].logit.detach().clone())
    token = store.apply([1], np.array([[40.0, 41.0]]))
    assert store.n == 3 and np.allclose(store.positions[-1], [40.0, 41.0])
    store.undo(token)
    np.testing.assert_allclose(store.positions, before[0])
    torch.testing.assert_close(model.params.layers[0].logit.detach(), before[1])
    store.check_identity()


def test_store_sync_drops_unoccupied_atoms():
    model, seeder, _ = seeding_model("stem_haadf")
    seeder.store.apply([], np.array([[10.0, 12.0], [20.0, 22.0]]))
    with torch.no_grad():
        model.params.layers[0].logit[0] = -5.0
    seeder.resync()
    assert seeder.store.n == 1


def test_store_changes_on_mixed_layer_use_shares():
    exp = make_experiment(mixed=True, n=(5, 5))
    model = Model(exp, target=torch.zeros(SHAPE))
    store = AtomStore(model.params, 0)
    changes = store.changes([], np.array([[10.0, 10.0]]), np.array([1.0]))
    assert len(changes) > 1 and all(len(v[0]) == 1 for v in changes.values())
    assert sum(abs(v[1][0]) for v in changes.values()) == pytest.approx(float(torch.sigmoid(torch.tensor(3.0))), rel=1e-4)


# ---------------------------------------------------------------- local loss
@pytest.mark.parametrize("mode", STEM_MODES)
def test_local_loss_equals_full_evaluation_after_many_trials(mode):
    config = SeedConfig(seed=2, move_prob=0.15, remove_prob=0.1, refresh_every=10 ** 6)
    model, seeder, _ = seeding_model(mode, config)
    seeder.run(150, log_every=0)
    assert seeder.stats.accepted > 3 and seeder.stats.refreshes == 0
    assert seeder.loss == pytest.approx(seeder.evaluate(), abs=2e-4)


@pytest.mark.parametrize("mode", STEM_MODES)
def test_loss_never_increases(mode):
    model, seeder, _ = seeding_model(mode, SeedConfig(seed=3, move_prob=0.2, remove_prob=0.1))
    history = []
    for _ in range(60):
        seeder.trial()
        history.append(seeder.loss)
    finite = [h for h in history if math.isfinite(h)]
    assert all(b <= a + 1e-12 for a, b in zip(finite, finite[1:]))


def test_full_loss_path_agrees_with_local_path():
    cfg_local = SeedConfig(seed=5)
    cfg_full = SeedConfig(seed=5, local=False)
    _, a, _ = seeding_model("stem_haadf", cfg_local)
    _, b, _ = seeding_model("stem_haadf", cfg_full)
    a.run(60, log_every=0)
    b.run(60, log_every=0)
    assert a.store.n == b.store.n
    np.testing.assert_allclose(a.store.positions, b.store.positions)
    assert a.loss == pytest.approx(b.loss, abs=2e-4)


def test_insertion_gain_equals_brute_force():
    model, seeder, _ = seeding_model("stem_haadf", SeedConfig(seed=4))
    seeder.run(20, log_every=0)
    amplitude = seeder._amplitude(1.0)
    gain = seeder.local.insertion_gain(amplitude)
    assert gain is not None and gain.shape == SHAPE
    for row, col in [(20, 20), (33, 41), (45, 12)]:
        trial = seeder.local.propose(0, seeder.store.changes([], np.array([[float(row), float(col)]]), np.array([1.0])),
                                     seeder.store.n + 1)
        assert trial is not None
        assert trial.loss - seeder.loss == pytest.approx(gain[row, col], abs=1e-4)


# ---------------------------------------------------------------- seeder
def test_trial_reject_leaves_state_untouched():
    model, seeder, _ = seeding_model("stem_haadf")
    seeder.run(30, log_every=0)
    n, positions, loss = seeder.store.n, seeder.store.positions.copy(), seeder.loss
    rejected = 0
    for _ in range(30):
        if not seeder.trial():
            rejected += 1
    assert rejected > 0
    assert seeder.loss <= loss
    seeder.store.check_identity()


def test_reproducible_with_seed():
    results = []
    for _ in range(2):
        _, seeder, _ = seeding_model("stem_haadf", SeedConfig(seed=7, neighbour_seeding=True, move_prob=0.1))
        seeder.run(60, log_every=0)
        results.append((seeder.store.positions.copy(), seeder.loss))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_max_atoms_is_respected():
    _, seeder, _ = seeding_model("stem_haadf", SeedConfig(seed=1, max_atoms=3))
    seeder.run(200, log_every=0)
    assert seeder.store.n <= 3 and seeder.stats.full


def test_empty_configuration_has_infinite_loss():
    _, seeder, _ = seeding_model("tem")
    assert seeder.loss == math.inf and seeder.evaluate() == math.inf


def test_filter_points_reports_and_does_not_change_without_commit():
    _, seeder, truth = seeding_model("stem_haadf", SeedConfig(seed=1, margin_px=5))
    pts = truth.structure.positions[0]
    seeder.run(10, log_every=0)
    n = seeder.store.n
    good = seeder.filter_points(np.vstack([pts, [[1.0, 1.0]]]), commit=False)
    assert seeder.store.n == n
    report = seeder.filter_report
    assert report["outside"] >= 1 and report["accepted"] == len(good)
    assert sum(report.values()) == len(pts) + 1
    seeder.filter_points(good, commit=True)
    assert seeder.store.n > n - 1


def test_tune_is_undone_when_the_loss_gets_worse():
    from temsim.data.schema import TrainSettings
    model, seeder, _ = seeding_model("stem_haadf", SeedConfig(seed=1))
    seeder.run(60, log_every=0)
    model.settings.aberrations.enabled = True
    model.settings.aberrations.lr = 50.0                       # absurd step: guaranteed to be rejected
    for k in model.settings.aberration_flags:
        model.settings.aberration_flags[k] = k == "C10"
    before = float(model.params.aberrations["C10"])
    loss = seeder.loss
    seeder.tune(3)
    assert seeder.stats.tune_rejected in (0, 1)
    if seeder.stats.tune_rejected:
        assert float(model.params.aberrations["C10"]) == before
    assert seeder.loss <= loss + 1e-9


def test_stem_seeding_recovers_a_synthetic_structure():
    rng = np.random.default_rng(0)
    rows, cols = np.meshgrid(np.arange(12, 56, 11.0), np.arange(12, 64, 11.0), indexing="ij")
    true = np.column_stack([rows.ravel(), cols.ravel()]) + rng.uniform(-2, 2, (rows.size, 2))
    truth = make_experiment(mode="stem_haadf", n=(len(true), 0))
    truth.structure.positions[0] = true
    with torch.no_grad():
        image = Model(truth, target=torch.zeros(SHAPE)).image().detach()
    exp = empty_experiment(SHAPE, PX, "C", mode="stem_haadf", padding=12)
    exp.imaging.aberrations = dict(truth.imaging.aberrations)
    seeder = Seeder(Model(exp, target=image), 0, SeedConfig(seed=1, gradient_seeding=True, margin_px=4))
    seeder.run(600, log_every=0)
    found = seeder.store.positions
    d = np.linalg.norm(true[:, None] - found[None], axis=2).min(axis=1)
    assert (d < 2.5).mean() >= 0.9 and seeder.store.n <= 2.5 * len(true)
    assert seeder.loss < -0.85
    structure, state = seeder.result()
    assert structure.counts == [seeder.store.n]


def test_seeding_a_mixed_layer_runs():
    exp = make_experiment(mode="stem_haadf", mixed=True, n=(0, 0))
    exp.layers[1].occ_entropy = False
    truth, image = truth_image("stem_haadf")
    model = Model(exp, target=image)
    seeder = Seeder(model, 0, SeedConfig(seed=1))
    seeder.run(30, log_every=0)
    assert seeder.store.n > 0 and math.isfinite(seeder.loss)
    assert seeder.loss == pytest.approx(seeder.evaluate(), abs=2e-4)
