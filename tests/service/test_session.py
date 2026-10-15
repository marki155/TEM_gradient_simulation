"""Part IV, Section 2: the session."""

import numpy as np
import pytest
import torch

from temsim.data import store
from temsim.learning.trainer import StepResult
from temsim.service.session import Session, SessionError

from ..learning.helpers import SHAPE
from .helpers import make_folder, make_run


@pytest.fixture
def run(tmp_path):
    path, exp, target = make_run(tmp_path / "run")
    session = Session("cpu")
    session.open_run(path)
    return session, path, target


@pytest.fixture
def stem(tmp_path):
    session = Session("cpu")
    session.open_dataset(make_folder(tmp_path / "f", mode="stem_haadf", with_lattice=False), border_px=5)
    return session


def test_nothing_open():
    s = Session("cpu")
    assert s.info() == {"open": False, "workspace": None}
    for call in (s.experiment, s.frame, s.save, lambda: s.set_param("x")):
        with pytest.raises(SessionError, match="nothing is open"):
            call()


def test_open_run_describes_itself(run):
    s, path, _ = run
    info = s.info()
    assert info["open"] and info["kind"] == "run" and info["shape"] == list(SHAPE) and info["groups"] == ["g1"]
    assert [l["atoms"] for l in info["layers"]] == [30, 20]
    assert len(s.experiment().structure.positions[0]) == 30


def test_open_dataset_routes(tmp_path):
    s = Session("cpu")
    assert s.open_dataset(make_folder(tmp_path / "a"))["kind"] == "lattice"
    assert s.open_dataset(make_folder(tmp_path / "b", mode="stem_haadf", with_lattice=False))["kind"] == "seed"
    s.save(str(tmp_path / "out"))
    assert s.open_dataset(str(tmp_path / "out"))["kind"] == "run"
    with pytest.raises(SessionError, match="unknown route"):
        s.open_dataset(str(tmp_path / "a"), route="nonsense")


def test_frame_tem_with_all_extras(run):
    s, *_ = run
    data, arrays = s.frame(("optics", "strain", "potentials", "gradients"), standardised=True)
    for key in ("sim", "target", "diff", "mask", "x", "y", "ctf_phase", "ctf_ring", "atoms.0.pos", "atoms.1.occ",
                "atoms.0.int", "potential.0", "grad.0"):
        assert key in arrays, key
    assert arrays["sim"].shape == SHAPE and arrays["mask"].dtype == np.uint8
    assert arrays["atoms.0.pos"].shape == (30, 2) and arrays["grad.0"].shape == (30, 2)
    assert any(k.startswith("strain.") for k in arrays)
    assert np.isfinite(data["loss"]) and data["mode"] == "tem" and [l["atoms"] for l in data["layers"]] == [30, 20]
    # gradients must leave the flags alone: a step afterwards still works
    s.trainer.step()
    assert "probe" not in arrays


def test_frame_stem_has_a_probe(tmp_path):
    path, *_ = make_run(tmp_path / "r", mode="stem_haadf")
    s = Session("cpu")
    s.open_run(path)
    _, arrays = s.frame(("optics",))
    assert "probe" in arrays and "ctf_phase" not in arrays
    assert arrays["probe"].ndim == 2 and arrays["probe"].max() > 0


def test_frame_without_extras_is_small(run):
    s, *_ = run
    _, arrays = s.frame()
    assert not any(k.startswith(("strain", "potential", "grad", "ctf", "probe")) for k in arrays)


def test_set_train_is_in_place_and_keeps_masks(run):
    s, *_ = run
    settings = s.model.settings
    wire = s.model.settings
    import copy
    new = copy.deepcopy(wire)
    new.masks, new.targets = {}, {}
    new.loss.ncc = 0.123
    revision = s.revision
    s.set_train(new)
    assert s.model.settings is settings and settings.loss.ncc == 0.123
    assert "g1" in settings.masks and s.revision == revision + 1
    s.trainer.step()


def test_set_param_loss_regularisation_group(run):
    s, *_ = run
    s.set_param("groups.g1.shift", enabled=False, lr=0.2)
    p = s.model.settings.param("groups.g1.shift")
    assert not p.enabled and p.lr == 0.2
    with pytest.raises(SessionError, match="unknown trainable key"):
        s.set_param("nope")
    s.set_loss(ncc=0.7)
    assert s.model.settings.loss.ncc == 0.7
    with pytest.raises(SessionError, match="unknown loss weight"):
        s.set_loss(bogus=1)
    s.set_regularisation(repulsion_a=2.0)
    assert s.model.settings.regularisation.repulsion_a == 2.0
    with pytest.raises(SessionError, match="unknown prior weight"):
        s.set_regularisation(bogus=1)
    s.set_active_group(None)
    assert s.model.settings.active_group is None
    with pytest.raises(SessionError, match="unknown group"):
        s.set_active_group("zz")


def test_merge_edits_by_hand_and_by_the_training_job(run):
    from temsim.service.jobs import TrainingJob
    s, *_ = run
    params = s.model.params
    n = len(params.layers[0])
    with torch.no_grad():
        params.layers[0].base[1] = params.layers[0].base[0] + torch.tensor([0.0, 0.1])
        params.layers[0].logit.fill_(4.0)
    assert s.edit("merge", layer=0, atoms=[0, 1])["merged"] == 1
    assert len(params.layers[0]) == n - 1
    with torch.no_grad():
        params.layers[0].base[1] = params.layers[0].base[0] + torch.tensor([0.0, 0.1])
    report = s.edit("merge_close", distance_A=0.5)
    assert report["merged"] >= 1 and report["atoms"][0] <= n - 2
    with torch.no_grad():
        params.layers[0].base[1] = params.layers[0].base[0] + torch.tensor([0.0, 0.1])
    count = len(params.layers[0])
    s.model.settings.regularisation.merge_every = 2
    s.model.settings.regularisation.merge_distance_A = 0.5
    job = TrainingJob(s)
    job.advance()
    assert len(params.layers[0]) == count
    job.advance()
    job.advance()
    assert job.merged >= 1 and len(params.layers[0]) < count


def test_edits(run):
    s, *_ = run
    params = s.model.params
    with torch.no_grad():
        params.layers[0].shifts += 0.5
        params.layers[0].intensity *= 2
        params.layers[0].logit[:5] = -10.0
    s.edit("reset_free_positions")
    assert float(params.layers[0].shifts.abs().max()) == 0.0
    s.edit("reset_intensities")
    assert torch.allclose(params.layers[0].intensity, params.layers[0].reference)
    r = s.edit("remove_low_occupancy", threshold=0.5)
    assert r["removed"] == 5 and r["atoms"][0] == 25
    s.edit("reset_occupancies")
    s.trainer.step()                       # the optimiser follows the changed tensors
    r = s.edit("resample_control", control_points=[5, 6])
    assert s.model.experiment.group("g1").control_points == [5, 6] and "max_deviation_px" in r
    s.trainer.step()
    with pytest.raises(SessionError, match="unknown edit"):
        s.edit("nope")
    with pytest.raises(SessionError, match="control_points"):
        s.edit("resample_control", control_points=[3])


def test_minima_and_insert_on_a_stem_seed(stem):
    s = stem
    points = s.minima(threshold=0.2, distance_px=3.0)
    assert points.ndim == 2 and points.shape[1] == 2 and len(points) > 0
    r = s.insert_minima(threshold=0.2, distance_px=3.0)
    assert r["candidates"] == len(points) and 0 < r["atoms"] <= r["atoms_before"] + r["accepted"]
    assert s.info()["layers"][0]["atoms"] == r["atoms"]
    _, arrays = s.frame()
    assert len(arrays["atoms.0.pos"]) == r["atoms"]


def test_minima_options(stem):
    s = stem
    plain = s.minima(threshold=0.2, distance_px=3.0, atom_distance=False)
    near = s.minima(threshold=0.2, distance_px=3.0)
    assert len(plain) >= len(near)
    checked = s.minima(threshold=0.2, distance_px=3.0,
                       verify={"check_margin": True, "check_collisions": True, "check_loss": False})
    assert len(checked) <= len(near)


def test_showing_minima_does_not_remove_atoms(run):
    s, *_ = run
    params = s.model.params
    with torch.no_grad():
        params.layers[0].logit[:5] = -10.0
    n = len(params.layers[0])
    verify = {"check_margin": True, "check_collisions": True, "check_loss": True}
    for _ in range(2):                                   # the first call creates the seeder, the second one reuses it
        s.minima(threshold=0.2, distance_px=3.0, verify=verify)
        assert len(params.layers[0]) == n


def test_seeding_config_validates(stem):
    cfg = stem.seeding_config()
    assert isinstance(cfg, dict) and cfg
    key = next(k for k, v in cfg.items() if isinstance(v, float) and v > 0)
    assert stem.seeding_config({key: cfg[key] * 2})[key] == cfg[key] * 2
    with pytest.raises(SessionError, match="unknown seeding option"):
        stem.seeding_config({"bogus": 1})


def test_refine_switches_to_position_training(stem):
    stem.refine()
    p = stem.model.settings.layer(stem.model.experiment.layers[0].name)
    assert p.free_positions.enabled and p.intensities.enabled


def test_save_roundtrip(run, tmp_path):
    s, path, target = run
    for _ in range(3):
        s.trainer.step()
    out = s.save(str(tmp_path / "saved"))
    loaded = store.load_run(out)
    np.testing.assert_allclose(loaded.target, target)
    assert len(loaded.experiment.structure.positions[0]) == 30
    s2 = Session("cpu")
    s2.open_run(out)
    assert s2.model.loss()[0].item() == pytest.approx(s.model.loss()[0].item(), rel=1e-4)


def test_values_are_polar_and_roundtrip(run):
    s, *_ = run
    values = s.values()
    assert {"optics.amp_contrast", "optics.C10", "normalisation", "groups.g1.shift", "groups.g1.rotation",
            "groups.g1.scale", "groups.g1.contrast"} <= set(values)
    assert not any(k.split(".")[-1][:1] in "ab" and k.split(".")[-1][1:].isdigit() for k in values)
    c12 = [k for k in values if k.startswith("optics.C12")]
    if c12:
        assert "optics.phi12" in values
        out = s.set_values({"optics.C12": 1.5, "optics.phi12": 0.3})
        assert out["optics.C12"] == pytest.approx(1.5, rel=1e-4) and out["optics.phi12"] == pytest.approx(0.3, rel=1e-4)
    before = s.model.image().detach().clone()
    out = s.set_values({"optics.C10": -20.0, "groups.g1.shift": [1.0, -2.0], "groups.g1.rotation": 0.01,
                        "normalisation": [0.1, 0.2], "optics.amp_contrast": 0.3})
    assert out["optics.C10"] == pytest.approx(-20.0) and out["groups.g1.shift"] == pytest.approx([1.0, -2.0])
    assert s.model.params.normalisation.tolist() == pytest.approx([0.1, 0.2])
    assert not torch.allclose(before, s.model.image().detach())


def test_set_values_is_all_or_nothing(run):
    s, *_ = run
    old = s.values()
    with pytest.raises(SessionError, match="unknown value keys"):
        s.set_values({"optics.C10": -3.0, "optics.nonsense": 1.0})
    with pytest.raises(SessionError, match="needs shape"):
        s.set_values({"optics.C10": -3.0, "groups.g1.shift": [1.0]})
    assert s.values() == old


def test_live_seed_patch_keeps_the_seeder(tmp_path):
    from temsim.service.session import Session
    from .helpers import make_run
    session = Session(device="cpu")
    session.open_run(make_run(tmp_path / "r")[0])
    seeder = session.make_seeder()
    session.seeding_config({"move_sigma_A": 0.4})
    assert session.seeder is seeder and seeder.config.move_sigma_A == 0.4
    session.seeding_config({"margin_px": 3.0})
    assert session.seeder is None


def test_frame_images_are_dark_outside_the_loss_mask(tmp_path):
    s = Session("cpu")
    s.open_run(make_run(tmp_path / "r")[0])
    data, arrays = s.frame()
    outside = ~arrays["mask"].astype(bool)
    assert outside.any()
    for key in ("sim", "target", "diff"):
        assert np.all(arrays[key][outside] == 0.0) and np.any(arrays[key][~outside] != 0.0)
    s.model.settings.active_group = None          # no active group: the loss mask is all pixels
    _, arrays = s.frame()
    assert np.any(arrays["sim"][outside] != 0.0)


def test_minima_stay_inside_the_loss_mask(tmp_path):
    s = Session("cpu")
    s.open_run(make_run(tmp_path / "r")[0])
    s.model.settings.active_group = None
    everywhere = s.minima(threshold=0.1, distance_px=2.0, atom_distance=False)
    assert len(everywhere) > 0
    s.model.settings.active_group = next(iter(s.model.settings.masks))
    inside = s.minima(threshold=0.1, distance_px=2.0, atom_distance=False)
    rows, cols = inside[:, 0].astype(int), inside[:, 1].astype(int)
    _, arrays = s.frame()
    mask = arrays["mask"].astype(bool)
    assert (~mask).any() and mask[rows, cols].all()


def test_frame_from_a_training_step_equals_the_evaluated_frame(run):
    s, *_ = run
    kept = {}
    with torch.no_grad():                                    # a forward pass without a step: the state does not move
        loss, parts, prior = s.model.loss(kept)
    result = StepResult(3, float(loss), parts, prior, kept)
    data, arrays = Session.compose(s.capture(kept, result))
    full, expected = s.frame()
    for key in expected:
        np.testing.assert_allclose(arrays[key], expected[key], rtol=1e-5, atol=1e-6, err_msg=key)
    assert data["partial"] and data["steps"] == result.step and data["layers"] == full["layers"]
    assert abs(data["loss"] - full["loss"]) < 1e-5


def test_a_capture_keeps_the_atoms_of_its_step(run):
    s, *_ = run
    result = s.trainer.step(keep=True)
    capture = s.capture(result.kept, result)
    before = capture.atoms[0][0].clone()
    with torch.no_grad():
        s.model.params.layers[0].shifts += 1.0               # the optimiser moves the atoms in place
    assert torch.equal(capture.atoms[0][0], before)
