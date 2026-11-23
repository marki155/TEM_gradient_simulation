"""The atom detector: labels, training on a run, prediction, comparison, insertion, files."""

import csv
import os

import numpy as np
import pytest

from temsim.learning import detection as det
from temsim.service.detection import DetectionError
from temsim.service.jobs import DetectorJob
from temsim.service.session import Session

from .helpers import make_run

SMALL = {"steps": 12, "batch": 2, "patch": 32, "base": 8, "depth": 2, "val_every": 6, "seed": 0}


def test_heatmap_peaks_round_trip():
    positions = np.array([[10.3, 12.6], [30.0, 40.2], [31.5, 20.0]])
    heights = np.array([1.0, 2.0, 0.5])
    heat = det.heatmap((48, 56), positions, heights, 1.5)
    found, h = det.find_peaks(heat, 0.2, 2.0)
    assert len(found) == 3
    order = np.argsort(-heights)
    assert np.allclose(h, heights[order], atol=0.05)
    assert np.abs(found - positions[order]).max() < 0.35


def test_weak_peak_next_to_a_bright_one_survives():
    # 9 px apart, minimum distance 8 px: the flank of the bright atom is higher than the weak peak within 8 px
    positions = np.array([[30.0, 30.0], [36.4, 36.4]])
    heat = det.heatmap((64, 64), positions, np.array([2.5, 0.6]), 3.0)
    found, h = det.find_peaks(heat, 0.3, 8.0)
    assert len(found) == 2 and np.allclose(h, [2.5, 0.6], atol=0.03)


def test_match_is_one_to_one_and_closest_first():
    pred = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
    fit = np.array([[0.0, 0.8], [20.0, 20.0]])
    p, f = det.match(pred, fit, 1.0)
    assert p.tolist() == [False, True, False] and f.tolist() == [True, False]
    assert det.scores(pred, fit, 1.0)[0] == pytest.approx(1 / 3)


@pytest.fixture
def run(tmp_path):
    path, exp, target = make_run(tmp_path / "run")
    np.save(os.path.join(path, "i_target.npy"), (target ** 2).astype(np.float32))
    session = Session("cpu")
    session.set_workspace(str(tmp_path / "ws"))
    session.open_run(path)
    return session


def test_detector_needs_the_second_target(tmp_path):
    path, _, _ = make_run(tmp_path / "run")
    s = Session("cpu")
    s.set_workspace(str(tmp_path / "ws"))
    s.open_run(path)
    with pytest.raises(DetectionError, match="i_target"):
        DetectorJob(s, name="m", **SMALL)


def test_detector_job_trains_saves_predicts_and_continues(run):
    s = run
    job = DetectorJob(s, name="m1", sigma_A=0.4, **SMALL)
    while not job.done:
        status = job.advance()
    assert status["count"] == 12 and status["val_loss"] is not None and os.path.isdir(status["path"])
    assert [m["name"] for m in s.detection.models()] == ["m1"] and s.info()["detector"] == "m1"
    meta = det.read_meta(status["path"])
    assert meta["steps"] == 12 and len(meta["sources"]) == 1 and meta["scale"] > 0

    summary, arrays = s.detection.result()
    n = summary["predicted"]
    assert arrays["pred.pos"].shape == (n, 2) and arrays["pred.int"].shape == (n,)
    assert int(arrays["pred.new"].sum()) == summary["new"]
    assert arrays["fit.unconfirmed"].shape == (summary["unconfirmed"], 2)

    again = DetectorJob(s, name="m1", resume=True, **{**SMALL, "base": 16})   # the saved architecture wins
    assert again.trainer.net.config["base"] == 8
    while not again.done:
        again.advance()
    meta = det.read_meta(status["path"])
    assert meta["steps"] == 24 and len(meta["sources"]) == 2


def test_insert_uses_the_predicted_intensity_and_export_writes_a_table(run):
    s = run
    s.detection.trainer = None
    s.detection.net = det.UNet(2, 8, 2)
    s.detection.meta = {"scale": 6.0}
    s.detection.name = "fake"
    # a hand-made prediction: one atom where the fit has none, one on a fitted atom
    fitted, _ = s.detection.fitted_atoms()
    free = np.array([32.0, 36.0])
    s.detection.prediction = det_prediction(np.array([free, fitted[0]]), np.array([3.0, 6.0]))
    new, unconfirmed, _ = s.detection.compare(0.5)
    assert new.tolist() == [True, False] and len(unconfirmed) == len(fitted) - 1
    before = s.info()["layers"][0]["atoms"]
    report = s.detection.insert(layer=0, check_loss=False, check_collisions=False)
    assert report["candidates"] == 1 and report["accepted"] == 1
    lp = s.model.params.layers[0]
    assert len(lp) == before + 1 and float(lp.intensity[-1]) == pytest.approx(3.0, rel=1e-4)
    path = s.detection.export()
    rows = list(csv.DictReader(open(path)))
    assert {r["status"] for r in rows} >= {"matched", "unconfirmed"} and os.path.exists(
        os.path.join(os.path.dirname(path), "ml_heatmap.npy"))


def det_prediction(positions, intensity):
    from temsim.service.detection import Prediction
    return Prediction(positions, intensity, np.zeros((64, 72), np.float32), "fake")
