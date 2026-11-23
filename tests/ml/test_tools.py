"""The ML tools (temsim_ml): model directories, the predict window and its handoff, the training worker."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore

from temsim.learning import detection as det
from temsim_ml import models
from temsim_ml.predict import PredictorWindow, write_handoff
from temsim_ml.train import TrainWorker

from ..service.helpers import make_run


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _model(directory, name="m"):
    return det.save_detector(os.path.join(directory, name), det.UNet(2, 8, 2), {"scale": 2.0, "pixel_A": 0.2,
                                                                                    "val_f1": 0.5})


def test_list_and_promote(tmp_path):
    local = _model(str(tmp_path / "local"))
    assert [m["name"] for m in models.list_models(str(tmp_path / "local"))] == ["m"]
    destination = models.promote(local, target_dir=str(tmp_path / "global"))
    assert det.is_detector(destination) and models.list_models(str(tmp_path / "global"))[0]["meta"]["scale"] == 2.0
    assert models.list_models(str(tmp_path / "missing")) == []


def test_predict_window_hands_the_confirmed_positions_back(tmp_path, qapp):
    handoff = str(tmp_path / "handoff.npz")
    window = PredictorWindow("cpu", on_confirm=lambda p, i: write_handoff(handoff, p, i))
    try:
        window._load_model(_model(str(tmp_path)))
        rng = np.random.default_rng(0)
        window.set_images(rng.random((40, 48)), rng.random((40, 48)))
        window.threshold.setValue(0.0)                  # an untrained net: take every peak
        window.min_distance.setValue(0.2)
        window.predict()
        assert len(window.positions) > 0 and window.confirm_btn.isEnabled()
        window.confirm()
        with np.load(handoff) as data:
            assert data["positions"].shape == (len(window.positions), 2)
            assert np.allclose(data["intensity"], window.intensity)
    finally:
        window.close()


def test_train_worker_saves_a_local_model(tmp_path, qapp):
    path, _, target = make_run(tmp_path / "run")
    np.save(os.path.join(path, "i_target.npy"), (target ** 2).astype(np.float32))
    seen, result = [], {}
    worker = TrainWorker(path, "local1", str(tmp_path / "local"), steps=30, sigma_A=0.4, holdout=0.2,
                         device="cpu", base=8, depth=2, batch=2, patch=32)
    worker.progress.connect(lambda *a: seen.append(a))
    worker.done.connect(lambda ok, message, where: result.update(ok=ok, message=message, path=where))
    worker.run()                                        # in this thread
    assert result["ok"], result["message"]
    assert seen and seen[-1][:2] == (30, 30)
    meta = det.read_meta(result["path"])
    assert meta["steps"] == 30 and meta["sources"][0]["data"][0]["source"] == path and meta["base"] == 8
    assert meta["sigma_A"] == pytest.approx(0.4)


def _collection_with_entries(tmp_path, n=2):
    path, _, target = make_run(tmp_path / "run")
    np.save(os.path.join(path, "i_target.npy"), (target ** 2).astype(np.float32))
    from temsim_ml.train import run_snapshot
    snapshot = run_snapshot(path, "cpu")
    collection = det.Collection(str(tmp_path / "coll"))
    ids = [collection.add(snapshot, f"e{k}", ["Optimiert"] if k == 0 else ["Handverlesen"]) for k in range(n)]
    return collection, ids


def test_train_worker_continues_a_model_on_collection_entries(tmp_path, qapp):
    collection, ids = _collection_with_entries(tmp_path)
    base = det.save_detector(str(tmp_path / "global" / "g"), det.UNet(2, 8, 2),
                             {"scale": 5.0, "sigma_A": 0.5, "pixel_A": 0.2, "steps": 7, "sources": [{"old": 1}]})
    result = {}
    worker = TrainWorker("", "g", str(tmp_path / "local"), steps=10, sigma_A=0.3, holdout=0.2, device="cpu",
                         entries=ids, collection=collection.root, resume=base, batch=2, patch=32, base=32)
    worker.done.connect(lambda ok, message, where: result.update(ok=ok, message=message, path=where))
    worker.run()
    assert result["ok"], result["message"]
    meta = det.read_meta(result["path"])
    # the model's own architecture, width and scale stay; steps and sources accumulate
    assert meta["base"] == 8 and meta["scale"] == 5.0 and meta["sigma_A"] == 0.5 and meta["steps"] == 17
    assert len(meta["sources"]) == 2 and [d["entry"] for d in meta["sources"][1]["data"]] == ids


def test_trainer_window_lists_tags_and_chooses_entries(tmp_path, qapp):
    from temsim_ml.train import TrainerWindow
    collection, ids = _collection_with_entries(tmp_path)
    window = TrainerWindow(device="cpu", collection=collection.root)
    try:
        window.use_collection.setChecked(True)
        assert window.entry_list.count() == 2 and sorted(window.chosen_entries()) == sorted(ids)
        window.filter_combo.setCurrentIndex(window.filter_combo.findData("Optimiert"))
        assert window.chosen_entries() == [ids[0]]                     # only the shown group is trained
        window.entry_list.item(0).setCheckState(QtCore.Qt.Unchecked)
        assert window.chosen_entries() == [] and not window.train_btn.isEnabled()
        window.filter_combo.setCurrentIndex(0)                         # all: the tick is remembered
        assert window.chosen_entries() == [ids[1]]
        window.entry_list.selectAll()
        window.tag_combo.setEditText("Handverlesen")
        window.tag_selected(True)
        assert all("Handverlesen" in e["groups"] for e in collection.entries())
    finally:
        window.close()
