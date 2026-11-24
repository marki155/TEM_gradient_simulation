"""The batch tool: inputs and stacks, the job file, the runner (frames, warm start, resume, stop), the window."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import tifffile

from temsim.learning import detection as det
from temsim_ml.batch import job as jobs
from temsim_ml.batch import runner

from ..service.helpers import PX, SHAPE, smooth_target


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _data(tmp_path):
    """A folder with one image and a pair of stacks (3 frames)."""

    folder = tmp_path / "single"
    folder.mkdir()
    target = smooth_target()
    tifffile.imwrite(str(folder / "target.tif"), target)
    tifffile.imwrite(str(folder / "i_target.tif"), target ** 2)
    with open(folder / "init.json", "w") as f:
        json.dump({"pixel_size_nm": PX, "voltage_kV": 80.0, "padding": 12, "imaging": {"mode": "tem"}}, f)
    frames = np.stack([smooth_target(seed=k) for k in range(3)])
    tifffile.imwrite(str(tmp_path / "stack.tif"), frames, photometric="minisblack")
    tifffile.imwrite(str(tmp_path / "stack_i.tif"), frames ** 2, photometric="minisblack")
    model = det.save_detector(str(tmp_path / "models" / "m"), det.UNet(2, 8, 2), {"scale": 2.0, "pixel_A": PX * 10})
    return str(folder), model


def _job(tmp_path, model, steps=3):
    single = jobs.folder_input(str(tmp_path / "single"))
    stack = jobs.file_input(str(tmp_path / "stack.tif"), str(tmp_path / "stack_i.tif"))
    stack["warm_start"] = "all"
    return {"output": str(tmp_path / "out"), "device": "cpu",
            "template": {"pixel_size_nm": PX, "voltage_kV": 80.0, "mode": "tem"},
            "inputs": [single, stack],
            "pipeline": [jobs.make_step("ml", model=model, threshold=0.0, min_distance_A=2.0),
                         jobs.make_step("train", max_steps=steps, patience=0)]}


def test_inputs_frames_and_init(tmp_path):
    folder, model = _data(tmp_path)
    single = jobs.folder_input(folder)
    assert single["frames"] == 1 and single["i_target"].endswith("i_target.tif") and single["init"]
    stack = jobs.file_input(str(tmp_path / "stack.tif"), str(tmp_path / "stack_i.tif"))
    assert stack["frames"] == 3 and jobs.read_frames(stack["target"]).shape == (3,) + SHAPE
    template = {"pixel_size_nm": 0.5, "voltage_kV": 200.0, "mode": "stem_haadf"}
    # the folder's own file wins over the template; a value given for the input wins over both
    assert jobs.read_init(single, template)["pixel_size_nm"] == PX
    single["voltage_kV"] = 60.0
    assert jobs.read_init(single, template)["voltage_kV"] == 60.0
    assert jobs.read_init(stack, template)["imaging"]["mode"] == "stem_haadf"
    with pytest.raises(jobs.JobError):
        jobs.read_init(stack, {})
    job = _job(tmp_path, model)
    assert jobs.check_job(job) == []
    job["inputs"][1]["i_target"] = ""
    assert any("i_target" in p for p in jobs.check_job(job))
    assert [s["kind"] for s in jobs.default_pipeline()] == ["ml", "train"]


def test_runner_processes_images_and_stacks_and_resumes(tmp_path):
    _, model = _data(tmp_path)
    job = _job(tmp_path, model)
    jobs.save_job(job)
    assert runner.run_job(job["output"]) == "finished"
    out = job["output"]
    state = jobs.status(out)
    assert state["state"] == "finished" and state["done"] == 4 and state["failed"] == 0
    single, stack = job["inputs"]
    report = jobs.read_json(os.path.join(out, "single", jobs.REPORT_FILE))
    assert report["ok"] and [s["kind"] for s in report["steps"]] == ["ml", "train"]
    assert report["steps"][1]["steps"] == 3
    for name in ("atoms.csv", "loss.csv", "ml_detections.csv", "run/experiment.json"):
        assert os.path.exists(os.path.join(out, "single", name)), name
    assert tifffile.imread(os.path.join(out, "single", "atoms_target.tif")).shape == SHAPE + (3,)
    assert os.path.exists(os.path.join(out, "single", "atoms.png"))
    for k in range(3):
        assert jobs.frame_done(out, stack, k)
    later = jobs.read_json(os.path.join(jobs.frame_dir(out, stack, 2), jobs.REPORT_FILE))
    assert later["warm_start"] == "all" and any("vorigen Frames" in n for n in later["notes"])
    assert tifffile.imread(os.path.join(out, "stack", "atoms.tif")).shape == (3,) + SHAPE + (3,)
    assert len(os.listdir(os.path.join(out, "stack", "png"))) == 6

    # a second start skips what is done; a lost frame is done again, starting from the saved previous one
    os.remove(os.path.join(jobs.frame_dir(out, stack, 2), jobs.REPORT_FILE))
    assert runner.run_job(out) == "finished"
    log = open(os.path.join(out, jobs.LOG_FILE)).read()
    assert "(3 schon fertig)" in log and "stack Frame 2: Start (aus Vorgänger" in log


def test_errors_are_logged_and_stop_ends_the_job(tmp_path):
    _, model = _data(tmp_path)
    job = _job(tmp_path, model)
    job["pipeline"][0]["params"]["model"] = str(tmp_path / "missing")
    jobs.save_job(job)
    assert runner.run_job(job["output"]) == "finished"
    state = jobs.status(job["output"])
    assert state["failed"] == 4 and state["done"] == 0
    assert "kein Detektor" in jobs.read_json(os.path.join(job["output"], "single", jobs.REPORT_FILE))["error"]

    runner.request_stop(job["output"])
    r = runner.Runner(job["output"])
    r.check_stop = lambda: (_ for _ in ()).throw(runner.Stopped())
    assert r.run() == "stopped" and jobs.status(job["output"])["state"] == "stopped"


def test_window_builds_and_starts_a_job(tmp_path, qapp, monkeypatch):
    from temsim_ml.batch.window import BatchWindow
    _, model = _data(tmp_path)
    started = []
    monkeypatch.setattr(runner, "launch", lambda path: started.append(path) or 1)
    window = BatchWindow()
    try:
        assert [s["kind"] for s in window.pipeline] == ["ml", "train"]
        window.add_folder_inputs([str(tmp_path / "single")])
        window.add_input(jobs.file_input(str(tmp_path / "stack.tif"), str(tmp_path / "stack_i.tif")))
        window.add_input(jobs.file_input(str(tmp_path / "stack.tif"), str(tmp_path / "stack_i.tif")))
        assert [e["name"] for e in window.inputs] == ["single", "stack", "stack_2"]
        window.table.selectRow(2)
        window.remove_inputs()
        assert window.table.rowCount() == 2 and window.count_label.text() == "2 Eingaben, 4 Bilder"
        window.kind_combo.setCurrentIndex(window.kind_combo.findData("propeller"))
        window.add_step()
        window.move_step(-1)
        assert [s["kind"] for s in window.pipeline] == ["ml", "propeller", "train"]
        window.step_list.setCurrentRow(0)
        window._set_param(0, "model", model)
        window.output_edit.setText(str(tmp_path / "out"))
        window.pixel_spin.setValue(PX)
        assert window.start(confirm=False) and started == [str(tmp_path / "out")]
        saved = jobs.load_job(str(tmp_path / "out"))
        assert saved["pipeline"][0]["params"]["model"] == model and len(saved["inputs"]) == 2
        other = BatchWindow(str(tmp_path / "out"))
        assert [s["kind"] for s in other.pipeline] == ["ml", "propeller", "train"]
        other.close()
    finally:
        window.close()
