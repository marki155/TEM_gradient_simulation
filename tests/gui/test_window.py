"""Part V, Sections 5-12: the window against a real core (offscreen)."""

import numpy as np
import pytest

from temsim.service.client import Client

from ..service.helpers import make_run

from temsim_gui.connection import Connection
from temsim_gui.main_window import MainWindow


@pytest.fixture
def window(server, tmp_path, qapp, pump):
    path, _, _ = make_run(tmp_path / "run", n=(20, 12))
    with Client(server.endpoint, timeout=60.0) as c:
        c.open_run(path)
    connection = Connection(server.endpoint, timeout=60.0)
    w = MainWindow(connection)
    pump(lambda: w.controller.experiment is not None and w.canvas.frame is not None)
    yield w
    w.controller.finish(False)
    connection.close()
    w.close()


def test_opens_and_fills_panels(window, pump):
    c = window.controller
    assert [g.name for g in c.experiment.groups] == ["g1"]
    assert len(window.canvas.frame.layers) == 2 and window.canvas.frame.shape == (64, 72)
    assert window.control.tile_combo.count() > 3
    pump(lambda: bool(window.values.current()))
    assert window.train.touched is False


def test_edit_buttons_round_trip(window, pump):
    window.control.remove_spin.setValue(2.0)
    count = sum(len(layer.pos) for layer in window.canvas.frame.layers)
    revision = window.canvas.frame.revision
    window.control.remove_btn.click()
    pump(lambda: window.canvas.frame.revision != revision)
    assert sum(len(layer.pos) for layer in window.canvas.frame.layers) < count


def test_training_runs_and_stops(window, pump):
    window.control.start_btn.click()
    pump(lambda: window.controller.state == "running")
    pump(lambda: len(window.canvas.history) > 3)
    window.control.stop_btn.click()
    pump(lambda: window.controller.state == "idle")
    assert np.isfinite(window.canvas.history.arrays()[1]).all()


def test_seeding_view_and_live_patch(window, pump):
    window.control.seed_btn.click()
    pump(lambda: window.controller.job_kind == "seeding")
    assert "ux" not in window.canvas.keys()
    window.seeding.spins["move_sigma_A"].setValue(0.3)
    pump(lambda: window.controller.state == "running" or window.controller.state == "idle")
    window.control.stop_btn.click()
    pump(lambda: window.controller.state == "idle")
    window.control.start_btn.click()
    pump(lambda: "ux" in window.canvas.keys())
    window.control.stop_btn.click()
    pump(lambda: window.controller.state == "idle")


def test_minima_marks(window, pump):
    window.minima.show_check.setChecked(True)
    pump(lambda: window.canvas.minima_points is not None)
    assert window.canvas.minima_points.shape[1] == 2


def test_spawned_core_and_launch_arguments(tmp_path, qapp, pump):
    from temsim_gui.connection import spawn_core
    from temsim_gui.launch import parse
    assert parse(["--run", "x", "--device", "cpu"]).device == "cpu"
    core = spawn_core("cpu")
    try:
        path, _, _ = make_run(tmp_path / "run", n=(10, 6))
        connection = Connection(core.endpoint, timeout=60.0)
        window = MainWindow(connection, core)
        window.controller.open_run(path)
        pump(lambda: window.canvas.frame is not None)
        window.close()
        pump(lambda: core.process.poll() is not None, timeout=30)
    finally:
        core.stop()
