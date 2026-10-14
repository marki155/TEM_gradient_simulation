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
    window.control.set_mode("insert")
    window.control.seed_btn.click()
    pump(lambda: window.controller.job_kind == "seeding")
    assert "ux" not in window.canvas.keys()
    window.seeding.spins["move_sigma_A"].setValue(0.3)
    pump(lambda: window.controller.state == "running" or window.controller.state == "idle")
    window.control.stop_btn.click()
    pump(lambda: window.controller.state == "idle")
    window.control.set_mode("training")
    window.control.start_btn.click()
    pump(lambda: "ux" in window.canvas.keys())
    window.control.stop_btn.click()
    pump(lambda: window.controller.state == "idle")


def test_minima_marks(window, pump):
    window.control.set_mode("insert")
    window.insert.method_combo.setCurrentIndex(window.insert.method_combo.findData("minima"))
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


def test_loss_blit_does_not_stack_ticks(window):
    canvas = window.canvas
    canvas.draw()
    canvas.history.reset()
    for n, step in enumerate(range(0, 4000, 40)):
        canvas.history.add(step, -0.01 - step / 11000.0)
        if n % 10 == 0:
            canvas.update_loss()
    canvas.update_loss()
    height = canvas.get_width_height(physical=True)[1]
    box = canvas._loss_box
    rows, columns = slice(height - int(box.y1), height - int(box.y0)), slice(0, 280)

    def ink():
        return int((np.asarray(canvas.buffer_rgba())[rows, columns, :3].max(axis=2) < 100).sum())

    incremental = ink()
    canvas.draw()
    assert incremental < 1.25 * ink()


def test_only_what_the_mode_needs_is_shown(window, pump):
    w, c = window, window.control

    def shown(widget):
        return not widget.isHidden()

    assert c.mode == "training"
    assert shown(c.start_btn) and not shown(c.seed_btn) and shown(c.gradient_box) and shown(w.editor_scroll)
    assert not shown(w.insert) and shown(w.strain_section)
    assert not shown(c.pause_btn) and not shown(c.resume_btn) and not shown(c.stop_btn)

    c.set_mode("insert")
    assert shown(w.insert) and shown(c.seed_btn) and not shown(c.start_btn)
    assert shown(w.seeding) and not shown(w.minima)
    assert not shown(c.gradient_box) and not shown(c.edit_section)
    assert not shown(w.editor_scroll) and not shown(w.strain_section)
    w.insert.method_combo.setCurrentIndex(w.insert.method_combo.findData("minima"))
    assert shown(w.minima) and not shown(w.seeding) and not shown(c.seed_btn)

    c.set_mode("training")
    w.insert.method_combo.setCurrentIndex(0)
    c.start_btn.click()
    pump(lambda: w.controller.state == "running")
    assert shown(c.pause_btn) and shown(c.stop_btn) and not shown(c.resume_btn)
    assert not c.mode_combo.isEnabled()
    c.pause_btn.click()
    pump(lambda: w.controller.state == "paused")
    assert shown(c.resume_btn) and not shown(c.pause_btn)
    c.stop_btn.click()
    pump(lambda: w.controller.state == "idle")


def test_insertion_options_follow_the_strategy_and_stay_in_sync(window, pump):
    s = window.seeding
    form = s.form
    assert form.isRowVisible(s.spins["max_atoms"])
    assert not form.isRowVisible(s.spins["bond_A"]) and not form.isRowVisible(s.spins["gradient_temperature"])
    s.strategy.setCurrentIndex(s.strategy.findData("neighbour"))
    assert form.isRowVisible(s.spins["bond_A"]) and not form.isRowVisible(s.spins["gradient_temperature"])
    s.set_stem(False)
    assert s.strategy.findData("gradient") < 0 and s.strategy.currentData() == "neighbour"
    s.set_stem(True)
    assert s.strategy.findData("both") >= 0
    s.show_values({"neighbour_seeding": True, "gradient_seeding": True})
    assert s.strategy.currentData() == "both" and form.isRowVisible(s.spins["gradient_temperature"])
    window.insert.show_values({"touch_A": 0.7, "merge_A": 1.3})
    assert window.insert.collision["touch_A"].value() == 0.7
    assert "touch_A" not in s.spins


def test_workspace_loading_and_saving_by_name(tmp_path, qapp, pump):
    from temsim.service.jobs import Clock
    from temsim.service.server import Server
    from temsim_gui.open_dialog import OpenDialog

    make_run(tmp_path / "ws" / "alt", n=(20, 12))
    server = Server("tcp://127.0.0.1:*", device="cpu", clock=Clock(frame_interval=0.0, save_interval=0),
                    workspace=str(tmp_path / "ws"))
    server.start()
    connection = Connection(server.endpoint, timeout=60.0)
    window = MainWindow(connection)
    try:
        c = window.controller
        pump(lambda: c.workspace is not None)
        listing = {}
        c.send(c.client.workspace, listing.update)
        pump(lambda: bool(listing))
        dialog = OpenDialog(listing)
        assert dialog.tree.topLevelItemCount() == 1 and dialog.selected() is None
        dialog.tree.topLevelItem(0).setSelected(True)
        entry = dialog.selected()
        assert entry["path"] == "alt" and dialog.open_button.isEnabled() and dialog.resume
        c.open_entry(entry)
        pump(lambda: c.experiment is not None and window.canvas.frame is not None)
        c.save_named("kopie")
        pump(lambda: (tmp_path / "ws" / "kopie" / "experiment.json").exists())
    finally:
        c.finish(False)
        connection.close()
        window.close()
        server.stop()
