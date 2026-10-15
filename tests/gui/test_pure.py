"""Part V, Sections 1-3: what needs no display."""

import numpy as np

from temsim_gui import overlays, tiles
from temsim_gui.frame_model import FrameModel, LossHistory


def wire(n=5, mode="tem", gradients=False):
    rng = np.random.default_rng(0)
    data = {"loss": 0.5, "data": {"ncc": 0.4}, "prior": {"entropy": 0.1}, "steps": 7, "mode": mode, "active_group": "a", "revision": 3,
            "layers": [{"name": "l0", "group": "a", "element": "C"}, {"name": "l1", "group": "b", "element": "W"}]}
    arrays = {"sim": rng.random((8, 8)), "target": rng.random((8, 8)), "diff": rng.random((8, 8)),
              "mask": np.ones((8, 8), np.uint8)}
    for i in range(2):
        arrays[f"atoms.{i}.pos"] = rng.random((n, 2)) * 8
        arrays[f"atoms.{i}.occ"] = np.linspace(0.1, 1.0, n)
        arrays[f"atoms.{i}.int"] = rng.random(n)
        if gradients:
            arrays[f"grad.{i}"] = rng.normal(size=(n, 2))
    return data, arrays


def test_frame_model_filters_by_group_and_occupancy():
    frame = FrameModel.from_wire(*wire())
    assert frame.group_names == ["a", "b"] and not frame.stem
    assert frame.active_layers() == [0]
    pos, ints = frame.atoms(min_occupancy=0.5)
    assert len(pos) == len(ints) == 3
    assert len(frame.atoms(active_only=False)[0]) == 10
    hist, edges = frame.histogram()
    assert len(hist) == 2 and len(edges) == 102 and sum(h.sum() for h in hist) == 10
    weighted, _ = frame.histogram(weighted=True)
    assert len(weighted) == 2 and sum(h.sum() for h in weighted) == 10
    assert frame.gradients() is None


def test_frame_model_gradients():
    frame = FrameModel.from_wire(*wire(gradients=True))
    pos, grad = frame.gradients()
    assert pos.shape == grad.shape == (5, 2)


def test_loss_history_stays_bounded():
    history = LossHistory(limit=100)
    for i in range(10000):
        history.add(i, 1.0 / (1 + i))
    assert len(history) <= 100
    steps, values = history.arrays()
    assert steps[0] == 0 and np.all(np.diff(steps) > 0)


def test_tile_plans():
    assert tiles.tile_keys(2)[:5] == ["loss", "sim", "diff", "target", "ctf"]
    assert "ux" not in tiles.tile_keys(2, seeding=True) and "hist" in tiles.tile_keys(2, seeding=True)
    assert "ux" not in tiles.tile_keys(2) and tiles.tile_keys(2)[-2:] == ["hist", "hist_w"]
    plan = tiles.plan(None, 2)
    assert plan.columns == 5 and plan.rows == -(-len(plan.keys) // 5)
    assert tiles.plan(tiles.COMPARE, 2).share == ("sim", "target", "diff")
    assert tiles.plan("sim", 2).keys == ("sim", "loss") and tiles.plan("loss", 2).keys == ("loss",)
    assert tiles.titles(["x"], stem=True)["ctf"] == "Probe" and tiles.titles(["x"])["pot0"] == "Potential x"


def test_marker_size_and_arrows():
    assert overlays.scatter_size(None) == 0.5 and overlays.scatter_size(1.0) == 60.0
    assert overlays.scatter_size(1e6) == 0.5
    pos = np.array([[1.0, 1.0], [2.0, 3.0], [50.0, 50.0]])
    grad = np.array([[1.0, 0.0], [0.0, 2.0], [9.0, 9.0]])
    a = overlays.arrows(pos, grad, (0, 10), (0, 10), length=3.0, reference=2.0)
    assert len(a.columns) == 2
    assert np.isclose(np.hypot(a.d_column, a.d_row).max(), 3.0 * 1.0)
    assert a.d_row[0] < 0                                           # descent: against the gradient
    b = overlays.arrows(pos, grad, (0, 10), (0, 10), length=3.0, descent=False, relative_to_view=True)
    assert np.isclose(np.hypot(b.d_column, b.d_row).max(), 3.0) and b.d_row[0] > 0
    assert overlays.arrows(pos, grad, (100, 110), (0, 10), 3.0, reference=2.0) is None
    assert overlays.display_limits(np.ones(4)) [1] > 1.0 and overlays.display_limits(np.array([np.nan])) == (0.0, 1.0)


def test_locality_of_an_endpoint():
    from temsim_gui.main_window import is_local
    assert is_local("tcp://127.0.0.1:5555") and is_local("tcp://localhost:1") and is_local("ipc:///tmp/x")
    assert not is_local("tcp://10.0.0.5:5555") and not is_local("tcp://gpu-box:5555")


def test_partial_frame_inherits_the_extras_of_the_previous_one():
    data, arrays = wire(gradients=True)
    arrays["potential.0"] = np.ones((8, 8))
    arrays["ctf_phase"] = np.ones((4, 4))
    arrays["strain.ux"] = np.ones((4, 4))
    old = FrameModel.from_wire(data, arrays)
    data2, arrays2 = wire()
    data2["partial"], data2["seq"] = True, 5
    new = FrameModel.from_wire(data2, arrays2, old)
    assert new.seq == 5 and new.ctf_phase is old.ctf_phase and "ux" in new.strain
    assert new.layers[0].potential is old.layers[0].potential and new.layers[1].gradient is old.layers[1].gradient
    assert FrameModel.from_wire(*wire(), old).ctf_phase is None          # a complete frame replaces the extras
    data3, arrays3 = wire(n=4)
    data3["partial"] = True
    assert FrameModel.from_wire(data3, arrays3, old).layers[0].gradient is None   # other atom count: no stale arrows
