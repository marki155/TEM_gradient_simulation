"""Part IV, Sections 4 and 5: server and client over a real socket."""

import threading
import time

import numpy as np
import pytest

from temsim.service.client import Client, RemoteError
from temsim.service.jobs import Clock
from temsim.service.server import Server

from .helpers import make_folder, make_run


class Events:
    def __init__(self):
        self.items = []
        self.lock = threading.Lock()

    def __call__(self, name, data, arrays):
        with self.lock:
            self.items.append((name, data, arrays))

    def names(self):
        with self.lock:
            return [n for n, _, _ in self.items]

    def wait(self, predicate, timeout=60.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self.lock:
                for item in self.items:
                    if predicate(*item):
                        return item
            time.sleep(0.02)
        raise AssertionError(f"event did not arrive; got {self.names()[-10:]}")


@pytest.fixture
def core():
    server = Server("tcp://127.0.0.1:*", device="cpu", clock=Clock(frame_interval=0.0, save_interval=0))
    server.start()
    events = Events()
    client = Client(server.endpoint, on_event=events, timeout=60.0)
    yield server, client, events
    client.close()
    server.stop()


def test_hello_and_errors_do_not_stop_the_server(core):
    server, client, _ = core
    hello = client.hello(extras=["optics"])
    assert hello["protocol"] == 1 and "open_run" in hello["commands"] and hello["info"] == {"open": False, "workspace": None}
    with pytest.raises(RemoteError, match="nothing is open") as exc:
        client.frame()
    assert "Traceback" in exc.value.trace
    with pytest.raises(RemoteError, match="unknown command"):
        client.call("no_such_command")
    assert client.info()["status"]["state"] == "idle"


def test_open_frame_and_train_a_fixed_number_of_steps(core, tmp_path):
    server, client, events = core
    path, exp, target = make_run(tmp_path / "run")
    client.hello(extras=["optics"])
    info = client.open_run(path)
    assert info["kind"] == "run" and client.get_experiment().layers[0].name == "top"
    assert len(client.get_experiment().structure.positions[0]) == 30
    client.get_state()
    train = client.get_train()
    np.testing.assert_allclose(train.targets.get("g1", target) if train.targets else target, target)
    assert train.masks["g1"].dtype == bool

    data, arrays = client.frame(["gradients"])
    assert "grad.0" in arrays and data["steps"] == 0

    status = client.start_training(steps=6)
    assert status["state"] == "running"
    _, last, _ = events.wait(lambda n, d, a: n == "light" and d.get("done"))
    assert last["step"] == 6 and last["kind"] == "training"
    events.wait(lambda n, d, a: n == "status" and d["state"] == "idle")
    name, data, arrays = events.wait(lambda n, d, a: n == "frame" and d["steps"] == 6)
    assert "ctf_phase" in arrays and "sim" in arrays      # the extras of hello
    assert client.info()["info"]["steps"] == 6


def test_pause_resume_stop_and_edit_between_steps(core, tmp_path):
    server, client, events = core
    path, *_ = make_run(tmp_path / "run")
    client.hello()
    client.open_run(path)
    client.start_training()                                   # no limit
    events.wait(lambda n, d, a: n == "light" and d.get("step", 0) >= 2)
    assert client.call("pause").result["status"]["state"] == "paused"
    steps = client.info()["info"]["steps"]
    time.sleep(0.3)
    assert client.info()["info"]["steps"] == steps            # paused: nothing advances
    report = client.edit("remove_low_occupancy", threshold=0.0)   # occupancies are >= 0: nothing is removed
    assert report["atoms"][0] == 30
    client.set_param("groups.g1.shift", lr=0.01)
    client.set_loss(ncc=0.3)
    assert client.get_train().loss.ncc == 0.3
    client.resume()
    events.wait(lambda n, d, a: n == "light" and d.get("step", 0) >= steps + 3)
    client.stop()
    assert client.info()["status"]["state"] == "idle"
    steps = client.info()["info"]["steps"]
    time.sleep(0.2)
    assert client.info()["info"]["steps"] == steps


def test_set_train_roundtrip_keeps_masks(core, tmp_path):
    _, client, _ = core
    path, *_ = make_run(tmp_path / "run")
    client.open_run(path)
    train = client.get_train()
    train.loss.ncc = 0.9
    client.set_train(train, with_arrays=False)
    again = client.get_train()
    assert again.loss.ncc == 0.9 and "g1" in again.masks


def test_seeding_over_the_wire(core, tmp_path):
    _, client, events = core
    client.hello()
    client.open_dataset(make_folder(tmp_path / "f", mode="stem_haadf", with_lattice=False), route="seed", border_px=5)
    client.seeding_config({})
    client.start_seeding(trials=20, chunk=10)
    _, light, _ = events.wait(lambda n, d, a: n == "light" and d.get("kind") == "seeding" and d.get("done"))
    assert light["trials"] == 20
    events.wait(lambda n, d, a: n == "status" and d["state"] == "idle")
    points = client.minima(threshold=0.2)
    assert points.ndim == 2
    report = client.insert_minima(threshold=0.2)
    assert report["atoms"] >= report["atoms_before"]


def test_save_and_autosave(core, tmp_path):
    server, client, events = core
    path, *_ = make_run(tmp_path / "run")
    client.hello()
    client.open_run(path)
    saved = client.save(str(tmp_path / "copy"))
    assert saved == str(tmp_path / "copy") and (tmp_path / "copy").is_dir()
    client.start_training(steps=2)
    events.wait(lambda n, d, a: n == "log" and "saved" in d["text"])         # autosave at the end of a job
    assert (tmp_path / "run" / "autosave").is_dir()


def test_a_failing_job_is_reported_and_ends(core, tmp_path):
    server, client, events = core
    path, *_ = make_run(tmp_path / "run")
    client.hello()
    client.open_run(path)
    original = server.session.trainer.step

    def broken(*args, **kwargs):
        raise RuntimeError("boom")
    server.session.trainer.step = broken
    client.start_training(steps=3)
    _, err, _ = events.wait(lambda n, d, a: n == "error")
    assert "boom" in err["error"] and err["where"] == "job"
    assert client.info()["status"]["state"] == "idle"
    server.session.trainer.step = original
    assert client.info()["info"]["open"]


def test_two_clients_both_get_events(core, tmp_path):
    server, client, events = core
    other_events = Events()
    other = Client(server.endpoint, on_event=other_events)
    try:
        client.hello(); other.hello()
        path, *_ = make_run(tmp_path / "run")
        client.open_run(path)
        other_events.wait(lambda n, d, a: n == "status" and d["open"])
    finally:
        other.close()


def test_timeout_and_shutdown(tmp_path):
    server = Server("tcp://127.0.0.1:*", device="cpu")
    server.start()
    client = Client(server.endpoint, timeout=5.0)
    client.hello()
    client.shutdown()
    server._thread.join(10.0)
    assert not server._thread.is_alive()
    with pytest.raises(TimeoutError):
        client.call("info", timeout=0.5)
    client.close()
    with pytest.raises(RemoteError, match="closed"):
        client.call("info")


def test_values_over_the_wire(core, tmp_path):
    _, client, _ = core
    path, *_ = make_run(tmp_path / "run")
    client.hello()
    client.open_run(path)
    values = client.get_values()
    assert "optics.C10" in values
    out = client.set_values(**{"optics.C10": -17.5, "groups.g1.shift": [0.5, 0.25]})
    assert out["optics.C10"] == pytest.approx(-17.5)
    assert client.get_values()["groups.g1.shift"] == pytest.approx([0.5, 0.25])
    with pytest.raises(RemoteError, match="unknown value keys"):
        client.set_values(nonsense=1.0)


def test_events_save_the_model_and_reopening_continues(core, tmp_path):
    server, client, events = core
    folder = make_folder(tmp_path / "f", mode="stem_haadf", with_lattice=False)
    client.hello()
    client.open_dataset(folder, border_px=5)
    autosave = tmp_path / "f" / "config_random" / "autosave"
    assert not autosave.exists()
    client.insert_minima(threshold=0.2)                          # an edit saves at once
    events.wait(lambda n, d, a: n == "log" and "(insert_minima)" in d["text"])
    atoms = client.info()["info"]["layers"][0]["atoms"]
    assert atoms > 0 and autosave.is_dir()

    client.start_training(steps=1000000)
    client.pause()
    events.wait(lambda n, d, a: n == "log" and "(pause)" in d["text"])
    client.stop()
    events.wait(lambda n, d, a: n == "log" and "(stop)" in d["text"])

    again = client.open_dataset(folder, border_px=5)             # same folder: continues
    assert again["layers"][0]["atoms"] == atoms and any("continued from" in n for n in again["notes"])
    fresh = client.open_dataset(folder, resume=False, border_px=5)
    assert fresh["layers"][0]["atoms"] == 0
    reopened = client.open_run(str(autosave))                    # an autosave is a run, saved to its own place
    assert reopened["out_dir"] == str(tmp_path / "f" / "config_random")


def test_training_sends_partial_frames_of_its_own_steps_and_waits_for_the_acknowledgement(tmp_path):
    server = Server("tcp://127.0.0.1:*", device="cpu", frame_steps=3,
                    clock=Clock(frame_interval=3600.0, save_interval=0))
    server.start()
    events = Events()
    client = Client(server.endpoint, on_event=events, timeout=60.0, auto_ack=False)
    try:
        path, *_ = make_run(tmp_path / "run")
        client.hello(extras=[])
        client.open_run(path)
        client.start_training()
        events.wait(lambda n, d, a: n == "frame")                       # the first one is the complete frame
        time.sleep(1.0)
        frames = [(d, a) for n, d, a in events.items if n == "frame"]
        assert len(frames) == 2                                         # two unanswered: the core waits
        assert all("seq" in d for d, _ in frames) and frames[-1][0].get("partial")
        assert frames[-1][0]["steps"] > 0 and "sim" in frames[-1][1] and "atoms.0.pos" in frames[-1][1]
        for d, _ in frames:
            client.ack_frame(d["seq"])
        events.wait(lambda n, d, a: n == "frame" and d["seq"] > frames[-1][0]["seq"])
        client.stop()
    finally:
        client.close()
        server.stop()


def test_a_client_that_acknowledges_by_itself_keeps_getting_frames(core, tmp_path):
    server, client, events = core
    server.frame_steps = 2
    server.clock.frame_interval = 3600.0
    path, *_ = make_run(tmp_path / "run")
    client.hello(extras=[])
    client.open_run(path)
    client.start_training()
    events.wait(lambda n, d, a: n == "frame" and d.get("partial") and d["steps"] >= 6)
    client.stop()


def test_the_target_travels_once_and_the_client_restores_float32(core, tmp_path):
    server, client, events = core
    server.frame_steps = 2
    server.clock.frame_interval = 3600.0
    path, *_ = make_run(tmp_path / "run")
    client.hello(extras=[])
    client.open_run(path)
    client.start_training()
    events.wait(lambda n, d, a: n == "frame" and d.get("partial") and d["steps"] >= 8)
    client.stop()
    full = client.frame()[1]["target"]
    frames = [(d, a) for n, d, a in events.items if n == "frame" and d.get("partial")]
    assert len(frames) >= 3
    for d, a in frames:
        assert a["target"].dtype == np.float32 and a["sim"].dtype == np.float32
        np.testing.assert_allclose(a["target"], full, rtol=2e-3, atol=2e-3)
        assert a["mask"].dtype == np.uint8
    assert len({d["target_key"] for d, _ in frames}) == 1
    assert not any(n == "target" for n in events.names())      # consumed by the client, not forwarded


def test_a_client_that_does_not_announce_acknowledgements_is_not_held_back(tmp_path):
    server = Server("tcp://127.0.0.1:*", device="cpu", frame_steps=2,
                    clock=Clock(frame_interval=3600.0, save_interval=0))
    server.start()
    events = Events()
    client = Client(server.endpoint, on_event=events, timeout=60.0, auto_ack=False)
    try:
        path, *_ = make_run(tmp_path / "run")
        client.call("hello", {"extras": []})                            # an older client: no ``acks``
        client.open_run(path)
        client.start_training()
        events.wait(lambda n, d, a: n == "frame" and d.get("partial") and d["steps"] >= 8)
        client.stop()
    finally:
        client.close()
        server.stop()
