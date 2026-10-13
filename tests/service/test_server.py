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
    assert hello["protocol"] == 1 and "open_run" in hello["commands"] and hello["info"] == {"open": False}
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

    def broken():
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
