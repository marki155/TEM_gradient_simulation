"""Part IV, Section 2: the workspace directory (listing, relative paths, save by name)."""

import pytest

from temsim.service import workspace
from temsim.service.client import Client, RemoteError
from temsim.service.jobs import Clock
from temsim.service.server import Server

from .helpers import make_folder, make_run


def build(root):
    make_run(root / "runs" / "alt")
    make_folder(root / "daten" / "probe1", with_lattice=False)
    (root / "leer").mkdir()
    (root / "runs" / "alt" / "autosave").mkdir()
    (root / "runs" / "neu.writing").mkdir()


def test_scan_lists_runs_and_datasets_but_not_autosave_or_staging(tmp_path):
    build(tmp_path)
    found = {e["path"]: e for e in workspace.scan(str(tmp_path))}
    assert set(found) == {"runs/alt", "daten/probe1"}
    assert found["runs/alt"]["kind"] == "run" and found["daten/probe1"]["kind"] == "dataset"
    assert not found["daten/probe1"]["autosave"]


def test_child_stays_inside_the_workspace(tmp_path):
    assert workspace.child(str(tmp_path), "a/b").endswith("a/b")
    for bad in ("", "..", "../x", "a/../../x", ".hidden", "autosave", "x.writing"):
        with pytest.raises(ValueError):
            workspace.child(str(tmp_path), bad)


@pytest.fixture
def core(tmp_path):
    build(tmp_path / "ws")
    server = Server("tcp://127.0.0.1:*", device="cpu", clock=Clock(frame_interval=0.0, save_interval=0),
                    workspace=str(tmp_path / "ws"))
    server.start()
    client = Client(server.endpoint, timeout=60.0)
    yield tmp_path / "ws", client
    client.close()
    server.stop()


def test_listing_opening_by_relative_path_and_saving_by_name(core):
    root, client = core
    assert client.info()["info"] == {"open": False, "workspace": str(root)}
    listing = client.workspace()
    assert listing["root"] == str(root) and {e["path"] for e in listing["entries"]} == {"runs/alt", "daten/probe1"}
    info = client.open_dataset("runs/alt")
    assert info["kind"] == "run" and client.info()["info"]["workspace"] == str(root)
    saved = client.save(name="kopie")
    assert saved == str(root / "kopie") and (root / "kopie" / "experiment.json").exists()
    assert "kopie" in {e["path"] for e in client.workspace()["entries"]}
    client.save(name="kopie")                                       # a run may be overwritten
    with pytest.raises(RemoteError):
        client.save(name="daten/probe1")                            # a data folder may not
    with pytest.raises(RemoteError):
        client.save(name="../draussen")
    assert (root / "daten" / "probe1" / "target.npy").exists()


def test_without_a_workspace_names_and_listing_are_refused(tmp_path):
    server = Server("tcp://127.0.0.1:*", device="cpu", clock=Clock(frame_interval=0.0, save_interval=0))
    server.start()
    try:
        with Client(server.endpoint, timeout=30.0) as client:
            assert client.info()["info"]["workspace"] is None
            with pytest.raises(RemoteError):
                client.workspace()
    finally:
        server.stop()
