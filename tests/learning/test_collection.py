"""The collection of good reconstructions: entries as snapshots, groups as tags, entries as samples."""

import numpy as np
import pytest

from temsim.learning import detection as det


def _snapshot(pixel_A=0.2, n=12):
    rng = np.random.default_rng(1)
    positions = rng.uniform(4, 36, (n, 2))
    return {"target": rng.random((40, 48)), "second": rng.random((40, 48)), "valid": None,
            "positions": positions, "intensity": np.full(n, 3.0), "pixel_A": pixel_A, "source": "/runs/a"}


def test_entries_groups_and_tags(tmp_path):
    c = det.Collection(str(tmp_path))
    a = c.add(_snapshot(), "a", ["Optimiert"])
    b = c.add(_snapshot(), "b", [" Handverlesen ", "Optimiert"])
    c.create_group("Leer")
    assert c.groups() == ["Handverlesen", "Leer", "Optimiert"]
    assert {e["id"] for e in c.entries("Optimiert")} == {a, b} and [e["id"] for e in c.entries("Handverlesen")] == [b]
    c.tag([a], "Handverlesen")
    c.tag([b], "Optimiert", member=False)
    assert {e["id"] for e in c.entries("Handverlesen")} == {a, b} and [e["id"] for e in c.entries("Optimiert")] == [a]
    c.delete_group("Handverlesen")
    assert "Handverlesen" not in c.groups() and len(c.entries()) == 2          # entries stay
    c.remove(a)
    assert [e["id"] for e in c.entries()] == [b]
    with pytest.raises(det.CollectionError):
        c.load(a)


def test_a_snapshot_is_frozen_and_becomes_a_sample(tmp_path):
    c = det.Collection(str(tmp_path))
    snapshot = _snapshot(pixel_A=0.1)
    entry = c.add(snapshot, "a")
    snapshot["positions"][:] = 0.0                                             # later changes do not reach it
    loaded = c.load(entry)
    assert loaded["positions"].min() > 1 and loaded["valid"].all() and loaded["pixel_A"] == 0.1
    assert loaded["name"] == "a" and loaded["source"] == "/runs/a"
    # the label width is physical: 0.4 A at 0.1 A/px is 4 px; the validation tolerance is the entry's own
    sample = det.snapshot_sample(loaded, scale=3.0, sigma_A=0.4, holdout=0.2)
    assert sample.label.max() == pytest.approx(1.0, abs=0.05)
    assert (sample.tolerance_px, sample.min_distance_px) == det.validation_scales(loaded["positions"], 0.1)
    with pytest.raises(det.CollectionError):
        c.add({**_snapshot(), "positions": np.zeros((0, 2)), "intensity": np.zeros(0)})
