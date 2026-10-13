"""Tests of Part III (temsim.data): codec, store, legacy formats, dataset folders."""

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from temsim.data import codec, dataset, legacy, store
from temsim.data.schema import (Experiment, Group, Imaging, Layer, ModelState, OpticsState,
                                Structure, TrainSettings)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _p(*parts):
    return os.path.join(ROOT, *parts)


def _small_experiment():
    rng = np.random.default_rng(0)
    layers = [Layer("Gr", "C", 0.0, "Gr", 100.0, False), Layer("S", "S", 0.3, "WS2"),
              Layer("W", "W", 0.5, "WS2")]
    pos = [rng.uniform(0, 64, (n, 2)) for n in (5, 3, 0)]
    z = [np.full(n, v) for n, v in zip((5, 3, 0), (6.0, 16.0, 74.0))]
    return Experiment(Imaging((64, 48), 0.01, aberrations={"C10": -5.0}), layers,
                      [Group("Gr", init_rotation=0.1)], Structure(pos, z))


# -- codec / schema -----------------------------------------------------------

def test_experiment_roundtrip():
    e = _small_experiment()
    r = codec.round_trip(e)
    assert r.layer_names == e.layer_names and r.group_names == ["Gr", "WS2"]
    assert r.imaging.shape == (64, 48) and isinstance(r.imaging.shape, tuple)
    assert np.allclose(r.structure.positions[0], e.structure.positions[0])
    assert r.structure.positions[2].shape == (0, 2)


def test_train_settings_roundtrip_without_bulk_arrays():
    e = _small_experiment()
    t = TrainSettings().ensure_for(e)
    t.group("Gr").shift.enabled = True
    t.masks["Gr"] = np.ones((4, 4), bool)
    r = codec.round_trip(t)
    assert r.group_enabled("Gr", "shift") and not r.masks
    assert set(t.params()) == set(r.params())
    assert "layers.W.occupancies" in t.params() and "groups.WS2.strain_field" in t.params()


def test_state_roundtrip():
    s = ModelState(position_shifts=[np.zeros((2, 2))], scales={"Gr": np.array([1.0, 1.1])},
                   optics=OpticsState(aberrations={"C10": 1.0}), normalisation=np.array([1.0, 0.0]))
    r = codec.round_trip(s)
    assert np.allclose(r.scales["Gr"], [1.0, 1.1]) and r.optics.aberrations["C10"] == 1.0


def test_invalid_names_rejected():
    with pytest.raises(ValueError):
        Experiment(Imaging((8, 8), 0.01), [Layer("a.b", "C")], [], Structure([np.zeros((0, 2))], [np.zeros(0)]))
    with pytest.raises(ValueError):
        Imaging((8, 8), 0.01, mode="nonsense")


def test_future_version_is_refused():
    raw = codec.encode(_small_experiment())
    raw["schema_version"] = 99
    with pytest.raises(ValueError, match="update"):
        store.experiment_from_dict(raw)


# -- run directory --------------------------------------------------------------

def test_run_roundtrip(tmp_path):
    e = _small_experiment()
    t = TrainSettings().ensure_for(e)
    t.masks["Gr"] = np.eye(8, dtype=bool)
    t.targets["Gr"] = np.arange(64, dtype=np.float32).reshape(8, 8)
    s = ModelState(position_shifts=[np.ones((5, 2)), np.zeros((3, 2)), np.zeros((0, 2))])
    store.save_run(str(tmp_path), store.Run(e, t, s, target=np.ones((8, 8), np.float32)))
    r = store.load_run(str(tmp_path))
    assert r.experiment.group_names == e.group_names
    assert r.train.masks["Gr"].dtype == bool and r.train.masks["Gr"].trace() == 8
    assert np.allclose(r.train.targets["Gr"], t.targets["Gr"])
    assert np.allclose(r.state.position_shifts[0], 1.0) and r.target.shape == (8, 8)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp") or ".tmp." in f]


# -- legacy formats (real files of this repository) -------------------------------

CONFIGS = [p for p in sorted(glob.glob(_p("**", "config.json"), recursive=True))
           if "/archive/" not in p and "/.git/" not in p]


@pytest.mark.skipif(not CONFIGS, reason="no legacy runs in this checkout")
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: os.path.relpath(p, ROOT))
def test_every_legacy_run_loads(path):
    run = store.load_run(os.path.dirname(path))
    e = run.experiment
    assert sum(e.structure.counts) > 0 or e.imaging.mode == "stem_haadf"
    assert set(run.train.groups) >= set(e.group_names)
    for g, m in run.train.masks.items():
        assert g in e.group_names and m.shape == tuple(e.imaging.shape)


@pytest.mark.skipif(not os.path.exists(_p("platelet_11", "best.json")), reason="no best.json")
def test_legacy_parameter_dump():
    e = store.load_run(_p("platelet_11", "config")).experiment
    s = store.load_state(_p("platelet_11", "best.json"), e)
    assert set(s.scales) == {"Gr", "WS2"} and s.scales["Gr"].shape == (2,)
    assert set(s.strain_control) == {"Gr", "WS2"} and s.strain_control["Gr"].ndim == 4
    assert [len(p) for p in s.position_shifts] == e.structure.counts
    assert s.optics is not None and "C10" in s.optics.aberrations
    assert set(s.contrast_factors) == {"Gr", "WS2"}


@pytest.mark.skipif(not os.path.exists(_p("platelet_11", "config", "area_masks.npy")), reason="no legacy masks")
def test_legacy_stacked_masks_and_group_target():
    r = store.load_run(_p("platelet_11", "config"))
    assert set(r.train.masks) == {"Gr", "WS2"} and "Gr" in r.train.targets
    assert r.train.masks["Gr"].dtype == bool


def test_legacy_four_layer_configuration():
    layers, groups = legacy.legacy_layers_and_groups([0.0, 0.3, 0.6, 0.9])
    assert [l.name for l in layers] == ["Gr", "S1", "W", "S2"]
    assert [g.name for g in groups] == ["Gr", "WS2"]
    assert {l.name: l.group for l in layers} == {"Gr": "Gr", "S1": "WS2", "W": "WS2", "S2": "WS2"}


def test_legacy_flat_train_flags():
    raw = {"graphen_shift": {"enabled": True, "lr": 0.5}, "WS2_free_positions": {"enabled": True, "lr": 0.2},
           "loss_ncc": 0.7, "only_train_small_area": 1, "train_normalisation_parameters": {"enabled": True, "lr": 1e-3}}
    t = legacy.train_settings_from_legacy(raw)
    assert t.group_enabled("Gr", "shift") and t.groups["Gr"].shift.lr == 0.5
    assert t.layer_enabled("S1", "free_positions") and not t.layer_enabled("Gr", "free_positions")
    assert t.loss.ncc == 0.7 and t.active_group == "WS2" and t.normalisation.enabled


def test_wrapped_envelopes_are_removed():
    raw = {"__type__": "dataclass", "data": {"a": {"__type__": "ndarray", "data": [1, 2]},
                                              "b": {"__type__": "tuple", "data": [1, 2]}}}
    assert legacy.unwrap(raw) == {"a": [1, 2], "b": [1, 2]}


# -- data folders ---------------------------------------------------------------------

STEM = _p("Ivan graphene", "DENSGraphen6_STEM")


@pytest.mark.skipif(not os.path.exists(os.path.join(STEM, "init.json")), reason="no ImageJ init.json")
def test_imagej_init_json():
    spec = dataset.spec_from_init(json.load(open(os.path.join(STEM, "init.json"))))
    g = spec.groups[0]
    assert g.name == "Gr" and np.isclose(g.max_rotation, np.deg2rad(20.0))
    assert spec.lattices[0].name == "Gr" and len(spec.lattices[0].basis) == 2
    assert spec.layer_sources[0].lattice == "Gr" and spec.layer_sources[0].lattice_layer == 0
    exp_imaging = spec.to_imaging((16, 16))
    assert exp_imaging.mode == "tem" and exp_imaging.aberrations["C10"] == -13.07
    t = TrainSettings()
    t.group("Gr")
    assert dataset.apply_init_train(t, spec.train) == []
    assert t.group_enabled("Gr", "shift") and t.normalisation.enabled
    assert t.groups["Gr"].shift.lr == 0.05 and t.loss.ncc == 0.5 and t.active_group == "Gr"


@pytest.mark.skipif(not os.path.exists(os.path.join(STEM, "target.tif")), reason="no data folder")
def test_load_data_folder():
    pytest.importorskip("tifffile")
    d = dataset.load_dataset(STEM)
    assert d.target.shape == d.spec.shape and d.target.dtype == np.float32
    assert "gr" in d.masks or "Gr" in d.masks
    assert d.target.min() > 0


def test_zero_pixels_take_the_median():
    img = np.array([[0, 2, 4], [6, 8, 0]], dtype=np.float32)
    out = dataset.prepare_target(img)
    assert (out > 0).all() and out[0, 0] == np.median(img)
    assert (dataset.prepare_target(img, "keep") == img).all()


def test_mask_names():
    assert dataset.group_name_from_mask("x/mask_WS2.tif") == "WS2"
    assert dataset.group_name_from_mask("gr_mask.tif") == "gr"
    assert dataset.group_name_from_mask("mask.npy") == "Gr"


# -- dependency rule ----------------------------------------------------------------------

def test_data_package_does_not_import_torch():
    code = ("import sys, temsim.data.store, temsim.data.dataset, temsim.data.legacy;"
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
