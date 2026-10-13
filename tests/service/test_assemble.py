"""Part IV, Section 3: the three routes to a model."""

import numpy as np
import pytest

from temsim.data import dataset, store
from temsim.service import assemble

from ..learning.helpers import SHAPE, PX
from .helpers import make_folder, make_run


def test_lattice_route(tmp_path):
    data = dataset.load_dataset(make_folder(tmp_path / "f"))
    a = assemble.lattice_assembly(data)
    exp = a.experiment
    assert a.kind == "lattice" and a.out_dir.endswith("config_test")
    assert exp.imaging.shape == SHAPE and exp.imaging.mode == "tem"
    assert [l.name for l in exp.layers] == ["Gr"] and len(exp.structure.positions[0]) > 50
    assert np.all(exp.structure.atomic_numbers[0] == 6.0)
    # the lattice reaches beyond the image by the allowed shift (Part III, Eq. 3.1)
    pos = exp.structure.positions[0]
    assert pos.min() < 0 and pos[:, 0].max() > SHAPE[0] - 1
    assert a.settings.active_group == "Gr" and a.settings.group_enabled("Gr", "shift")
    assert a.settings.groups["Gr"].shift.lr == 0.05 and a.settings.loss.ncc == 0.5
    assert a.settings.masks["Gr"].shape == SHAPE and a.settings.masks["Gr"].dtype == bool
    assert a.state is None and not a.notes


def test_lattice_route_reports_unknown_keys_and_missing_pieces(tmp_path):
    init_extra = {"train": {"enable": ["groups.Gr.shift", "no.such.key"], "active_group": "Gr"}}
    data = dataset.load_dataset(make_folder(tmp_path / "f", **init_extra))
    assert assemble.lattice_assembly(data).notes == ["unknown training key in init file: no.such.key"]
    with pytest.raises(ValueError, match="seeding route"):
        assemble.lattice_assembly(dataset.load_dataset(make_folder(tmp_path / "g", with_lattice=False)))
    bad = {"layers": [{"name": "Gr", "element": "C", "group": "Gr", "lattice": "nope"}]}
    with pytest.raises(ValueError, match="unknown lattice"):
        assemble.lattice_assembly(dataset.load_dataset(make_folder(tmp_path / "h", **bad)))


def test_lattice_heights_are_taken_from_the_lattice_if_unset(tmp_path):
    extra = {"lattices": [{"name": "L", "type": "custom", "a": 2.46, "gamma_deg": 120.0,
                           "basis": [["W", 0, 0, 0.0], ["S", 1 / 3, 2 / 3, 0.15], ["S", 1 / 3, 2 / 3, -0.15]]}],
             "layers": [{"name": "top", "element": "S", "group": "Gr", "lattice": "L", "lattice_layer": 0},
                        {"name": "mid", "element": "W", "group": "Gr", "lattice": "L", "lattice_layer": 1}]}
    a = assemble.lattice_assembly(dataset.load_dataset(make_folder(tmp_path / "f", **extra)))
    assert a.experiment.layers[0].z_nm != a.experiment.layers[1].z_nm


def test_seeding_route(tmp_path):
    data = dataset.load_dataset(make_folder(tmp_path / "f", mode="stem_haadf", with_lattice=False))
    a = assemble.seeding_assembly(data, border_px=5)
    exp = a.experiment
    assert a.kind == "seed" and a.out_dir.endswith(assemble.SEED_OUT_DIR)
    assert exp.imaging.mode == "stem_haadf" and exp.imaging.pixel_size_nm == pytest.approx(PX)
    assert len(exp.layers) == 1 and len(exp.structure.positions[0]) == 0
    mask = a.settings.masks[exp.groups[0].name]
    assert not mask[:5].any() and not mask[:, -5:].any() and mask[8, 8] and not mask[7, 7]
    assert a.settings.aberrations.enabled and a.settings.normalisation.enabled
    assert a.region is not None and a.border_px == 5


def test_seeding_route_needs_a_pixel_size(tmp_path):
    folder = make_folder(tmp_path / "f", mode="stem_haadf", with_lattice=False)
    import os
    os.remove(os.path.join(folder, "init.json"))
    data = dataset.load_dataset(folder)
    with pytest.raises(ValueError, match="pixel size"):
        assemble.seeding_assembly(data)
    a = assemble.seeding_assembly(data, fov_nm=PX * SHAPE[1], mode="stem_haadf")
    assert a.experiment.imaging.pixel_size_nm == pytest.approx(PX)


def test_training_start_changes_a_copy(tmp_path):
    data = dataset.load_dataset(make_folder(tmp_path / "f", mode="stem_haadf", with_lattice=False))
    a = assemble.seeding_assembly(data)
    name = a.experiment.layers[0].name
    t = assemble.training_start(a.settings, name, a.region)
    assert t.layer(name).free_positions.enabled and t.layer(name).intensities.enabled
    assert t.regularisation.repulsion_a == 10.0 and t.regularisation.occ_entropy_a == 0.0
    assert t.aberration_flags["C10"] and not t.aberration_flags.get("C12", False)
    assert not a.settings.layer(name).free_positions.enabled and a.settings.regularisation.repulsion_a == 0.0


def test_run_route(tmp_path):
    path, exp, target = make_run(tmp_path / "run")
    a = assemble.run_assembly(path)
    assert a.kind == "run" and a.out_dir == path
    np.testing.assert_allclose(a.target, target)
    assert a.settings.group_enabled("g1", "shift") and "g1" in a.settings.masks
    # without a stored target the one of the parent folder is used
    import os
    os.remove(os.path.join(path, "target.npy"))
    with pytest.raises(FileNotFoundError):
        assemble.run_assembly(path)
    np.save(tmp_path / "target.npy", target)
    np.testing.assert_allclose(assemble.run_assembly(path).target, target)
    np.save(tmp_path / "target.npy", target[:10])
    with pytest.raises(ValueError, match="does not fit"):
        assemble.run_assembly(path)
