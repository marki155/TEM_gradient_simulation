"""Tests fuer den Testset-Builder (headless): Modell, Speichern, Einlesen durch Dataset, Seeding, GUI-Klicks."""

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import testset_builder_gui as tb


def small_model(mode="stem"):
    model = tb.TestsetModel({"height": 64, "width": 64, "pixel_A": 0.25, "padding": 20, "mode": mode,
                             "noise_sigma": 0.02, "dose": 0.0})
    model.fill_graphene(10.0, 6)
    return model


def test_model_render_and_save():
    from mains import main_random_atoms as mra
    for mode in ("stem", "tem"):
        model = small_model(mode)
        assert len(model.positions) > 20
        image = model.render(noise=True)
        assert image.shape == (64, 64) and np.isfinite(image).all() and image.std() > 0.05
        model.params["mask_mode"] = "margin"
        model.params["margin_px"] = 8
        with tempfile.TemporaryDirectory() as folder:
            model.save(folder, image=image)
            for name in ("target.tif", "mask_Gr.tif", "init_random.json", tb.TRUTH_NPY, tb.TRUTH_XYZ, tb.BUILDER_FILE):
                assert os.path.exists(os.path.join(folder, name)), name
            data = mra.Dataset(folder)
            assert data.image.shape == (64, 64) and data.mode == mode
            assert abs(data.pixel_size_nm - 0.025) < 1e-9 and data.group_name == "Gr"
            assert data.mask.sum() == (64 - 16) ** 2
            again = tb.TestsetModel.load(folder)
            assert np.allclose(again.positions, model.positions) and again.params["mode"] == mode
            assert np.allclose(again.render(noise=True), image, atol=1e-5), "Laden reproduziert das Bild"
        print(f"ok  Modell {mode}: {len(model.positions)} Atome, Datenordner wird von Dataset gelesen")


def test_seeding_on_saved_folder():
    from mains import main_random_atoms as mra
    model = small_model("stem")
    model.params["noise_sigma"] = 0.0
    with tempfile.TemporaryDirectory() as folder:
        model.save(folder)
        assert mra.main([folder, "--trials", "800", "--border-px", "4", "--tune-every", "0",
                         "--log-every", "0", "--seed", "1"]) == 0
        assert os.path.exists(os.path.join(folder, "config_random", "config.json"))
    print("ok  Seeding laeuft auf dem gespeicherten Testset")


def test_tools():
    pts = tb.graphene_points((64, 64), 0.25, 0.0, 4)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(pts, k=2)
    assert abs(np.median(d[:, 1]) - 1.42 / 0.25) < 0.05, np.median(d[:, 1])
    ring = tb.hexagon((10, 10), 0, 0.25)
    assert np.allclose(np.hypot(*(ring - 10).T), 1.42 / 0.25)
    model = tb.TestsetModel({"height": 64, "width": 64})
    model.add([[10, 10], [20, 20]])
    assert model.nearest((11, 11), 3) == 0 and model.nearest((40, 40), 3) is None
    print("ok  Werkzeuge: Gitter, Ring, naechstes Atom")


def test_gui_clicks():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = tb.build_window_class()()
    window.auto_render.setChecked(False)
    window._widgets["height"].setValue(48)
    window._widgets["width"].setValue(48)
    window._widgets["padding"].setValue(16)
    assert window.model.shape == (48, 48)

    window.click(10, 10)
    window.click(30, 30)
    window.click(30.5, 30.5)                       # trifft vorhandenes Atom: Ziehen statt Setzen
    assert len(window.model.positions) == 2 and window._grab == 1
    window._grab = None
    window.click(10, 10, button=3)
    assert len(window.model.positions) == 1
    window.undo()
    assert len(window.model.positions) == 2
    window.tool.setCurrentIndex(1)
    window.click(24, 24)
    assert len(window.model.positions) == 8
    window.fill_margin.setValue(0)
    window.fill_graphene()
    assert len(window.model.positions) > 20
    window.jitter_atoms()
    window.vacancies()

    window.mode.setCurrentText("tem")
    assert window.model.params["mode"] == "tem"
    window.mode.setCurrentText("stem")
    window.render()
    assert window._clean is not None and not window._stale and window._clean.shape == (48, 48)
    window.show_mask.setChecked(True)
    window._widgets["mask_radius_A"].setValue(0.8)
    window.mask_mode.setCurrentText("atoms")
    assert not window.model.mask().all()
    window.probe_table.item(0, 1).setText("2.5")
    assert window.model.params["probe_aberrations"]["C10"] == 2.5

    with tempfile.TemporaryDirectory() as folder:
        window.folder_edit.setText(folder)
        window.save_folder()
        from mains import main_random_atoms as mra
        data = mra.Dataset(folder)
        assert data.image.shape == (48, 48) and data.mask is not None
        assert data.optics["probe_aberrations"]["C10"] == 2.5
        n = len(window.model.positions)
        window.clear_atoms()
        window.load_folder()
        assert len(window.model.positions) == n
        command = window.seeding_command(folder)
        assert command[1].endswith("main_random_atoms.py") and "--gui" in command
    print("ok  GUI: Klicks, Ring, Gitter, Undo, Parameter, Speichern/Laden")


def main():
    test_tools()
    test_model_render_and_save()
    test_seeding_on_saved_folder()
    test_gui_clicks()
    print("\nalle Builder-Tests bestanden")


if __name__ == "__main__":
    main()
