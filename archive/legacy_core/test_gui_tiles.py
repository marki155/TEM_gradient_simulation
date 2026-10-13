"""Einzelansicht der GUI: jede Kachel gross, Zoom bleibt, Lernraten sichtbar.

Laeuft ohne Bildschirm (QT_QPA_PLATFORM=offscreen) und ohne Training - es wird
nur aufgebaut, einmal gerendert und geprueft.

    python3 test_gui_tiles.py
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import torch
from PySide6 import QtWidgets

from layers import graphene_only, legacy_layers_and_groups
from nn import NN_particles, ParticlesConfig
from training_parameter import TrainMethod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)

FAILS = []


def check(name, condition, detail=""):
    print(f"  {name:<46} {'ok' if condition else 'FEHLER'} {detail}")
    if not condition:
        FAILS.append(name)


def build_window(layers_and_groups=None):
    rng = np.random.default_rng(0)
    layers, groups = layers_and_groups or graphene_only()
    shape, n_atoms = (128, 128), 200
    config = ParticlesConfig(
        init_positions=[rng.uniform(10, shape[0] - 10, size=(n_atoms, 2)) for _ in layers],
        init_intensities=[np.full(n_atoms, 6.0) for _ in layers],
        shape=shape, pixel_size_nm=0.012, voltage_kV=80, padding=8,
        init_aberrations={"C10": -13.0, "C12": 1.0, "phi12": 0.0, "C30": 4.0},
        init_ctf_env_params={"source_size": 1e-4, "Cc_um": 1e3, "deltaE": 0.35,
                             "instab_defocus_nm": 0.01, "use_detector": False},
        init_amp_contrast=0.1, ctf_alpha_max_mrad=50.0,
        layers=layers, groups=groups,
    )
    train_method = TrainMethod().ensure_for_config(config)
    train_method.train_small_area_squares = [(20, 20, 100, 100)]

    model = NN_particles(config=config, pixel_size_nm=config.pixel_size_nm,
                         voltage_kV=config.voltage_kV, train_method=train_method,
                         device=DEVICE, train_methods=train_method)
    model.build_multislice()

    with torch.no_grad():
        target = model.model(None)[0].detach()

    from gui_qt import LiveTrainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = LiveTrainWindow(model=model, target=target, pot_layer=model.model[0],
                             ctf_layer=model.imaging_layer, train_method=train_method,
                             plot_every=5, heavy_every=10, max_steps=10)
    window.resize(1400, 900)
    window.show()
    for _ in range(3):
        app.processEvents()
    return app, window, config, model


def select(app, window, label):
    combo = window.single_tile_combo
    labels = [combo.itemText(i) for i in range(combo.count())]
    combo.setCurrentIndex(labels.index(label))
    for _ in range(3):
        app.processEvents()


def main():
    app, window, config, model = build_window()
    canvas, combo = window.canvas, window.single_tile_combo

    # -- Lernraten: je Gruppen- und Schichtparameter ein eigenes Feld --------
    rows = {key: info for key, info in window.editor.widgets.items()
            if info.get("kind") == "trainparam"}
    check("eigene Lernrate je Trainingsparameter", len(rows) >= 8, f"{len(rows)} Zeilen")
    check("Lernraten sichtbar", all(info["lr"].isVisible() for info in rows.values()))
    key = "groups.Gr.shift"
    if key in rows:
        rows[key]["lr"].setValue(0.25)
        check("Lernrate wirkt auf die TrainMethod",
              abs(window.train_method.get_train_param(key).lr - 0.25) < 1e-9,
              str(window.train_method.get_train_param(key).lr))

    # -- jede Kachel einzeln -------------------------------------------------
    every_tile_alone = True
    for index in range(combo.count()):
        combo.setCurrentIndex(index)
        for _ in range(3):
            app.processEvents()
        tile = combo.itemData(index)
        expected = [tile] if tile in (None, "loss") else sorted([tile, "loss"])
        if tile == "compare":
            expected = sorted(["sim", "target", "diff", "loss"])
        if tile is not None and sorted(canvas.axes_cmap) != expected:
            every_tile_alone = False
            print(f"    {tile} zeigt {sorted(canvas.axes_cmap)}")
    check("jede Kachel laesst sich einzeln zeigen (mit Loss daneben)", every_tile_alone,
          f"{combo.count() - 1} Kacheln")

    # -- Zoom ----------------------------------------------------------------
    select(app, window, "Simulated")
    axis = canvas.axes_cmap["sim"]
    full = axis.get_xlim()
    check("Einzelansicht startet auf dem ganzen Bild",
          abs(full[1] - config.shape[1] + 0.5) < 1.0, str(full))
    axis.set_xlim(30, 70)
    axis.set_ylim(30, 70)
    window._trigger_heavy_refresh()
    for _ in range(3):
        app.processEvents()
    check("Zoom ueberlebt eine Aktualisierung", abs(axis.get_xlim()[0] - 30) < 1e-6,
          str(axis.get_xlim()))
    canvas.reset_zoom()
    check("Zoom zuruecksetzen wirkt auf die Einzelkachel",
          abs(axis.get_xlim()[1] - full[1]) < 1e-6, str(axis.get_xlim()))

    # -- Loss-Verlauf und Target-Zubehoer ------------------------------------
    canvas.update_light(3, 1.0, np.array([1.4, 1.2, 1.05, 1.0], dtype=np.float32))
    select(app, window, "Loss")
    check("Loss-Kurve ueberlebt den Kachelwechsel",
          canvas.loss_line is not None and len(canvas.loss_line.get_xdata()) == 4,
          f"{len(canvas.loss_line.get_xdata()) if canvas.loss_line else 0} Punkte")

    select(app, window, "Simulated")
    check("Loss-Verlauf neben dem Einzelbild",
          "loss" in canvas.axes_cmap and canvas.loss_line is not None
          and len(canvas.loss_line.get_xdata()) == 4,
          f"Kacheln {sorted(canvas.axes_cmap)}")

    select(app, window, "Target")
    check("Atompositionen auch in der Einzelansicht",
          canvas.im_tgt_sc is not None and len(canvas.im_tgt_sc.get_offsets()) > 50,
          f"{len(canvas.im_tgt_sc.get_offsets())} Punkte")
    check("Trainingsrechtecke wieder eingezeichnet",
          len(canvas.axes_cmap["target"].patches) >= 1,
          f"{len(canvas.axes_cmap['target'].patches)} Rechtecke")

    # -- zurueck -------------------------------------------------------------
    select(app, window, "alle Kacheln")
    # Diff ist hier konstant null (das Ziel kommt aus demselben Modell), also
    # nur pruefen, dass Daten anliegen - nicht, dass sie variieren
    with_data = [k for k, art in canvas.image_artists().items()
                 if art.get_array() is not None and np.ndim(art.get_array()) >= 2]
    varying = [k for k in with_data
               if np.nanmax(canvas.image_artists()[k].get_array())
               != np.nanmin(canvas.image_artists()[k].get_array())]
    check("zurueck zu allen Kacheln, wieder gefuellt",
          len(canvas.axes_cmap) > 5 and {"sim", "diff", "target"} <= set(with_data)
          and {"sim", "target"} <= set(varying),
          f"{len(canvas.axes_cmap)} Kacheln, Daten in: {','.join(sorted(with_data))}")

    check_reset_intensities(app, window, model)
    check_reset_only_active_group()

    print("\n" + ("alle Tests durch" if not FAILS else "FEHLER: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


def check_reset_intensities(app, window, model):
    """Der Knopf muss auf die Startwerte zurueck, nicht auf 0."""

    potential = model.model[0]
    params = potential.tunable_parameters
    start = potential.get_init_intensities(0).clone()

    with torch.no_grad():
        params.intensities[0].fill_(7.5)
    weights_before = potential.get_atom_weights()[0].detach().clone()

    window.reset_intensities_btn.click()
    for _ in range(3):
        app.processEvents()

    now = params.intensities[0].detach()
    check("Reset intensities holt die Startwerte zurueck",
          torch.allclose(now, start, atol=1e-5),
          f"{float(now[0]):.3f} statt {float(start[0]):.3f}")
    check("nicht auf 0 gesetzt", float(now.abs().min()) > 0.0, f"min {float(now.min()):.3f}")

    weights_after = potential.get_atom_weights()[0].detach()
    occupancy = torch.sigmoid(params.occupancies[0].detach())
    check("danach gilt wieder Gewicht = Occupancy",
          torch.allclose(weights_after, occupancy, atol=1e-5),
          f"max Abweichung {float((weights_after - occupancy).abs().max()):.2e}")
    check("der Knopf hat ueberhaupt etwas geaendert",
          not torch.allclose(weights_before, weights_after, atol=1e-5))


def check_reset_only_active_group():
    """Wie die anderen Reset-Knoepfe: nur die Schichten der aktiven Gruppe."""

    app, window, config, model = build_window(legacy_layers_and_groups([0.0, 0.33, 0.4875, 0.645]))
    potential = model.model[0]
    params = potential.tunable_parameters

    window.train_method.set_active_group("Gr")
    active = set(window._active_layer_indices())
    with torch.no_grad():
        for i in range(len(config.layers)):
            params.intensities[i].fill_(99.0)

    window.reset_intensities_btn.click()
    for _ in range(3):
        app.processEvents()

    restored, untouched = [], []
    for i, layer in enumerate(config.layers):
        back = torch.allclose(params.intensities[i].detach(),
                              potential.get_init_intensities(i), atol=1e-5)
        (restored if back else untouched).append(layer.name)

    check("nur die aktive Gruppe zurueckgesetzt",
          set(i for i, l in enumerate(config.layers) if l.name in restored) == active,
          f"zurueck: {restored}, unveraendert: {untouched}")


if __name__ == "__main__":
    sys.exit(main())
