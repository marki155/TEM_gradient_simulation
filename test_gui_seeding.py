"""GUI beim Seeding: Sonde statt CTF (STEM), Intensitaets-Histogramm, keine Strain-Kacheln, Parameter."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtCore, QtWidgets

import test_atom_seeding as T
from atom_seeding import RandomAtomSeeder
from gui_qt import LiveTrainWindow

FAILS = []


def check(name, ok, detail=""):
    print(f"  {name:55s} {'ok' if ok else 'FEHLER'} {detail}")
    if not ok:
        FAILS.append(name)


def pump(app, seconds=0.0, rounds=3):
    import time
    end = time.monotonic() + seconds
    for _ in range(rounds):
        app.processEvents()
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.02)


def main():
    true = T.graphene_lattice(T.SHAPE, T.PIXEL_NM)
    target = T.render_target("stem", true)
    model, _t, train_method = T.make("stem", border_px=4)
    seeder = RandomAtomSeeder(model, target, loss="ncc", seed=1, margin_px=4)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = LiveTrainWindow(model=model, target=target, pot_layer=model.potential_layer,
                             ctf_layer=model.imaging_layer, train_method=train_method,
                             plot_every=25, heavy_every=50, seeder=seeder,
                             seeding=dict(trials=300, tune_every=0))
    window.resize(1400, 900)
    window.show()
    canvas, combo = window.canvas, window.single_tile_combo
    pump(app)

    tiles = set(canvas.axes_cmap)
    check("Seeding: keine Strain-Kacheln", not tiles & set(canvas.STRAIN_TILES), str(sorted(tiles)))
    check("Kachelauswahl ohne Strain",
          not {combo.itemData(i) for i in range(combo.count())} & set(canvas.STRAIN_TILES))
    check("STEM: die Kachel heisst Probe", canvas.tile_titles()["ctf"] == "Probe")

    # -- Parameter ------------------------------------------------------------
    spins = window.seeding_spins
    spins["merge_A"].setValue(1.7)
    spins["touch_A"].setValue(0.4)
    spins["move_prob"].setValue(0.25)
    spins["max_atoms"].setValue(1000)
    check("Radien wirken auf den Seeder",
          abs(seeder.merge_A - 1.7) < 1e-9 and abs(seeder.touch_A - 0.4) < 1e-9
          and abs(seeder.merge_px - 1.7 / (T.PIXEL_NM * 10)) < 1e-9,
          f"merge {seeder.merge_A:.3f} A = {seeder.merge_px:.2f} px")
    check("Wahrscheinlichkeit und Atomgrenze", seeder.move_prob == 0.25 and seeder.max_atoms == 1000)

    window.seeding_neighbor_check.setChecked(True)
    spins["bond_A"].setValue(1.5)
    spins["neighbor_max_A"].setValue(2.2)
    check("Nachbar-Seeding einstellbar",
          seeder.neighbor_seeding and abs(seeder.bond_A - 1.5) < 1e-9
          and abs(seeder.neighbor_max_px - 2.2 / (T.PIXEL_NM * 10)) < 1e-9)
    window.seeding_neighbor_check.setChecked(False)
    check("Nachbar-Seeding abschaltbar", not seeder.neighbor_seeding)
    window.seeding_gradient_check.setChecked(True)
    spins["gradient_temperature"].setValue(0.3)
    check("Gradienten-Seeding einstellbar", seeder.gradient_seeding and abs(seeder.gradient_temperature - 0.3) < 1e-9)

    # -- laufendes Seeding ------------------------------------------------------
    window.start_seeding()
    pump(app, 4.0)
    spins["merge_A"].setValue(0.9)
    check("Parameter lassen sich im Lauf aendern", abs(seeder.merge_A - 0.9) < 1e-9)
    if window.thread is not None:
        window.stop_btn.click()
    pump(app, 3.0)

    probe = canvas.im_ctf.get_array()
    check("Probe-Kachel gefuellt, positiv, ein Maximum in der Mitte",
          probe is not None and probe.min() >= 0 and probe.max() > 0
          and np.unravel_index(np.argmax(probe), probe.shape) == (probe.shape[0] // 2, probe.shape[1] // 2),
          f"{probe.shape}, max bei {np.unravel_index(np.argmax(probe), probe.shape)}")
    check("keine CTF-Ringmaske im STEM", canvas.im_ring is None)

    # -- Intensitaets-Histogramm: mehrere Werte -> mehrere Klassen ---------------
    from gui_qt import collect_layer_visuals
    potential = model.potential_layer
    import torch
    with torch.no_grad():
        values = potential.tunable_parameters.intensities[0]
        values += torch.linspace(0, 3, values.numel(), device=values.device)
    _p, _i, counts, edges = collect_layer_visuals(potential)
    nonzero = int((counts[0] > 0).sum())
    check("Histogramm zeigt die Verteilung, nicht einen Peak", nonzero > 10, f"{nonzero} Klassen belegt")
    check("Histogramm zaehlt nur besetzte Atome",
          abs(counts[0].sum() - np.isfinite(_i[0]).sum()) < 1e-6,
          f"{counts[0].sum():.0f} von {len(_i[0])}")

    # -- Training zurueck: Strain-Kacheln wieder da -----------------------------
    window.start_training()
    pump(app, 1.0)
    window.stop_btn.click()
    pump(app, 2.0)
    check("Training: Strain-Kacheln wieder da", set(canvas.STRAIN_TILES) <= set(canvas.axes_cmap),
          str(sorted(canvas.axes_cmap)))

    print("\n" + ("alle Tests durch" if not FAILS else "FEHLER: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
