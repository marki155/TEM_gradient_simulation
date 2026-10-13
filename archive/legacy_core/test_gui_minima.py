"""Minima im Diff-Bild zeigen und als neue Atome einfuegen."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import torch
from PySide6 import QtWidgets
from scipy.spatial import cKDTree

import test_atom_seeding as T
from atom_seeding import AtomStore
from gui_qt import LiveTrainWindow

FAILS = []


def check(name, ok, detail=""):
    print(f"  {name:60s} {'ok' if ok else 'FEHLER'} {detail}")
    if not ok:
        FAILS.append(name)


def main():
    true = T.graphene_lattice(T.SHAPE, T.PIXEL_NM)
    target = T.render_target("stem", true)
    model, _t, train_method = T.make("stem", border_px=4)

    rng = np.random.default_rng(3)
    inner = np.flatnonzero((true > 15).all(axis=1) & (true < T.SHAPE[0] - 15).all(axis=1))
    missing = rng.choice(inner, 5, replace=False)
    present = np.setdiff1d(np.arange(len(true)), missing)
    AtomStore(model.potential_layer).apply([], true[present])

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    saved = []
    window = LiveTrainWindow(model=model, target=target, pot_layer=model.potential_layer,
                             ctf_layer=model.imaging_layer, train_method=train_method,
                             plot_every=25, heavy_every=50, save_callback=saved.append)
    window.resize(1500, 900)
    window.show()
    canvas = window.canvas
    window._trigger_heavy_refresh()
    for _ in range(5):
        app.processEvents()

    # Anzeige ab Occ: alle Atome haben sigmoid(6) = 0.9975 -> ab 0.999 ist keins mehr sichtbar
    shown = lambda: sum(int(np.isfinite(i).sum()) for i in window._build_local_heavy_payload(
        0, 0.0, torch.zeros(T.SHAPE), torch.zeros(T.SHAPE), None).pot_intensities)
    n_all = shown()
    window.show_occ_spin.setValue(1.0)
    n_none = shown()
    window.show_occ_spin.setValue(0.5)
    check("Anzeige ab Occ filtert", n_all > 0 and n_none == 0 and shown() == n_all, f"{n_all} -> {n_none}")

    window.show_minima_checkbox.setChecked(True)
    window.minima_threshold_spin.setValue(0.3)
    app.processEvents()
    points = canvas.minima_positions()
    raw = canvas.minima_positions(filtered=False)
    check("Minima gefunden", len(points) >= 5, f"{len(points)} Minima")
    distance, _ = cKDTree(points).query(true[missing])
    check("Minima liegen auf den fehlenden Atomen", (distance < 2.0).all(), f"max {distance.max():.2f} px")
    check("Minima gezeichnet", len(canvas._minima_artists) == 1)

    # Seeding-Filter: Punkt im Hexagonmittelpunkt (nirgends ein Atom) und am Rand fliegen raus
    hexagon = true[missing[0]]
    neighbours = cKDTree(true).query(hexagon, k=4)[1][1:]
    centre = np.mean([true[i] for i in neighbours[:3]], axis=0) + (true[neighbours[0]] - hexagon)
    candidates = np.array([true[missing[0]], centre, [1.0, 1.0]])
    kept = window._filter_minima(candidates)
    check("Filter: echtes Atom bleibt", len(kept) >= 1 and np.hypot(*(kept[0] - candidates[0])) < 1.0)
    check("Filter: Rand und Loss-Test werfen Punkte raus", len(kept) < 3, f"{len(kept)} von 3")
    for box in window.minima_checks.values():
        box.setChecked(False)
    unfiltered = window.canvas.minima_positions()
    check("alle Kriterien aus: Punkte unveraendert", len(window._filter_minima(candidates)) == 3)
    window.minima_checks["loss"].setChecked(False)
    window.minima_checks["margin"].setChecked(True)
    window.minima_spins["margin_px"].setValue(5.0)
    check("nur Rand: Randpunkt-Test", len(window._filter_minima(np.array([[0.2, 0.2], [50., 50.]]))) == 1)
    window.minima_spins["margin_px"].setValue(0.0)
    window.minima_checks["collision"].setChecked(True)
    window.minima_checks["loss"].setChecked(True)
    check("Filter an: nie mehr Minima als ohne", len(canvas.minima_positions()) <= len(unfiltered))

    window.minima_threshold_spin.setValue(1.0)
    check("Schwelle 1: nur das tiefste Minimum", len(canvas.minima_positions()) == 1)
    window.minima_threshold_spin.setValue(0.3)

    n_before = model.potential_layer.tunable_parameters.occupancies[0].numel()
    window.insert_minima_btn.click()
    for _ in range(5):
        app.processEvents()
    n_after = model.potential_layer.tunable_parameters.occupancies[0].numel()
    check("Atome eingefuegt", n_after > n_before and n_after <= n_before + len(raw), f"{n_before} -> {n_after}")
    positions = (model.potential_layer.init_positions_0
                 + model.potential_layer.tunable_parameters.position_shifts[0]).detach().cpu().numpy()
    distance, _ = cKDTree(positions).query(true[missing])
    check("Speichern nach dem Einfuegen ausgeloest", len(saved) == 1)
    check("fehlende Atome sind jetzt im Modell", (distance < 2.0).all(), f"max {distance.max():.2f} px")
    check("danach keine Minima mehr an den alten Stellen",
          len(canvas.minima_positions()) < len(points), f"{len(canvas.minima_positions())} uebrig")

    before = len(saved)
    window.close()
    check("Speichern beim Schliessen", len(saved) == before + 1)

    print("\n" + ("alle Minima-Tests bestanden" if not FAILS else "FEHLER: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
