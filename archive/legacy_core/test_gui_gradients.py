"""Gradient-Vektoren je Atom und Zoom-Vergleich Sim | Target | Diff + Loss."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import torch
from PySide6 import QtWidgets

import test_atom_seeding as T
from atom_seeding import AtomStore
from gui_qt import LiveTrainWindow, position_gradients

FAILS = []


def check(name, ok, detail=""):
    print(f"  {name:60s} {'ok' if ok else 'FEHLER'} {detail}")
    if not ok:
        FAILS.append(name)


def main():
    true = T.graphene_lattice(T.SHAPE, T.PIXEL_NM)
    target = T.render_target("stem", true)
    model, _t, train_method = T.make("stem", border_px=4)

    shifted = true.copy()
    center = np.array(T.SHAPE) / 2
    k = int(np.argmin(np.hypot(*(true - center).T)))
    offset = np.array([1.6, -1.2])
    shifted[k] += offset
    AtomStore(model.potential_layer).apply([], shifted)

    layer_name = model.potential_layer.layer_names[0]
    flag_before = train_method.layer(layer_name).free_positions.enabled
    grads = position_gradients(model, model.potential_layer, train_method, target)
    check("Flags nach der Gradientenberechnung wiederhergestellt",
          train_method.layer(layer_name).free_positions.enabled == flag_before)
    g = grads[0]
    positions = model.potential_layer.get_positions()
    check("ein Gradient je Atom", g.shape[0] == len(shifted) and g.shape[1] == 2, str(g.shape))

    norms = np.hypot(g[:, 0], g[:, 1])
    check("das verschobene Atom hat den groessten Gradienten", int(np.argmax(norms)) == k,
          f"argmax {int(np.argmax(norms))} vs {k}")
    descent = -g[k]
    cos = float(descent @ (-offset) / (np.linalg.norm(descent) * np.linalg.norm(offset)))
    check("Abstiegsrichtung zeigt zur wahren Position zurueck", cos > 0.9, f"cos = {cos:.3f}")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = LiveTrainWindow(model=model, target=target, pot_layer=model.potential_layer,
                             ctf_layer=model.imaging_layer, train_method=train_method,
                             plot_every=25, heavy_every=50)
    window.resize(1500, 900)
    window.show()
    canvas = window.canvas
    app.processEvents()

    window.show_gradients_checkbox.setChecked(True)
    for _ in range(5):
        app.processEvents()
    check("Pfeile auf dem Target", len(canvas._quivers) == 1 and canvas._arrow_data is not None)

    row, col = shifted[k]
    window.single_tile_combo.setCurrentIndex(window.single_tile_combo.findData("compare"))
    for _ in range(5):
        app.processEvents()
    check("Zoom-Vergleich: Sim, Target, Diff und Loss", set(canvas.axes_cmap) == {"sim", "target", "diff", "loss"})
    check("Pfeile auf Sim, Target und Diff", len(canvas._quivers) == 3, str(len(canvas._quivers)))

    ax = canvas.axes_cmap["sim"]
    ax.set_xlim(col - 8, col + 8)
    ax.set_ylim(row - 8, row + 8)
    canvas._apply_view_dependent()
    app.processEvents()
    check("gemeinsamer Zoom", all(canvas.axes_cmap[key].get_xlim() == ax.get_xlim() for key in ("target", "diff")))
    n_visible = len(canvas._quivers[0].U)
    check("nur Pfeile im Ausschnitt", 0 < n_visible < len(shifted), f"{n_visible} von {len(shifted)}")
    longest = float(np.hypot(canvas._quivers[0].U, canvas._quivers[0].V).max())
    check("laengster Pfeil hat die eingestellte Laenge", abs(longest - 3.0) < 1e-3, f"{longest:.3f}")
    low, high = canvas.im_sim.get_clim()
    check("Farbskala folgt dem Ausschnitt", canvas.im_sim.get_clim() == canvas.im_tgt.get_clim()
          and canvas.im_diff.get_clim()[0] == -canvas.im_diff.get_clim()[1])

    quiver_len = np.hypot(canvas._quivers[0].U, canvas._quivers[0].V)
    check("Pfeillaenge waechst mit dem Betrag des Gradienten",
          quiver_len.min() < 0.9 * quiver_len.max(), f"{quiver_len.min():.3f} .. {quiver_len.max():.3f}")
    ratio = quiver_len.max() / 3.0
    check("Laenge = Laenge * |g| / max |g|", abs(ratio - 1.0) < 1e-3)

    window.gradient_descent_checkbox.setChecked(False)
    app.processEvents()
    check("Vorzeichen umschaltbar", np.sign(canvas._quivers[0].U[np.argmax(np.hypot(canvas._quivers[0].U, canvas._quivers[0].V))])
          != 0)

    window.gradient_length_spin.setValue(6.0)
    app.processEvents()
    longest = float(np.hypot(canvas._quivers[0].U, canvas._quivers[0].V).max())
    check("Pfeillaenge einstellbar", abs(longest - 6.0) < 1e-3, f"{longest:.3f}")

    window.show_gradients_checkbox.setChecked(False)
    app.processEvents()
    check("Pfeile abschaltbar", not canvas._quivers)

    # -- Atome unter einer Occupancy entfernen ----------------------------------
    n_before = model.potential_layer.tunable_parameters.occupancies[0].numel()
    with torch.no_grad():
        occ = model.potential_layer.tunable_parameters.occupancies[0]
        occ[:10] = -3.0
        occ[10:15] = 0.2
    window.remove_low_occ_spin.setValue(0.5)
    window.remove_low_occ_btn.click()
    app.processEvents()
    n_after = model.potential_layer.tunable_parameters.occupancies[0].numel()
    check("Atome unter Occ 0.5 entfernt", n_after == n_before - 10, f"{n_before} -> {n_after}")
    check("Positions-/Intensitaets-Tensoren passen zur Atomzahl",
          model.potential_layer.init_positions_0.shape[0] == n_after
          and model.potential_layer.tunable_parameters.position_shifts[0].shape[0] == n_after)
    window.remove_low_occ_spin.setValue(0.0)
    window.remove_low_occ_btn.click()
    check("Schwelle 0 entfernt nichts", model.potential_layer.tunable_parameters.occupancies[0].numel() == n_after)
    loss = float(model.loss_multislice(target)[3])
    check("Modell rechnet nach dem Entfernen weiter", np.isfinite(loss), f"loss {loss:.4f}")

    shot = os.environ.get("GRADIENT_SHOT")
    if shot:
        window.show_gradients_checkbox.setChecked(True)
        window.gradient_descent_checkbox.setChecked(True)
        for _ in range(5):
            app.processEvents()
        canvas.fig.savefig(shot, dpi=90)

    print("\n" + ("alle Gradient-Tests bestanden" if not FAILS else "FEHLER: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
