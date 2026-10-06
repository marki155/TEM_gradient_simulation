"""Modell aus zufaellig eingefuegten Atomen aufbauen (statt aus einem Gitter).

Alternative zu ``main_graphene.py`` / ImageJ-Plugin: Es gibt kein Startgitter.
Atome werden einzeln an Zufallsstellen eingefuegt und behalten, wenn die
Simulation besser zum Bild passt (Regel wie ``ATOM ADDER`` in ``test.py``).
Danach kann das Modell in der GUI normal weitertrainiert werden.

Beispiel (STEM, Graphen, Bild mit 6.81 nm Breite)::

    python3 mains/main_random_atoms.py bild.tif --mode stem --fov-nm 6.81 \\
        --trials 100000 --out seed_run --gui

Aufruf mit ``--help`` zeigt alle Optionen.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multislice  # noqa: E402

multislice.CHECK_PHASE = False

from atom_seeding import (  # noqa: E402
    AtomPool, RandomAtomSeeder, graphene_capacity, seed_config)
from nn import NN_particles  # noqa: E402
from training_parameter import TrainMethod, save_train_methods  # noqa: E402


def load_image(path):
    from mains.imagej_main import _prepare_target, _read_image
    image = _prepare_target(_read_image(path))
    if image.ndim != 2:
        raise ValueError(f"Bild muss 2D sein, ist {image.shape}")
    return image


def build_seeding_train_method(config, mask=None, lr_aberrations=1e-2, lr_norm=1e-3):
    """Nur Mikroskop-Parameter frei - die Atome setzt der Seeder."""

    tm = TrainMethod().ensure_for_config(config)
    tm.train_ctf_aberrations.enabled = True
    tm.train_ctf_aberrations.lr = lr_aberrations
    tm.train_normalisation_parameters.enabled = True
    tm.train_normalisation_parameters.lr = lr_norm
    name = config.group_names[0]
    tm.set_active_group(name)
    if mask is not None:
        tm.masks[name] = mask
    return tm


def border_mask(shape, border_px, device):
    mask = torch.ones(shape, dtype=torch.float32, device=device)
    if border_px > 0:
        mask[:border_px] = 0
        mask[-border_px:] = 0
        mask[:, :border_px] = 0
        mask[:, -border_px:] = 0
    return mask


def build_seeding_model(image, pixel_size_nm, mode, capacity, voltage_kV=80.0,
                        border_px=10, convergence_mrad=24.81, detector_mrad=(60.0, 200.0),
                        source_size_A=0.3, aberrations=None, padding=100, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    config = seed_config(
        image.shape, pixel_size_nm, capacity, voltage_kV=voltage_kV,
        imaging_mode="stem_haadf" if mode == "stem" else "tem",
        convergence_mrad=convergence_mrad, detector_mrad=detector_mrad,
        source_size_A=source_size_A, aberrations=aberrations, padding=padding)

    train_method = build_seeding_train_method(config, border_mask(image.shape, border_px, device))
    model = NN_particles(config=config, pixel_size_nm=config.pixel_size_nm,
                         voltage_kV=config.voltage_kV, train_method=train_method,
                         device=device, train_methods=train_method)
    model.build()

    target = torch.tensor(image, dtype=torch.float32, device=device)
    target = (target - target.min()) / (target.max() - target.min())
    return model, target, train_method


def write_learned_microscope(model):
    """Gelernte Aberrationen in die Config zurueckschreiben (sie hat sonst nur Startwerte)."""

    learned = {k: float(v.detach()) for k, v in
               model.imaging_layer.tunable_parameters.aberrations.items()}
    if model.config.imaging_mode == "stem_haadf":
        model.config.init_probe_aberrations = learned
    else:
        model.config.init_aberrations = learned


def save_preview(seeder, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        simulated = seeder.model.model(None)[0].detach().cpu().numpy()
    target = seeder.target.detach().cpu().numpy()
    positions = seeder.positions_px()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(target, cmap="gray")
    axes[0].set_title("Ziel")
    axes[1].imshow(simulated, cmap="gray")
    axes[1].set_title(f"Simulation, {len(positions)} Atome")
    axes[2].imshow(target, cmap="gray")
    axes[2].scatter(positions[:, 1], positions[:, 0], s=8, c="r")
    axes[2].set_title("Atome")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="Zielbild (.tif / .npy)")
    size = p.add_mutually_exclusive_group(required=True)
    size.add_argument("--pixel-size-nm", type=float, help="Pixelgroesse in nm")
    size.add_argument("--fov-nm", type=float, help="Bildbreite in nm (Pixelgroesse = fov / Spalten)")
    p.add_argument("--mode", choices=["stem", "tem"], default="stem")
    p.add_argument("--voltage-kV", type=float, default=80.0)
    p.add_argument("--trials", type=int, default=20000, help="Zahl der Versuche")
    p.add_argument("--capacity", type=int, default=None,
                   help="Platz im Pool (Standard: 1.5 x erwartete Graphen-Atomzahl)")
    p.add_argument("--loss", choices=["model", "ncc"], default="model",
                   help="model = Loss aus der TrainMethod, ncc = Korrelation wie im Skript")
    p.add_argument("--touch-A", type=float, default=0.5, help="Atome naeher als das verschwinden")
    p.add_argument("--merge-A", type=float, default=1.0, help="Atome naeher als das werden verschmolzen")
    p.add_argument("--border-px", type=int, default=10, help="Bildrand ohne Wertung (Randartefakte)")
    p.add_argument("--move-prob", type=float, default=0.0, help="Anteil Verschiebe-Versuche (optional)")
    p.add_argument("--remove-prob", type=float, default=0.0, help="Anteil Entfern-Versuche (optional)")
    p.add_argument("--tune-every", type=int, default=2500,
                   help="alle n Versuche Aberrationen/Normierung nachziehen (0 = nie)")
    p.add_argument("--tune-steps", type=int, default=3)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--out", default="seed_run", help="Ausgabeordner")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--convergence-mrad", type=float, default=24.81)
    p.add_argument("--source-size-A", type=float, default=0.3)
    p.add_argument("--gui", action="store_true", help="danach die GUI zum Weitertrainieren oeffnen")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    image = load_image(args.image)
    pixel_size_nm = args.pixel_size_nm or args.fov_nm / image.shape[1]
    capacity = args.capacity or graphene_capacity(image.shape, pixel_size_nm)

    model, target, train_method = build_seeding_model(
        image, pixel_size_nm, args.mode, capacity, voltage_kV=args.voltage_kV,
        border_px=args.border_px, convergence_mrad=args.convergence_mrad,
        source_size_A=args.source_size_A)
    print(f"[seed] {args.mode.upper()}, Bild {image.shape}, {pixel_size_nm * 10:.4f} A/px, "
          f"Pool {capacity} Atome", flush=True)

    seeder = RandomAtomSeeder(
        model, target, loss=args.loss, touch_A=args.touch_A, merge_A=args.merge_A,
        margin_px=args.border_px, move_prob=args.move_prob, remove_prob=args.remove_prob,
        seed=args.seed)

    os.makedirs(args.out, exist_ok=True)

    def save(s):
        write_learned_microscope(s.model)
        s.save(args.out)
        save_preview(s, os.path.join(args.out, "seeded_preview.png"))
        print(f"[seed] gespeichert in {args.out}", flush=True)

    stop = {"flag": False}
    try:
        seeder.run(args.trials, tune_every=args.tune_every, tune_steps=args.tune_steps,
                   log_every=args.log_every, save_every=args.save_every, on_save=save,
                   should_stop=lambda: stop["flag"])
    except KeyboardInterrupt:
        print("[seed] abgebrochen - speichere den aktuellen Stand")

    save(seeder)
    save_train_methods(train_method, os.path.join(args.out, "train_method.json"))
    print(f"[seed] fertig: {seeder.pool.n_active} Atome, Loss {seeder.loss:.5f}, "
          f"{seeder.stats.accepted}/{seeder.stats.trials} Versuche angenommen")

    if args.gui:
        open_gui(model, target, train_method)
    return 0


def open_gui(model, target, train_method):
    """Die Atome sind gesetzt: ab jetzt Positionen, Intensitaeten und Mikroskop trainieren."""

    layer = model.config.layer_names[0]
    train_method.layer(layer).free_positions.enabled = True
    train_method.layer(layer).intensities.enabled = True
    model.potential_layer.apply_trainable_selection()

    from gui_qt import launch_gui_qt
    launch_gui_qt(model=model, target=target,
                  pot_layer=model.potential_layer, ctf_layer=model.imaging_layer,
                  train_method=train_method, plot_every=25, heavy_every=50,
                  max_steps=2_000_000)


if __name__ == "__main__":
    raise SystemExit(main())
