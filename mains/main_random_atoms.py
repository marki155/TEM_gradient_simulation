"""Modell aus zufaellig eingefuegten Atomen aufbauen (statt aus einem Gitter).

Alternative zu ``main_graphene.py`` / ``imagej_main.py``: Es gibt kein
Startgitter. Atome werden einzeln an Zufallsstellen eingefuegt und behalten,
wenn die Simulation besser zum Bild passt (Regel wie ``ATOM ADDER`` in
``test.py``). Danach laesst sich das Modell in der GUI weitertrainieren.

Wie bei den Vorgaengern arbeitet das Skript auf einem Datenordner::

    mein_datensatz/
        target.tif            Zielbild (oder target.npy)
        mask_<Gruppe>.tif     Maske der Bildbereiche mit Atomen (oder gr_mask.tif)
        init.json             optional - dieselbe Datei wie beim ImageJ-Plugin
        config_random/        wird angelegt: config.json, train_method.json, ...

Aufruf (STEM, Graphen, Bild mit 6.81 nm Breite)::

    python3 mains/main_random_atoms.py mein_datensatz --mode stem --fov-nm 6.81 \\
        --trials 100000 --gui

Pixelgroesse, Spannung und Abbildungsmodus stehen sonst in der ``init.json``
des ImageJ-Plugins (``imaging.mode``, ``pixel_size_nm``, ``voltage_kV``,
``imaging.stem``); mit ``--pixel-size-nm`` / ``--fov-nm`` / ``--mode`` werden
sie ueberschrieben. Beim ersten Lauf ohne ``init.json`` legt das Skript eine
``init_random.json`` mit den Angaben an, danach reicht der Ordner. Die Gitter
und weitere Schichten der ``init.json`` werden ignoriert: der Seeder fuellt die
erste Gruppe/Schicht. Mit ``--resume`` geht es mit den Atomen eines frueheren
Laufs weiter.

Aufruf mit ``--help`` zeigt alle Optionen.
"""

import argparse
import glob
import json
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

TARGET_NAMES = ("target.tif", "target.tiff", "target.npy")
INIT_NAMES = ("init.json", "init_random.json")
DEFAULT_CONFIG_DIR = "config_random"
DEFAULT_GROUP = "Gr"


# ---------------------------------------------------------------------------
# Datenordner lesen
# ---------------------------------------------------------------------------

def read_image(path):
    from mains.imagej_main import _prepare_target, _read_image
    image = _prepare_target(_read_image(path))
    if image.ndim != 2:
        raise ValueError(f"Bild muss 2D sein, ist {image.shape}")
    return image


def group_name_from_mask(filename):
    """``mask_Gr.tif`` -> ``Gr``, ``gr_mask.tif`` -> ``gr``, ``mask.tif`` -> ``Gr``."""

    stem = os.path.splitext(os.path.basename(filename))[0]
    if stem.startswith("mask_"):
        stem = stem[len("mask_"):]
    elif stem.endswith("_mask"):
        stem = stem[:-len("_mask")]
    elif stem == "mask":
        stem = ""
    return stem or DEFAULT_GROUP


def find_masks(folder):
    found = []
    for pattern in ("mask*", "*_mask.*"):
        for path in sorted(glob.glob(os.path.join(folder, pattern))):
            if path.lower().endswith((".tif", ".tiff", ".npy")) and path not in found:
                found.append(path)
    return found


class Dataset:
    """Was aus dem Ordner (und ``init.json``) fuer den Seeder gebraucht wird."""

    def __init__(self, path, pixel_size_nm=None, fov_nm=None, mode=None, voltage_kV=None):
        path = os.path.abspath(path)
        self.init = {}
        self.init_file = None
        if os.path.isdir(path):
            self.base_dir = path
            for name in INIT_NAMES:
                if os.path.exists(os.path.join(path, name)):
                    self.init_file = os.path.join(path, name)
                    break
        elif path.endswith(".json"):
            self.base_dir = os.path.dirname(path)
            self.init_file = path
        else:
            raise ValueError(f"{path} ist weder ein Ordner noch eine init.json")
        if self.init_file:
            with open(self.init_file) as f:
                self.init = json.load(f)

        init = self.init
        imaging = init.get("imaging", {})

        # Zielbild
        target_name = init.get("target_file")
        if not target_name:
            for name in TARGET_NAMES:
                if os.path.exists(os.path.join(self.base_dir, name)):
                    target_name = name
                    break
        if not target_name or not os.path.exists(os.path.join(self.base_dir, target_name)):
            raise FileNotFoundError(
                f"Kein Zielbild in {self.base_dir} (erwartet: {', '.join(TARGET_NAMES)})")
        self.target_file = target_name
        self.image = read_image(os.path.join(self.base_dir, target_name))

        # Gruppe, Schicht und Maske
        groups = init.get("groups") or []
        layers = init.get("layers") or []
        group = groups[0] if groups else {}
        layer = layers[0] if layers else {}
        if len(groups) > 1 or len(layers) > 1:
            print("[seed] Hinweis: nur die erste Gruppe/Schicht der init.json wird befuellt")

        self.mask_file = group.get("mask_file")
        if not self.mask_file:
            masks = find_masks(self.base_dir)
            if len(masks) > 1:
                print(f"[seed] Hinweis: mehrere Masken gefunden, benutzt wird "
                      f"{os.path.basename(masks[0])}")
            if masks:
                self.mask_file = os.path.relpath(masks[0], self.base_dir)
        self.group_name = group.get("name") or (
            group_name_from_mask(self.mask_file) if self.mask_file else DEFAULT_GROUP)
        self.mask = None
        if self.mask_file:
            from mains.imagej_main import _read_image
            mask = np.asarray(_read_image(os.path.join(self.base_dir, self.mask_file))) > 0
            if mask.shape != self.image.shape:
                raise ValueError(f"Maske {mask.shape} und Zielbild {self.image.shape} "
                                 f"haben verschiedene Groesse")
            self.mask = mask
        else:
            print("[seed] Keine Maske gefunden - das ganze Bild wird benutzt")

        self.element = layer.get("element") or "C"
        self.debye_waller_u = layer.get("debye_waller_u")

        # Abbildung
        if pixel_size_nm is None and fov_nm is not None:
            pixel_size_nm = fov_nm / self.image.shape[1]
        if pixel_size_nm is None:
            pixel_size_nm = init.get("pixel_size_nm")
        if pixel_size_nm is None:
            raise ValueError("Pixelgroesse unbekannt: --pixel-size-nm oder --fov-nm angeben "
                             "(oder eine init.json mit pixel_size_nm in den Ordner legen)")
        self.pixel_size_nm = float(pixel_size_nm)
        if init.get("shape") and tuple(init["shape"]) != self.image.shape:
            raise ValueError(f"shape der init.json {tuple(init['shape'])} passt nicht zum "
                             f"Zielbild {self.image.shape}")

        if mode is None:
            mode = "tem" if imaging.get("mode", "stem_haadf") == "tem" else "stem"
        self.mode = mode
        self.voltage_kV = float(voltage_kV if voltage_kV is not None else init.get("voltage_kV", 80.0))
        self.padding = int(init.get("padding", 100))
        self.config_dir = init.get("config_dir_random", DEFAULT_CONFIG_DIR)

        stem = imaging.get("stem", {})
        self.optics = dict(
            aberrations=imaging.get("aberrations") or None,
            env_params=imaging.get("envelope") or None,
            amp_contrast=float(imaging.get("amp_contrast", 0.1)),
            alpha_max_mrad=float(imaging.get("alpha_max_mrad", 50.0)),
            probe_aberrations=stem.get("aberrations") or None,
            convergence_mrad=float(stem.get("convergence_mrad", 24.81)),
            detector_mrad=tuple(stem.get("detector_mrad", (60.0, 200.0))),
            source_size_A=float(stem.get("source_size_A", 0.3)))

    @property
    def out_dir(self):
        return os.path.join(self.base_dir, self.config_dir)

    def write_init_if_missing(self):
        """Pixelgroesse & Co. festhalten, damit der naechste Aufruf nur den Ordner braucht."""

        if self.init_file:
            return None
        path = os.path.join(self.base_dir, INIT_NAMES[1])
        with open(path, "w") as f:
            json.dump({"pixel_size_nm": self.pixel_size_nm, "voltage_kV": self.voltage_kV,
                       "imaging": {"mode": "tem" if self.mode == "tem" else "stem_haadf"}},
                      f, indent=2)
        return path


# ---------------------------------------------------------------------------
# Modell bauen
# ---------------------------------------------------------------------------

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
                        source_size_A=0.3, aberrations=None, padding=100, device=None,
                        mask=None, group_name="Gr", element="C", debye_waller_u=None,
                        probe_aberrations=None, env_params=None, amp_contrast=0.1,
                        alpha_max_mrad=50.0, mask_file=None, target_file=None,
                        start_positions=None):
    """Pool-Modell zum Seeden. ``mask`` (bool, optional) begrenzt die Wertung."""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    config = seed_config(
        image.shape, pixel_size_nm, capacity, voltage_kV=voltage_kV,
        imaging_mode="stem_haadf" if mode == "stem" else "tem",
        convergence_mrad=convergence_mrad, detector_mrad=detector_mrad,
        source_size_A=source_size_A, aberrations=aberrations, padding=padding,
        group_name=group_name, element=element, debye_waller_u=debye_waller_u,
        probe_aberrations=probe_aberrations, env_params=env_params,
        amp_contrast=amp_contrast, alpha_max_mrad=alpha_max_mrad,
        mask_file=mask_file, target_file=target_file, start_positions=start_positions)

    loss_mask = border_mask(image.shape, border_px, device)
    if mask is not None:
        loss_mask = loss_mask * torch.tensor(np.asarray(mask) > 0, dtype=torch.float32, device=device)
    train_method = build_seeding_train_method(config, loss_mask)
    model = NN_particles(config=config, pixel_size_nm=config.pixel_size_nm,
                         voltage_kV=config.voltage_kV, train_method=train_method,
                         device=device, train_methods=train_method)
    model.build()

    target = torch.tensor(image, dtype=torch.float32, device=device)
    target = (target - target.min()) / (target.max() - target.min())
    return model, target, train_method


def build_from_dataset(data, capacity=None, border_px=10, resume_positions=None, device=None,
                       overrides=None):
    """Modell, Zielbild und TrainMethod fuer einen Datensatz."""

    if capacity is None:
        if data.mask is not None:
            area_nm2 = float(data.mask.sum()) * data.pixel_size_nm ** 2
            capacity = int(np.ceil(area_nm2 * 38.2 * 1.5))
        else:
            capacity = graphene_capacity(data.image.shape, data.pixel_size_nm)
        capacity = max(capacity, 1)
    if resume_positions is not None:
        capacity = max(capacity, int(len(resume_positions) * 1.2) + 10)

    options = dict(data.optics)
    options.update(overrides or {})
    return build_seeding_model(
        data.image, data.pixel_size_nm, data.mode, capacity, voltage_kV=data.voltage_kV,
        border_px=border_px, padding=data.padding, device=device, mask=data.mask,
        group_name=data.group_name, element=data.element, debye_waller_u=data.debye_waller_u,
        mask_file=data.mask_file, start_positions=resume_positions, **options)


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


def save_dataset_state(seeder, train_method, data, group_name):
    """Config-Ordner anlegen: config.json, train_method.json, target.npy, xyz, Vorschau."""

    out_dir = data.out_dir
    os.makedirs(out_dir, exist_ok=True)
    write_learned_microscope(seeder.model)
    seeder.save(out_dir, config_file="config.json")
    np.save(os.path.join(out_dir, "target.npy"), seeder.target.detach().cpu().numpy())
    save_preview(seeder, os.path.join(out_dir, "seeded_preview.png"))

    # fuer das Weitertrainieren: Maske ohne Randstreifen, Atome frei
    saved_mask = train_method.masks.get(group_name)
    if data.mask is not None:
        train_method.masks[group_name] = torch.tensor(data.mask)
    layer = seeder.model.config.layer_names[0]
    train_method.layer(layer).free_positions.enabled = True
    train_method.layer(layer).intensities.enabled = True
    try:
        save_train_methods(train_method, os.path.join(out_dir, "train_method.json"))
    finally:
        if saved_mask is not None:
            train_method.masks[group_name] = saved_mask
        train_method.layer(layer).free_positions.enabled = False
        train_method.layer(layer).intensities.enabled = False
    return out_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="Datenordner mit target.tif und mask_*.tif (oder Pfad zu einer init.json)")
    size = p.add_mutually_exclusive_group()
    size.add_argument("--pixel-size-nm", type=float, help="Pixelgroesse in nm")
    size.add_argument("--fov-nm", type=float, help="Bildbreite in nm (Pixelgroesse = fov / Spalten)")
    p.add_argument("--mode", choices=["stem", "tem"], default=None,
                   help="Abbildungsmodus (Standard: aus der init.json, sonst stem)")
    p.add_argument("--voltage-kV", type=float, default=None)
    p.add_argument("--trials", type=int, default=20000, help="Zahl der Versuche")
    p.add_argument("--capacity", type=int, default=None,
                   help="Platz im Pool (Standard: 1.5 x erwartete Graphen-Atomzahl in der Maske)")
    p.add_argument("--resume", action="store_true",
                   help="mit den Atomen aus dem Config-Ordner weitermachen")
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
    p.add_argument("--config-dir", default=None,
                   help=f"Ausgabeordner im Datenordner (Standard: {DEFAULT_CONFIG_DIR})")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--convergence-mrad", type=float, default=None)
    p.add_argument("--source-size-A", type=float, default=None)
    p.add_argument("--gui", action="store_true", help="danach die GUI zum Weitertrainieren oeffnen")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    data = Dataset(args.folder, pixel_size_nm=args.pixel_size_nm, fov_nm=args.fov_nm,
                   mode=args.mode, voltage_kV=args.voltage_kV)
    if args.config_dir:
        data.config_dir = args.config_dir
    created = data.write_init_if_missing()
    if created:
        print(f"[seed] Einstellungen abgelegt in {created}")

    resume = None
    if args.resume:
        path = os.path.join(data.out_dir, "seeded_positions_px.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"--resume: {path} fehlt")
        resume = np.load(path)

    overrides = {k: v for k, v in dict(convergence_mrad=args.convergence_mrad,
                                       source_size_A=args.source_size_A).items() if v is not None}
    if resume is not None:
        # gelernte Mikroskop-Parameter des frueheren Laufs uebernehmen
        from mains.main import deserialize
        from nn import CONFIG_CLASS_REGISTRY
        with open(os.path.join(data.out_dir, "config.json")) as f:
            previous = deserialize(json.load(f), CONFIG_CLASS_REGISTRY)
        if data.mode == "stem" and previous.init_probe_aberrations:
            overrides.setdefault("probe_aberrations", previous.init_probe_aberrations)
        elif data.mode == "tem" and previous.init_aberrations:
            overrides.setdefault("aberrations", previous.init_aberrations)
        overrides.setdefault("env_params", previous.init_ctf_env_params)
        overrides.setdefault("amp_contrast", previous.init_amp_contrast)
    model, target, train_method = build_from_dataset(
        data, capacity=args.capacity, border_px=args.border_px,
        resume_positions=resume, overrides=overrides)
    capacity = model.potential_layer.get_init_positions(0).shape[0]
    print(f"[seed] {data.mode.upper()}, Bild {data.image.shape}, "
          f"{data.pixel_size_nm * 10:.4f} A/px, Gruppe {data.group_name!r}"
          f"{', Maske ' + data.mask_file if data.mask_file else ''}, Pool {capacity} Atome",
          flush=True)

    seeder = RandomAtomSeeder(
        model, target, loss=args.loss, touch_A=args.touch_A, merge_A=args.merge_A,
        margin_px=args.border_px, move_prob=args.move_prob, remove_prob=args.remove_prob,
        seed=args.seed, region=data.mask)
    if resume is not None:
        print(f"[seed] weiter mit {seeder.pool.n_active} Atomen, Loss {seeder.loss:.5f}")

    def save(s):
        out = save_dataset_state(s, train_method, data, data.group_name)
        print(f"[seed] gespeichert in {out}", flush=True)

    try:
        seeder.run(args.trials, tune_every=args.tune_every, tune_steps=args.tune_steps,
                   log_every=args.log_every, save_every=args.save_every, on_save=save)
    except KeyboardInterrupt:
        print("[seed] abgebrochen - speichere den aktuellen Stand")

    save(seeder)
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
