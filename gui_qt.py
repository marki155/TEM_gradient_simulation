import math
import sys
import time
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import torch

from PySide6 import QtCore, QtWidgets

import matplotlib

from ctf import TunableCTFParameters
from nn import ParticlesTunableParameters, is_train_param_instance
from parameter_frame import ParametersFrame
from potential import TunablePotentialParameters
from training_parameter import TrainMethod

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import patches

from train_methods_gui_builder_qt import TrainMethodEditor

from scipy.ndimage import gaussian_filter

SHOW_POTENTIAL=True

#: Anzeige-Interpolation der Bildpanels. "antialiased" (Matplotlib-Default)
#: filtert beim Zeichnen ueber das ganze Bild und kostet hier ~1 s pro Redraw.
IMAGE_INTERPOLATION = "nearest"

#: Potentialbilder nur in dieser Verkleinerung anzeigen (spart Zeichenzeit;
#: das Panel ist ohnehin nur ein paar hundert Pixel breit).
POTENTIAL_DISPLAY_DOWNSAMPLE = 2

#: Kantenlaenge, mit der die Strain-Karten fuer die Anzeige abgetastet werden.
#: Das Feld ist ein Spline mit ~10x10 Kontrollpunkten - auf dem vollen Gitter
#: (hier 1892x2045) dauert das ueber 7 s pro Aktualisierung, hier 0.1 s.
STRAIN_MAP_RESOLUTION = 256



@dataclass
class VizPayload:
    step: int
    loss: float
    losses: np.ndarray

    sim_img: Optional[np.ndarray] = None
    diff_img: Optional[np.ndarray] = None
    target_img: Optional[np.ndarray] = None
    mask_img: Optional[np.ndarray] = None

    ctf_phase: Optional[np.ndarray] = None
    #: STEM: Sondenintensitaet statt CTF-Phase
    probe_img: Optional[np.ndarray] = None
    ring_mask: Optional[np.ndarray] = None
    p_diff: Optional[np.ndarray] = None

    pot_positions: Optional[List[np.ndarray]] = None
    pot_intensities: Optional[List[np.ndarray]] = None

    hist_counts: Optional[List[np.ndarray]] = None
    hist_edges: Optional[np.ndarray] = None

    train_method: TrainMethod = None

    strain_maps: Optional[Dict[str, np.ndarray]] = None
    strain_group: Optional[str] = None

    status: str = "idle"

    potential_maps: List = None

    #: Schicht-/Gruppen-Info, damit die Anzeige nicht auf 4 Schichten festhaengt
    layer_names: Optional[List[str]] = None
    layer_groups: Optional[List[str]] = None
    active_group: Optional[str] = None


def potential_images_for_display(pot_layer, downsample=POTENTIAL_DISPLAY_DOWNSAMPLE):
    """Potentialbilder fuer die Anzeige - bei Bedarf ausgeduennt."""

    images = pot_layer.get_potential_images()
    if downsample and downsample > 1:
        images = [image[::downsample, ::downsample] for image in images]
    return images


def collect_layer_visuals(pot_layer):
    """Positionen, Intensitaeten und Histogramme aller Schichten einsammeln."""

    pot_positions = []
    pot_intensities = []
    for i in range(pot_layer.N_l):
        (px, py), ints = pot_layer.get_pos_and_ints(i)
        pot_positions.append(np.column_stack([px, py]).astype(np.float32))
        pot_intensities.append(np.asarray(ints, dtype=np.float32))

    # nur besetzte Atome (die anderen sind NaN), alle Schichten auf denselben
    # Klassen, damit die Kurven vergleichbar sind
    hist_counts = None
    hist_edges = None
    values = [v[np.isfinite(v)] for v in pot_intensities]
    filled = [v for v in values if v.size]
    if filled:
        low = min(float(v.min()) for v in filled)
        high = max(float(v.max()) for v in filled)
        if high - low < 1e-6:
            low, high = low - 0.5, high + 0.5
        hist_edges = np.linspace(low, high, 101, dtype=np.float32)
        hist_counts = [np.histogram(v, bins=hist_edges)[0].astype(np.float32) for v in values]

    return pot_positions, pot_intensities, hist_counts, hist_edges


PROBE_VIEW_MAX_PX = 64


def probe_image(ctf_layer):
    """STEM: Sondenintensitaet, zentriert und auf den sichtbaren Kern zugeschnitten."""

    with torch.no_grad():
        profile = ctf_layer.probe_profile().detach().cpu().numpy()
    height, width = profile.shape
    cy, cx = height // 2, width // 2
    yy, xx = np.indices(profile.shape)
    radius = np.hypot(yy - cy, xx - cx).ravel()
    order = np.argsort(radius)
    cumulative = np.cumsum(profile.ravel()[order])
    cumulative /= max(cumulative[-1], 1e-30)
    r90 = radius[order][min(np.searchsorted(cumulative, 0.9), len(order) - 1)]
    half = int(np.clip(np.ceil(2.5 * r90), 8, min(PROBE_VIEW_MAX_PX, cy, cx)))
    return profile[cy - half:cy + half + 1, cx - half:cx + half + 1]


def ctf_visuals(ctf_layer, stem):
    """(CTF-Phase, Ring, Sonde) - im STEM nur die Sonde."""

    try:
        if stem:
            return None, None, probe_image(ctf_layer)
        dp, env, ring_mask, psf = ctf_layer.to_diffraction_patterns(
            max_angle_mrad=42.0, gpts=1300)
        return (np.angle(dp.detach().cpu().numpy()), ring_mask.detach().cpu().numpy(), None)
    except Exception as exc:
        print("CTF update failed:", exc)
        return None, None, None


def strain_maps_for_active_group(pot_layer, train_method, resolution=STRAIN_MAP_RESOLUTION):
    """Strain-Maps der aktiven Gruppe (ohne aktive Gruppe: letzte Gruppe)."""

    name = train_method.active_group
    if name is None or name not in pot_layer.strain_fields:
        name = pot_layer.group_names[-1]

    field = pot_layer.strain_fields[name]
    scale = resolution / max(field.H, field.W)
    H = max(32, int(round(field.H * scale)))
    W = max(32, int(round(field.W * scale)))
    return field.strain_maps(H=H, W=W), name


class TrainingWorker(QtCore.QObject):
    progress_light = QtCore.Signal(int, float, object)
    progress_heavy = QtCore.Signal(object)
    status_changed = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(
        self,
        model,
        optimizer,
        target,
        pot_layer,
        ctf_layer,
        train_method,
        plot_every=5,
        heavy_every=10,
        max_steps=2_000_000,
        parent=None,
    ):
        super().__init__(parent)

        self.model = model
        self.optimizer = optimizer
        self.target = target
        self.pot_layer = pot_layer
        self.ctf_layer = ctf_layer
        self.train_method = train_method

        self.plot_every = plot_every
        self.heavy_every = heavy_every
        self.max_steps = max_steps

        self.show_positions_on_target = True

        self._stop = False
        self._pause = False
        self.parked = False
        self.losses = []
        self.deleted_steps = 0


    def _set_optimizer_lr(self):
        lr = None
        if hasattr(self.train_method, "learning_rate"):
            lr = float(self.train_method.learning_rate)
        elif hasattr(self.train_method, "learing_rate"):
            lr = float(self.train_method.learing_rate)

        if lr is None:
            return

        if hasattr(self.optimizer, "param_groups"):
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        elif hasattr(self.optimizer, "learning_rate"):
            self.optimizer.learning_rate = lr

    def sync_optimizer_lrs_from_train_method(self):
        for pg in self.optimizer.param_groups:
            name = pg.get("name")
            if not name or not hasattr(self.train_method, name):
                continue

            tp = getattr(self.train_method, name)
            if is_train_param_instance(tp):
                pg["lr"] = float(tp.lr)

    #spin_values_changed = QtCore.Signal(float, float, float, float, float, float, float,n  bool)  # scale, rot, shift_x, shift_y
    parameter_update = QtCore.Signal(ParticlesTunableParameters)

    #: Strain-Karten kosten Zeit und sind beim Seeding ohne Bedeutung
    with_strain = True

    def _stem(self):
        return getattr(self.model.config, "imaging_mode", "tem") == "stem_haadf"

    def _strain_maps(self):
        if not self.with_strain:
            return None, None
        return strain_maps_for_active_group(self.pot_layer, self.train_method)

    def _wait_while_paused(self):
        if self._pause and not self._stop:
            self.parked = True
            self.status_changed.emit("paused")
            while self._pause and not self._stop:
                QtCore.QThread.msleep(20)
            self.parked = False

    def wait_until_parked(self, timeout=5.0):
        """Wartet, bis der Worker mitten in keinem Schritt mehr steckt.

        Danach darf die GUI das Modell selbst auswerten, ohne dass ein Versuch
        im Worker gleichzeitig die Tensoren umbaut.
        """

        end = time.monotonic() + timeout
        while not self.parked and time.monotonic() < end:
            QtCore.QThread.msleep(5)
        return self.parked

    def reset_losses(self):
        self.deleted_steps = len(self.losses)
        self.losses = []

    def get_step(self):
        return len(self.losses) + self.deleted_steps

    @QtCore.Slot()
    def request_heavy_refresh(self):
        print("heavy refresh received")
        #try:
        with torch.no_grad():
            simulated_image, target_image, p_diff, loss = self.model.loss_multislice(
                self.target
            )

            loss_item = float(loss.detach().cpu().item())
            payload = self.build_heavy_payload(
                step=len(self.losses) - 1 if len(self.losses) > 0 else 0,
                loss_item=loss_item,
                simulated_image=simulated_image,
                target_image=target_image,
                p_diff=p_diff,
            )
            self.progress_heavy.emit(payload)
        #except Exception as exc:
        #    print("request_heavy_refresh failed:", exc)

    def emit_parameter_update(self):
        self.parameter_update.emit(self.model.get_tunable_parameters())

    def build_heavy_payload(self, step: int, loss_item: float, simulated_image, target_image, p_diff):
        _, mask_img = self.model.get_current_target_and_mask_img(target_image)
        if mask_img is not None:
            mask_img = mask_img.detach().cpu().numpy()
        sim_img = simulated_image.detach().cpu().numpy()
        tgt_img = target_image.detach().cpu().numpy()
        diff_img = sim_img - tgt_img

        (pot_positions, pot_intensities,
         hist_counts, hist_edges) = collect_layer_visuals(self.pot_layer)

        ctf_phase, ring, probe_img = ctf_visuals(self.ctf_layer, self._stem())

        p_diff_np = None
        if p_diff is not None:
            p_diff_np = p_diff.detach().cpu().numpy()

        strain_maps, strain_group = self._strain_maps()

        pot_imgs = potential_images_for_display(self.pot_layer)





        payload = VizPayload(
            step=step,
            loss=loss_item,
            losses=np.asarray(self.losses, dtype=np.float32).copy(),
            sim_img=sim_img,
            diff_img=diff_img,
            target_img=tgt_img,
            mask_img=mask_img,
            ctf_phase=ctf_phase,
            probe_img=probe_img,
            ring_mask=ring,
            p_diff=p_diff_np,
            pot_positions=pot_positions,
            pot_intensities=pot_intensities,
            hist_counts=hist_counts,
            hist_edges=hist_edges,
            status="running",
            train_method=self.train_method,
            strain_maps=strain_maps,
            strain_group=strain_group,
            potential_maps=pot_imgs,
            layer_names=self.pot_layer.layer_names,
            layer_groups=[l.group for l in self.pot_layer.layers],
            active_group=self.train_method.active_group,
        )
        return payload

    @QtCore.Slot()
    def run(self):
        # Slot wird nicht ueberschrieben: PySide fuehrt einen ueberschriebenen
        # Slot im GUI-Thread aus statt im Worker-Thread (Seeding hing dort)
        self._execute()

    def _execute(self):
        self._stop = False
        self._pause = False
        self.status_changed.emit("running")

        for step in range(self.max_steps):
            if self._stop:
                break

            self._wait_while_paused()

            if self._stop:
                break

            self.optimizer.zero_grad(set_to_none=True)
            self.sync_optimizer_lrs_from_train_method()
            #self._set_optimizer_lrs()

            simulated_image, target_image, p_diff, loss = self.model.loss_multislice(self.target)
            try:
                loss.backward()
            except:
                print("Loss failed:", sys.exc_info()[0])

            if hasattr(self.pot_layer, "zero_grads_for_frozen"):
                self.pot_layer.zero_grads_for_frozen()

            self.optimizer.step()

            loss_item = float(loss.detach().cpu().item())
            self.losses.append(loss_item)

            if step % self.plot_every == 0 or step == 0:
                self.progress_light.emit(
                    step,
                    loss_item,
                    np.asarray(self.losses, dtype=np.float32).copy()
                )
                #self.emit_current_spin_values()
                self.emit_parameter_update()

            if step % self.heavy_every == 0 or step == 0:
                payload = self.build_heavy_payload(
                    step=step,
                    loss_item=loss_item,
                    simulated_image=simulated_image,
                    target_image=target_image,
                    p_diff=p_diff,
                )
                self.progress_heavy.emit(payload)

            QtCore.QThread.msleep(1)

        self.status_changed.emit("stopped" if self._stop else "done")
        self.finished.emit()

    @QtCore.Slot()
    def pause(self):
        self._pause = True
        self.status_changed.emit("paused")

    @QtCore.Slot()
    def resume(self):
        self._pause = False
        self.status_changed.emit("running")

    @QtCore.Slot()
    def stop(self):
        self._stop = True
        self._pause = False
        self.status_changed.emit("stopping...")


class SeedingWorker(TrainingWorker):
    """Setzt Atome zufaellig ein (``RandomAtomSeeder``) statt Gradientenschritte zu machen.

    Gleiche Signale wie der Trainings-Worker, damit die GUI das Einfuegen live
    zeigt: Loss-Kurve, Simulation, Diff und die Atome auf dem Zielbild.
    """

    with_strain = False

    def __init__(self, seeder, max_trials=100_000, tune_every=0, tune_steps=3,
                 light_interval=0.25, heavy_interval=1.5, save_callback=None,
                 save_interval=120.0, **kwargs):
        super().__init__(model=seeder.model, optimizer=None, **kwargs)
        self.seeder = seeder
        self.max_trials = max_trials
        self.tune_every = tune_every
        self.tune_steps = tune_steps
        self.light_interval = light_interval
        self.heavy_interval = heavy_interval
        self.save_callback = save_callback
        self.save_interval = save_interval

    def _emit_heavy(self):
        started = time.monotonic()
        with torch.no_grad():
            simulated_image, target_image, p_diff, _loss = self.model.loss_multislice(self.target)
        loss = self.seeder.loss
        payload = self.build_heavy_payload(
            step=self.seeder.stats.trials, loss_item=loss if math.isfinite(loss) else 0.0,
            simulated_image=simulated_image, target_image=target_image, p_diff=p_diff)
        self.progress_heavy.emit(payload)
        # die GUI darf nicht von Bildern ueberrollt werden, wenn das Zeichnen langsam ist
        self.heavy_interval = max(self.heavy_interval, 3.0 * (time.monotonic() - started))

    def _emit_light(self):
        stats = self.seeder.stats
        self.progress_light.emit(stats.trials, float(self.seeder.loss),
                                 np.asarray(self.losses, dtype=np.float32).copy())
        self.status_changed.emit(
            f"seeding: {self.seeder.n_atoms} Atome, {stats.accepted}/{stats.trials} angenommen")
        self.emit_parameter_update()

    def _save(self):
        if self.save_callback is not None:
            try:
                self.save_callback(self.seeder)
            except Exception as exc:
                print("seeding: Speichern fehlgeschlagen:", exc)

    def _execute(self):
        self._stop = False
        self._pause = False
        self.status_changed.emit("seeding")

        seeder = self.seeder
        started = time.monotonic()
        try:
            reason = self._seed_loop(seeder, started)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            reason = f"Fehler: {exc}"
        try:
            self._emit_light()
            self._emit_heavy()
            self._save()
        except Exception:
            import traceback
            traceback.print_exc()
        self.status_changed.emit(f"{reason}: {seeder.n_atoms} Atome")
        self.finished.emit()

    def _seed_loop(self, seeder, started):
        last_light = last_heavy = last_save = 0.0
        done = 0
        reason = "done"
        while done < self.max_trials:
            self._wait_while_paused()
            if self._stop:
                reason = "stopped"
                break

            seeder.trial()
            done += 1
            if seeder.stats.full:
                reason = "Atomgrenze erreicht"
                break
            if self.tune_every and done % self.tune_every == 0:
                seeder.tune(self.tune_steps)
            if math.isfinite(seeder.loss):
                self.losses.append(seeder.loss)

            now = time.monotonic() - started
            if self.losses and now - last_light >= self.light_interval:
                last_light = now
                self._emit_light()
            if now - last_heavy >= self.heavy_interval:
                last_heavy = now
                self._emit_heavy()
            if now - last_save >= self.save_interval:
                last_save = now
                self._save()
            QtCore.QThread.msleep(1)
        return reason


class MplCanvas(FigureCanvasQTAgg):
    #: feste Kacheln vor bzw. nach den Potential-Kacheln
    LEADING_TILES = ("loss", "sim", "diff", "target", "ctf")
    TRAILING_TILES = ("hist", "ux", "uy", "eps_xx", "eps_yy", "eps_xy")
    COLUMNS = 5
    STRAIN_TILES = ("ux", "uy", "eps_xx", "eps_yy", "eps_xy")

    def __init__(self, target_tensor=None, parent=None, train_method=None, config=None):
        self.fig = Figure(figsize=(12, 8), dpi=100, constrained_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.train_method = train_method
        self.config = config

        self.layer_names = list(config.layer_names) if config is not None else []
        self.layer_groups = [l.group for l in config.layers] if config is not None else []

        self.target_tensor = target_tensor
        self._target_shape = None

        #: None = alle Kacheln, sonst der Schluessel der einen gezeigten Kachel
        self.single_key = None
        #: Loss-Verlauf, damit die Kurve einen Kachelwechsel ueberlebt
        self._losses = np.zeros(0, dtype=np.float32)

        self.show_positions_on_target = True

        self.stem = getattr(config, "imaging_mode", "tem") == "stem_haadf"
        #: beim Seeding gibt es kein Strain-Feld - die Kacheln entfallen
        self.seeding = False

        self._rebuild_layout()

    # ------------------------------------------------------------------
    # Kacheln
    # ------------------------------------------------------------------
    def all_tile_keys(self):
        """Alle Kacheln in der Reihenfolge, in der sie im Raster liegen."""

        n_layers = max(len(self.layer_names), 1)
        trailing = [k for k in self.TRAILING_TILES
                    if not (self.seeding and k in self.STRAIN_TILES)]
        return (list(self.LEADING_TILES)
                + [f"pot{i}" for i in range(n_layers)]
                + trailing)

    def set_seeding(self, seeding):
        """Strain-Kacheln aus- bzw. einblenden; True, wenn sich das Raster geaendert hat."""

        seeding = bool(seeding)
        if seeding == self.seeding:
            return False
        self.seeding = seeding
        if self.single_key not in self.all_tile_keys():
            self.single_key = None
        self._rebuild_layout()
        self.draw_idle()
        return True

    def tile_titles(self):
        """{Kachel: Anzeigename} - auch fuer die Auswahl in der GUI."""

        titles = {"loss": "Loss", "sim": "Simulated", "diff": "Diff",
                  "target": "Target", "ctf": "Probe" if self.stem else "CTF", "hist": "Histogramm",
                  "ux": "ux", "uy": "uy",
                  "eps_xx": "eps xx", "eps_yy": "eps yy", "eps_xy": "eps xy"}
        for i in range(max(len(self.layer_names), 1)):
            name = self.layer_names[i] if i < len(self.layer_names) else str(i)
            titles[f"pot{i}"] = f"Potential {name}"
        return titles

    def set_single_tile(self, key):
        """Eine Kachel gross zeigen; ``None`` zeigt wieder alle.

        Die Artists werden dabei neu angelegt, das Bild ist also erst nach der
        naechsten Aktualisierung wieder gefuellt - die GUI stoesst sie an.
        """

        if key == self.single_key:
            return
        self.single_key = key
        self._rebuild_layout()
        self.draw_idle()

    def _rebuild_layout(self):
        self.fig.clear()

        keys = self.all_tile_keys()
        titles = self.tile_titles()

        if self.single_key in keys:
            # eine Kachel gross, daneben immer der Loss-Verlauf
            keys = [self.single_key] if self.single_key == "loss" else [self.single_key, "loss"]
        if len(keys) == 2 and keys[1] == "loss" and self.single_key in keys:
            grid = self.fig.add_gridspec(1, 2, width_ratios=[2.5, 1])
            self.axes = [self.fig.add_subplot(grid[0, i]) for i in range(2)]
        else:
            columns = 1 if len(keys) == 1 else self.COLUMNS
            rows = int(np.ceil(len(keys) / float(columns)))
            self.axes = [self.fig.add_subplot(rows, columns, i + 1) for i in range(len(keys))]
        self.axes_cmap = {key: ax for key, ax in zip(keys, self.axes)}
        for key, ax in self.axes_cmap.items():
            ax.set_title(titles.get(key, key))
            if key not in ("loss", "hist"):
                ax.set_xticks([])
                ax.set_yticks([])

        if self.target_tensor is not None:
            dummy = np.zeros_like(self.target_tensor.detach().cpu().numpy(), dtype=np.float32)
        else:
            dummy = np.zeros((64, 64), dtype=np.float32)

        def image(key, cmap, data=None, **kwargs):
            ax = self.axes_cmap.get(key)
            if ax is None:
                return None
            return ax.imshow(dummy if data is None else data, cmap=cmap, origin="lower",
                             interpolation=IMAGE_INTERPOLATION, **kwargs)

        self.loss_line = None
        if "loss" in self.axes_cmap:
            loss_ax = self.axes_cmap["loss"]
            self.loss_line, = loss_ax.plot([], [])
            loss_ax.set_xlabel("Step")
            loss_ax.set_ylabel("Loss")
            if len(self._losses) > 1:
                # der Verlauf steckt im Canvas, nicht in der Linie - sonst
                # waere die Kurve nach jedem Kachelwechsel leer
                self.loss_line.set_data(np.arange(len(self._losses)), self._losses)
                loss_ax.relim()
                loss_ax.autoscale_view()

        self.im_sim = image("sim", "viridis")
        self.im_diff = image("diff", "viridis")
        self.im_tgt = image("target", "viridis")
        self.im_tgt_sc = (self.axes_cmap["target"].scatter([], [], c=[], cmap="Wistia", s=2)
                          if "target" in self.axes_cmap else None)

        self.im_ux = image("ux", "RdBu")
        self.im_uy = image("uy", "RdBu")
        self.im_eps_xx = image("eps_xx", "RdBu")
        self.im_eps_yy = image("eps_yy", "RdBu")
        self.im_eps_xy = image("eps_xy", "RdBu")

        blank = np.zeros((10, 10), dtype=np.float32)
        if self.stem:
            self.im_ctf = image("ctf", "viridis", data=blank)
            self.im_ring = None
        else:
            self.im_ctf = image("ctf", "RdBu", data=blank, vmin=-np.pi, vmax=np.pi)
            self.im_ring = image("ctf", "Greens", data=blank, alpha=0.35)

        self.pot_scatters = []
        for i in range(max(len(self.layer_names), 1)):
            ax = self.axes_cmap.get(f"pot{i}")
            if ax is None:
                self.pot_scatters.append(None)
            elif SHOW_POTENTIAL:
                self.pot_scatters.append(image(f"pot{i}", "RdBu"))
            else:
                self.pot_scatters.append(ax.scatter([], [], s=2, c=[], cmap="viridis"))

        self.hist_lines = []
        if "hist" in self.axes_cmap:
            for i in range(max(len(self.layer_names), 1)):
                label = self.layer_names[i] if i < len(self.layer_names) else str(i)
                line, = self.axes_cmap["hist"].plot([], [], label=label)
                self.hist_lines.append(line)
            self.axes_cmap["hist"].legend(fontsize="x-small")

    def set_target_rectangles(self, rectangles):
        if rectangles is None or "target" not in self.axes_cmap:
            return
        for x0, y0, x1, y1 in rectangles:
            rect = patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=1,
                edgecolor="red",
                facecolor="none",
            )
            self.axes_cmap["target"].add_patch(rect)

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------
    def target_axis(self):
        """Die Target-Achse, falls sie gerade gezeigt wird."""

        return self.axes_cmap.get("target")

    def image_artists(self):
        """{Kachel: Bild-Artist} - nur die, die es gerade wirklich gibt."""

        pairs = {"sim": self.im_sim, "diff": self.im_diff, "target": self.im_tgt,
                 "ctf": self.im_ctf,
                 "ux": self.im_ux, "uy": self.im_uy, "eps_xx": self.im_eps_xx,
                 "eps_yy": self.im_eps_yy, "eps_xy": self.im_eps_xy}
        if SHOW_POTENTIAL:
            for i, artist in enumerate(self.pot_scatters):
                pairs[f"pot{i}"] = artist
        return {key: art for key, art in pairs.items() if art is not None}

    def reset_zoom(self):
        """Zoom aller sichtbaren Bildpanels auf das ganze Bild zuruecksetzen."""

        for key, artist in self.image_artists().items():
            ax = self.axes_cmap.get(key)
            data = artist.get_array()
            if ax is None or data is None or np.ndim(data) < 2:
                continue
            height, width = data.shape[0], data.shape[1]
            ax.set_xlim(-0.5, width - 0.5)
            ax.set_ylim(-0.5, height - 0.5)
        self.draw_idle()

    def _scatter_size(self, ax):
        """Punktgroesse an den Zoom anpassen - sonst sieht man beim
        Reinzoomen nur noch Staub, beim Rauszoomen nur noch Farbe."""

        if ax is None or self._target_shape is None:
            return 0.5
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        span = max(abs(x1 - x0), abs(y1 - y0), 1e-6)
        return float(np.clip(1200.0 / span, 0.5, 60.0))

    def _apply_target_image(self, artist, ax, image):
        """Bild setzen, ohne den Zoom zu verlieren."""

        shape = image.shape
        first_or_changed = self._target_shape != shape
        self._target_shape = shape

        artist.set_data(image)
        vmin = float(np.nanmin(image))
        vmax = float(np.nanmax(image))
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax == vmin:
            vmax = vmin + 1e-12
        artist.set_clim(vmin, vmax)
        artist.set_extent((-0.5, shape[1] - 0.5, -0.5, shape[0] - 0.5))

        if first_or_changed:
            ax.set_xlim(-0.5, shape[1] - 0.5)
            ax.set_ylim(-0.5, shape[0] - 0.5)

    def _update_target_scatter(self, scatter, ax, payload):
        pos, ints = self.get_pos_and_ints(payload)
        ints = np.asarray(ints, dtype=np.float32)

        if len(ints) == 0:
            scatter.set_offsets(np.empty((0, 2), dtype=np.float32))
            scatter.set_array(np.array([], dtype=np.float32))
            return

        scatter.set_offsets(np.column_stack([np.asarray(pos[1]), np.asarray(pos[0])]))
        scatter.set_array(ints)
        finite = ints[np.isfinite(ints)]
        if finite.size:
            imin, imax = float(finite.min()), float(finite.max())
            scatter.set_clim(imin, imax if imax > imin else imin + 1e-12)
        scatter.set_sizes(np.full(len(ints), self._scatter_size(ax)))

    def _safe_set_image(self, artist, arr):
        artist.set_data(arr)
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax):
            vmax = 1.0
        if vmin == vmax:
            vmax = vmin + 1e-12
        artist.set_clim(vmin, vmax)

    def update_light(self, step: int, loss: float, losses: np.ndarray):
        self._losses = np.asarray(losses, dtype=np.float32)
        loss_ax = self.axes_cmap.get("loss")
        if len(losses) > 1 and loss_ax is not None and self.loss_line is not None:
            xs = np.arange(len(losses), dtype=np.int32)
            self.loss_line.set_data(xs, losses)
            loss_ax.relim()
            loss_ax.autoscale_view()
            self.redraw_loss_only()

    def redraw_loss_only(self) -> bool:
        """Nur das Loss-Panel neu zeichnen statt der ganzen Figur.

        Ein kompletter Redraw kostet bei 15 Panels mit Megapixel-Bildern
        ueber eine Sekunde - fuer die Loss-Kurve reicht die eine Achse.
        """

        loss_ax = self.axes_cmap.get("loss")
        if loss_ax is None:
            return False

        try:
            renderer = self.get_renderer()
            loss_ax.draw(renderer)
            bbox = loss_ax.get_tightbbox(renderer)
            self.blit(bbox.expanded(1.15, 1.15) if bbox is not None else loss_ax.bbox)
            return True
        except Exception as exc:  # Fallback: normal zeichnen
            print("Loss-Blit fehlgeschlagen, zeichne komplett:", exc)
            self.draw_idle()
            return False

        #self.axes[1].set_title(f"Simulated (Step {step})")
        #self.axes[2].set_title(f"Diff (Step {step})")
        #self.axes[3].set_title(f"Target (Step {step})")

    def get_pos_and_ints(self, payload):
        """Atompositionen fuer die Target-Ansicht - nur die aktive Gruppe."""

        if not self.show_positions_on_target or not payload.pot_positions:
            return ([], []), []

        layer_groups = payload.layer_groups or []
        active = payload.active_group

        if active is None or not layer_groups:
            indices = list(range(len(payload.pot_positions)))
        else:
            indices = [i for i, g in enumerate(layer_groups) if g == active]
        if not indices:
            return ([], []), []

        pos = np.concatenate([payload.pot_positions[i] for i in indices], axis=0)
        ints = np.concatenate([payload.pot_intensities[i] for i in indices], axis=0)

        x_min = 0
        y_min = 0
        x_max = payload.target_img.shape[0]
        y_max = payload.target_img.shape[1]

        mask_borders = ((pos[:, 0] >= x_min) & (pos[:, 0] <= x_max) &
                        (pos[:, 1] >= y_min) & (pos[:, 1] <= y_max))
        positions = pos[mask_borders]
        intensities = ints[mask_borders]

        mask_img = payload.mask_img
        if mask_img is not None and len(positions) > 0:
            inside = ((positions[:, 0] < mask_img.shape[0]) &
                      (positions[:, 1] < mask_img.shape[1]))
            positions = positions[inside]
            intensities = intensities[inside]
            mask = mask_img[positions[:, 0].astype(int), positions[:, 1].astype(int)] > 0
            positions = positions[mask]
            intensities = intensities[mask]

        return (positions[:, 0] - x_min, positions[:, 1] - y_min), intensities

    def update_heavy(self, payload: VizPayload, ctf_layer=None):
        # Jede Kachel kann fehlen - in der Einzelansicht gibt es nur eine
        if self.im_sim is not None and payload.sim_img is not None:
            self._safe_set_image(self.im_sim, payload.sim_img)

        if self.im_diff is not None and payload.diff_img is not None:
            self._safe_set_image(self.im_diff, payload.diff_img)

        if self.im_tgt is not None and payload.target_img is not None:
            # Artists wiederverwenden statt die Achse zu leeren, damit der
            # Zoom (Toolbar) ueber die Aktualisierungen hinweg stehen bleibt
            ax_target = self.axes_cmap["target"]
            self._apply_target_image(self.im_tgt, ax_target, payload.target_img)
            self._update_target_scatter(self.im_tgt_sc, ax_target, payload)
            ax_target.set_title(f"Target (Step {payload.step})")


        if self.im_ux is not None and payload.strain_maps and payload.strain_maps["ux"] is not None:
            self._safe_set_image(self.im_ux, payload.strain_maps["ux"])

        if self.im_uy is not None and payload.strain_maps and payload.strain_maps["uy"] is not None:
            self._safe_set_image(self.im_uy, payload.strain_maps["uy"])

        if self.im_eps_xx is not None and payload.strain_maps and payload.strain_maps["eps_xx"] is not None:
            self._safe_set_image(self.im_eps_xx, payload.strain_maps["eps_xx"])

        if self.im_eps_yy is not None and payload.strain_maps and payload.strain_maps["eps_yy"] is not None:
            self._safe_set_image(self.im_eps_yy, payload.strain_maps["eps_yy"])

        if self.im_eps_xy is not None and payload.strain_maps and payload.strain_maps["eps_xy"] is not None:
            self._safe_set_image(self.im_eps_xy, payload.strain_maps["eps_xy"])

        if self.im_ctf is not None and payload.ctf_phase is not None:
            self.im_ctf.set_data(payload.ctf_phase)

        if self.im_ctf is not None and self.stem and payload.probe_img is not None:
            probe = payload.probe_img
            self.im_ctf.set_data(probe)
            self.im_ctf.set_extent((-0.5, probe.shape[1] - 0.5, -0.5, probe.shape[0] - 0.5))
            self.im_ctf.set_clim(float(probe.min()), float(probe.max()) or 1e-12)
            ax = self.axes_cmap.get("ctf")
            if ax is not None:
                ax.set_xlim(-0.5, probe.shape[1] - 0.5)
                ax.set_ylim(-0.5, probe.shape[0] - 0.5)

        if self.im_ring is not None and payload.ring_mask is not None:
            self.im_ring.set_data(payload.ring_mask)


        if payload.strain_group is not None:
            for key in ("ux", "uy"):
                if key in self.axes_cmap:
                    self.axes_cmap[key].set_title(f"{key} ({payload.strain_group})")

        if payload.pot_positions is not None and payload.pot_intensities is not None:
            if SHOW_POTENTIAL:
                for i, sc in enumerate(self.pot_scatters):
                    if sc is None or payload.potential_maps is None or i >= len(payload.potential_maps):
                        continue
                    self._safe_set_image(sc, payload.potential_maps[i])
            else:
                for i, sc in enumerate(self.pot_scatters):
                    if sc is None:
                        continue
                    pos = payload.pot_positions[i]
                    ints = payload.pot_intensities[i]

                    if pos.size == 0:
                        sc.set_offsets(np.empty((0, 2), dtype=np.float32))
                        sc.set_array(np.array([], dtype=np.float32))
                        continue

                    valid_mask = np.isfinite(ints)
                    pos = pos[valid_mask]
                    ints = ints[valid_mask]

                    if len(pos) == 0:
                        sc.set_offsets(np.empty((0, 2), dtype=np.float32))
                        sc.set_array(np.array([], dtype=np.float32))
                        continue

                    sc.set_offsets(pos)
                    sc.set_array(ints)

                    imin = float(np.nanmin(ints))
                    imax = float(np.nanmax(ints))
                    if imin == imax:
                        imax = imin + 1e-12
                    sc.set_clim(imin, imax)

                    xmin, ymin = np.min(pos[:, 0]), np.min(pos[:, 1])
                    xmax, ymax = np.max(pos[:, 0]), np.max(pos[:, 1])

                    if xmin == xmax:
                        xmax = xmin + 1e-6
                    if ymin == ymax:
                        ymax = ymin + 1e-6

                    self.axes_cmap["pot" + str(i)].set_xlim(xmin, xmax)
                    self.axes_cmap["pot" + str(i)].set_ylim(ymin, ymax)


        if ("hist" in self.axes_cmap and payload.hist_counts is not None
                and payload.hist_edges is not None):
            centers = 0.5 * (payload.hist_edges[:-1] + payload.hist_edges[1:])
            ymax = 1.0
            for i, counts in enumerate(payload.hist_counts):
                if i >= len(self.hist_lines):
                    break
                self.hist_lines[i].set_data(centers, counts)
                if len(counts) > 0:
                    ymax = max(ymax, float(np.max(counts)))

            self.axes_cmap["hist"].set_xlim(float(centers[0]), float(centers[-1]))
            self.axes_cmap["hist"].set_ylim(0, ymax * 1.05)

        c10 = None
        c30 = None
        if ctf_layer is not None:
            aberrations = ctf_layer.tunable_parameters.aberrations
            try:
                c10 = float(aberrations["C10"].detach().cpu())
            except Exception:
                pass
            try:
                c30 = float(aberrations["C30"].detach().cpu())
            except Exception:
                pass

        if "ctf" not in self.axes_cmap:
            return

        if c10 is not None or c30 is not None:
            self.axes_cmap["ctf"].set_title(
                f"{'Probe' if self.stem else 'CTF'} (Step {payload.step})\n"
                f"C10: {c10 if c10 is not None else np.nan:.2f} nm   "
                f"C30: {c30 if c30 is not None else np.nan:.2f} um"
            )
        else:
            self.axes_cmap["ctf"].set_title(f"{'Probe' if self.stem else 'CTF'} (Step {payload.step})")


class LiveTrainWindow(QtWidgets.QMainWindow):
    request_heavy_refresh_signal = QtCore.Signal()
    show_positions_signal = QtCore.Signal(bool)


    def __init__(
        self,
        model,
        target,
        pot_layer,
        ctf_layer,
        train_method,
        plot_every=5,
        heavy_every=10,
        max_steps=2_000_000,
        parent=None,
        seeder=None,
        seeding=None,
    ):
        super().__init__(parent)
        self.updating_gui = False
        self.seeder = seeder
        self.seeding = dict(seeding or {})
        self._seeding_view = seeder is not None
        self.setWindowTitle("Live Training Qt")

        self.model = model
        self.optimizer = model.build_optimizer()
        self.target = target
        self.pot_layer = pot_layer
        self.ctf_layer = ctf_layer
        self.train_method = train_method

        self.plot_every = plot_every
        self.heavy_every = heavy_every
        self.max_steps = max_steps

        self.thread = None
        self.worker = None

        self._build_ui()
        self._connect_actions()
        self._apply_trainable_selection()

        if hasattr(self.train_method, "train_small_area_squares"):
            self.canvas.set_target_rectangles(self.train_method.train_small_area_squares)

        self._heavy_refresh_running = False
        self._heavy_refresh_pending = False




    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)

        # Splitter statt fester Aufteilung: die rechte Spalte mit den
        # Lernraten laesst sich damit breit ziehen, wenn man sie braucht
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter)

        left_panel = QtWidgets.QWidget()
        right_panel = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_panel)
        right = QtWidgets.QVBoxLayout(right_panel)
        left.setContentsMargins(0, 0, 0, 0)
        right.setContentsMargins(0, 0, 0, 0)

        # die rechte Spalte scrollt als Ganzes, wenn das Fenster fuer alle Boxen zu niedrig ist
        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        right_scroll.setWidget(right_panel)
        self.right_scroll = right_scroll

        splitter.addWidget(left_panel)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        right_scroll.setMinimumWidth(400)


        self.parameter_frame = ParametersFrame(self.model.get_tunable_parameters(),
                                              config=self.model.config)
        self.parameter_frame.parameter_tuned.connect(self._trigger_heavy_refresh)
        left.addWidget(self.parameter_frame, stretch=0)

        self.canvas = MplCanvas(target_tensor=self.target, train_method=self.train_method,
                                config=self.model.config)
        self.canvas.set_seeding(self._seeding_view)
        # Matplotlib-Toolbar: Lupe zum Aufziehen eines Ausschnitts, Hand zum
        # Verschieben, Haus zurueck zur Gesamtansicht
        self.nav_toolbar = NavigationToolbar2QT(self.canvas, self)
        left.addWidget(self.nav_toolbar, stretch=0)
        left.addWidget(self.canvas, stretch=1)



        status_box = QtWidgets.QGroupBox("Status")
        status_layout = QtWidgets.QFormLayout(status_box)

        self.status_label = QtWidgets.QLabel("idle")
        self.step_label = QtWidgets.QLabel("step: -")
        self.loss_label = QtWidgets.QLabel("loss: -")

        status_layout.addRow("State", self.status_label)
        status_layout.addRow("Step", self.step_label)
        status_layout.addRow("Loss", self.loss_label)

        right.addWidget(status_box)

        control_box = QtWidgets.QGroupBox("Training control")
        control_layout = QtWidgets.QGridLayout(control_box)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.resume_btn = QtWidgets.QPushButton("Resume")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.seed_btn = QtWidgets.QPushButton("Start Seeding")
        self.seed_btn.setToolTip(
            "Atome zufaellig einfuegen und behalten, wenn die Korrelation besser wird.\n"
            "Pause/Resume/Stop gelten auch hierfuer.")
        self.seed_btn.setVisible(self.seeder is not None)

        self.single_tile_combo = QtWidgets.QComboBox()
        self.single_tile_combo.setToolTip(
            "Eine Kachel gross zeigen - Simulation, Diff, Target, CTF, Potential ...\n"
            "Zoomen und Verschieben mit der Toolbar unter dem Bild.")
        self._fill_tile_combo()
        self.show_atoms_on_target_checkbox = QtWidgets.QCheckBox("Show atom positions on target")
        self.show_atoms_on_target_checkbox.setChecked(True)
        self.reset_occupancies_btn = QtWidgets.QPushButton("Reset occupancies (aktive Gruppe)")
        self.reset_intensities_btn = QtWidgets.QPushButton("Reset intensities (aktive Gruppe)")
        self.reset_intensities_btn.setToolTip(
            "Intensitaeten zurueck auf die Startwerte aus der Config -\n"
            "nicht auf 0, die Intensitaet ist die Ordnungszahl.")
        self.reset_free_positions_ws2_btn = QtWidgets.QPushButton("Reset free positions (aktive Gruppe)")
        self.render_btn = QtWidgets.QPushButton("Render")
        self.reset_losses = QtWidgets.QPushButton("Reset losses")
        self.reset_zoom_btn = QtWidgets.QPushButton("Zoom zuruecksetzen")


        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        control_layout.addWidget(self.start_btn, 0, 0)
        control_layout.addWidget(self.pause_btn, 1, 0)
        control_layout.addWidget(self.resume_btn, 2, 0)
        control_layout.addWidget(self.stop_btn, 3, 0)
        control_layout.addWidget(self.render_btn, 4, 0)
        control_layout.addWidget(self.seed_btn, 5, 0)


        control_layout.addWidget(self.single_tile_combo, 0, 1)
        control_layout.addWidget(self.show_atoms_on_target_checkbox, 1, 1)
        control_layout.addWidget(self.reset_occupancies_btn, 2, 1)
        control_layout.addWidget(self.reset_intensities_btn, 3, 1)
        control_layout.addWidget(self.reset_free_positions_ws2_btn, 4, 1)
        control_layout.addWidget(self.reset_losses, 5, 1)
        control_layout.addWidget(self.reset_zoom_btn, 6, 1)


        right.addWidget(control_box)

        self._build_seeding_box(right)

        quick_box = QtWidgets.QGroupBox("Quick presets")
        quick_layout = QtWidgets.QHBoxLayout(quick_box)

        self.group_preset_buttons = {}
        for name in self.model.config.group_names:
            btn = QtWidgets.QPushButton(f"nur {name}")
            btn.clicked.connect(lambda _=False, n=name: self._train_only_group(n))
            quick_layout.addWidget(btn)
            self.group_preset_buttons[name] = btn

        self.free_positions_btn = QtWidgets.QPushButton("freie Positionen")
        quick_layout.addWidget(self.free_positions_btn)

        right.addWidget(quick_box)

        strain_box = QtWidgets.QGroupBox("Strain-Kontrollgitter")
        strain_layout = QtWidgets.QFormLayout(strain_box)
        strain_box.setToolTip(
            "Zahl der Kontrollpunkte je Gruppe. Beim Uebernehmen wird das\n"
            "vorhandene Feld interpoliert - Verfeinern kostet fast nichts,\n"
            "Vergroebern verliert Struktur.")

        self.control_point_spins = {}
        for group in self.model.config.groups:
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            current = self.pot_layer.control_points(group.name)
            spins = []
            for index, value in enumerate(current):
                spin = QtWidgets.QSpinBox()
                spin.setRange(2, 200)
                spin.setValue(int(value))
                row_layout.addWidget(spin)
                if index == 0:
                    row_layout.addWidget(QtWidgets.QLabel("x"))
                spins.append(spin)
            self.control_point_spins[group.name] = spins
            strain_layout.addRow(group.name, row)

        self.apply_control_points_btn = QtWidgets.QPushButton("uebernehmen (interpoliert)")
        strain_layout.addRow("", self.apply_control_points_btn)
        right.addWidget(strain_box)

        self.editor = TrainMethodEditor(self.train_method, config=self.model.config)
        editor_scroll = QtWidgets.QScrollArea()
        editor_scroll.setWidgetResizable(True)
        editor_scroll.setWidget(self.editor)
        # stretch=1 und KEIN addStretch danach: die Lernraten bekommen die
        # ganze restliche Hoehe statt eines 250-px-Streifens
        editor_scroll.setMinimumHeight(260)
        right.addWidget(editor_scroll, stretch=1)

        # Kein Preset mehr erzwingen - die geladene TrainMethod gilt.
        self._apply_trainable_selection()



    def _fill_tile_combo(self):
        """Kachelauswahl neu fuellen; die gewaehlte Kachel bleibt, wenn es sie noch gibt."""

        combo = self.single_tile_combo
        current = combo.currentData() if combo.count() else None
        titles = self.canvas.tile_titles()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("alle Kacheln", None)
        for key in self.canvas.all_tile_keys():
            combo.addItem(titles.get(key, key), key)
        index = combo.findData(current)
        combo.setCurrentIndex(max(index, 0))
        combo.blockSignals(False)

    def _set_view(self, seeding):
        """Seeding-Ansicht (ohne Strain-Karten) oder Trainings-Ansicht."""

        self._seeding_view = bool(seeding) and self.seeder is not None
        if self.canvas.set_seeding(self._seeding_view):
            self._fill_tile_combo()
            if hasattr(self.train_method, "train_small_area_squares"):
                self.canvas.set_target_rectangles(self.train_method.train_small_area_squares)
            self._trigger_heavy_refresh()

    #: (Schluessel, Beschriftung, Minimum, Maximum, Schritt, Nachkommastellen, Tooltip)
    SEEDING_FIELDS = (
        ("touch_A", "Entfernen bis [A]", 0.0, 10.0, 0.05, 2,
         "Atome, die naeher als das am neuen Atom liegen, werden entfernt."),
        ("merge_A", "Verschieben bis [A]", 0.0, 10.0, 0.05, 2,
         "Zwischen 'Entfernen bis' und diesem Radius wird nicht neu eingefuegt:\n"
         "das Nachbaratom wandert in die Mitte zum neuen Punkt."),
        ("move_prob", "P(Atom verschieben)", 0.0, 1.0, 0.05, 2,
         "Anteil der Versuche, ein vorhandenes Atom zu verschieben statt eins einzufuegen."),
        ("remove_prob", "P(Atom entfernen)", 0.0, 1.0, 0.05, 2,
         "Anteil der Versuche, ein vorhandenes Atom wegzunehmen."),
        ("move_sigma_A", "Verschiebung sigma [A]", 0.0, 10.0, 0.05, 2,
         "Streuung der Gauss-Verschiebung eines Atoms."),
        ("gradient_temperature", "Gradient Temperatur", 0.01, 5.0, 0.05, 2,
         "Klein: fast immer die beste Stelle der Landschaft; gross: breiter gestreut.\n"
         "Einheit: Anteil der groessten erwarteten Loss-Verbesserung."),
        ("bond_A", "Bindungslaenge [A]", 0.0, 10.0, 0.05, 2,
         "Wahrscheinlichster Abstand zum Nachbaratom (Gauss-Maximum)."),
        ("bond_sigma_A", "Bindung sigma [A]", 0.01, 5.0, 0.05, 2,
         "Breite der Gauss-Verteilung um die Bindungslaenge."),
        ("neighbor_min_A", "Nachbar min. Abstand [A]", 0.0, 10.0, 0.05, 2, "Harter Cutoff nach unten."),
        ("neighbor_max_A", "Nachbar max. Abstand [A]", 0.0, 10.0, 0.05, 2, "Harter Cutoff nach oben."),
        ("max_atoms", "max. Atome (0 = frei)", 0, 1_000_000, 100, 0,
         "Seeding endet, sobald so viele Atome gesetzt sind."),
        ("max_trials", "max. Versuche", 1, 100_000_000, 1000, 0, ""),
        ("tune_every", "Training alle N Versuche (0 = aus)", 0, 1_000_000, 100, 0,
         "Zwischendurch ein paar Gradientenschritte auf Sonde/Mikroskop-Parametern."),
        ("tune_steps", "Schritte je Training", 1, 1000, 1, 0, ""),
    )

    def _seeding_value(self, key):
        seeder = self.seeder
        if key == "max_atoms":
            return seeder.max_atoms or 0
        if key in ("max_trials", "tune_every", "tune_steps"):
            default = {"max_trials": 100_000, "tune_every": 0, "tune_steps": 3}[key]
            alias = "trials" if key == "max_trials" else key
            return self.seeding.get(alias, default)
        return getattr(seeder, key)

    def _build_seeding_box(self, parent_layout):
        self.seeding_spins = {}
        box = QtWidgets.QGroupBox("Seeding-Parameter")
        box.setVisible(self.seeder is not None)
        form = QtWidgets.QFormLayout(box)
        self.seeding_box = box
        if self.seeder is None:
            return
        check = QtWidgets.QCheckBox("Nachbar-Seeding")
        check.setToolTip("Neue Atome der Reihe nach im Bindungsabstand um jedes vorhandene Atom setzen\n"
                         "(Gauss um die Bindungslaenge) statt gleichverteilt ueber das Bild.\n"
                         "Das erste Atom wird zufaellig gesetzt.")
        check.setChecked(self.seeder.neighbor_seeding)
        check.toggled.connect(lambda on: setattr(self.seeder, "neighbor_seeding", bool(on)))
        form.addRow(check)
        self.seeding_neighbor_check = check

        gradient = QtWidgets.QCheckBox("Gradienten-Seeding (nur STEM)")
        gradient.setToolTip("Positionen nach der Loss-Landschaft fuers Einfuegen ziehen (aus dem Gradienten\n"
                            "der Loss, eine Kreuzkorrelation mit dem Atombild) statt gleichverteilt.\n"
                            "Zusammen mit Nachbar-Seeding: die Landschaft im Ring um das jeweilige Atom.\n"
                            "Angenommen wird weiterhin nur nach der exakten lokalen Loss.")
        supported = self.seeder.gradient_supported
        gradient.setEnabled(supported)
        gradient.setChecked(self.seeder.gradient_seeding and supported)
        gradient.toggled.connect(lambda on: setattr(self.seeder, "gradient_seeding", bool(on)))
        form.addRow(gradient)
        self.seeding_gradient_check = gradient
        for key, label, low, high, step, decimals, tip in self.SEEDING_FIELDS:
            if decimals:
                spin = QtWidgets.QDoubleSpinBox()
                spin.setDecimals(decimals)
            else:
                spin = QtWidgets.QSpinBox()
            spin.setRange(low, high)
            spin.setSingleStep(step)
            spin.setValue(self._seeding_value(key))
            if tip:
                spin.setToolTip(tip)
            spin.valueChanged.connect(lambda value, k=key: self._on_seeding_value(k, value))
            form.addRow(label, spin)
            self.seeding_spins[key] = spin
        parent_layout.addWidget(box)

    def _on_seeding_value(self, key, value):
        """Wirkt sofort, auch auf einen laufenden Worker."""

        worker = self.worker if isinstance(self.worker, SeedingWorker) else None
        if key == "max_atoms":
            self.seeder.max_atoms = int(value) or None
        elif key == "max_trials":
            self.seeding["trials"] = int(value)
            if worker is not None:
                worker.max_trials = int(value)
        elif key in ("tune_every", "tune_steps"):
            self.seeding[key] = int(value)
            if worker is not None:
                setattr(worker, key, int(value))
        else:
            setattr(self.seeder, key, float(value))

    def _current_transform_group(self):
        return self.train_method.active_group

    def _active_layer_indices(self):
        """Schicht-Indizes der aktiven Gruppe (ohne aktive Gruppe: alle)."""

        active = self.train_method.active_group
        if active is None:
            return list(range(len(self.model.config.layers)))
        return self.model.config.layer_indices(active)

    def _on_show_positions_changed(self, checked: bool):
        #self.show_positions_checkbox.setChecked(checked)
        self.canvas.show_positions_on_target = checked
        self._trigger_heavy_refresh()



    @QtCore.Slot()
    def on_parameter_update(self, params: ParticlesTunableParameters):
        self.parameter_frame.refresh_from_params(self.worker.get_step(), self.worker.losses[-1])



    def _connect_actions(self):
        self.start_btn.clicked.connect(self.start_training)
        self.seed_btn.clicked.connect(self.start_seeding)
        self.pause_btn.clicked.connect(self.pause_training)
        self.resume_btn.clicked.connect(self.resume_training)
        self.stop_btn.clicked.connect(self.stop_training)

        self.free_positions_btn.clicked.connect(self._free_positions_training)

        self.editor.changed.connect(self._apply_trainable_selection)

        self.single_tile_combo.currentIndexChanged.connect(self._on_single_tile_changed)
        self.show_atoms_on_target_checkbox.toggled.connect(self._on_show_positions_changed)
        self.reset_occupancies_btn.clicked.connect(self.reset_occupancies_ws2)
        self.apply_control_points_btn.clicked.connect(self.apply_control_points)
        self.reset_intensities_btn.clicked.connect(self.reset_intensities)
        self.reset_free_positions_ws2_btn.clicked.connect(self.reset_free_positions_ws2)

        self.render_btn.clicked.connect(self._trigger_heavy_refresh)
        self.reset_losses.clicked.connect(self._reset_losses)
        self.reset_zoom_btn.clicked.connect(self.canvas.reset_zoom)

    def _on_single_tile_changed(self, _index):
        """Einzelansicht umschalten und die neuen Artists gleich fuellen."""

        self.canvas.set_single_tile(self.single_tile_combo.currentData())
        if hasattr(self.train_method, "train_small_area_squares"):
            self.canvas.set_target_rectangles(self.train_method.train_small_area_squares)
        self._trigger_heavy_refresh()

    def _reset_losses(self):
        self.worker.reset_losses()

    def reset_occupancies_ws2(self):
        self.parameter_frame.reset_occupancies(self._active_layer_indices())
        self._trigger_heavy_refresh()

    def apply_control_points(self):
        """Kontrollgitter im laufenden Betrieb umstellen.

        Das Feld wird interpoliert, danach muss der Optimizer neu gebaut
        werden - sein Adam-Zustand hat noch die alte Form. Laeuft gerade ein
        Training, wird dafuer kurz pausiert.
        """

        was_running = self.worker is not None and not self.worker._pause
        if was_running:
            self.pause_training()

        notes = []
        for name, spins in self.control_point_spins.items():
            shape = tuple(spin.value() for spin in spins)
            if shape == tuple(self.pot_layer.control_points(name)):
                continue
            before = self.pot_layer.control_points(name)
            deviation = self.pot_layer.set_control_points(name, shape)
            notes.append(f"{name}: {before[0]}x{before[1]} -> {shape[0]}x{shape[1]} "
                         f"(Feld aendert sich um {deviation:.3g} px)")

        if not notes:
            print("[strain] Kontrollgitter unveraendert")
            if was_running:
                self.resume_training()
            return

        self.optimizer = self.model.build_optimizer()
        if self.worker is not None:
            self.worker.optimizer = self.optimizer
        self.parameter_frame.refresh_from_params()
        print("[strain] " + "; ".join(notes))
        self._trigger_heavy_refresh()

        if was_running:
            self.resume_training()

    def reset_intensities(self):
        """Startwerte holt das Potential selbst - dort steht der Bezug fuer
        Gewicht = (I / I_start) * sigmoid(occ)."""

        start_values = {}
        for index in self._active_layer_indices():
            try:
                start_values[index] = self.pot_layer.get_init_intensities(index)
            except AttributeError:
                pass
        if not start_values:
            print("[reset] keine Startintensitaeten gefunden")
            return
        self.parameter_frame.reset_intensities(start_values)
        self._trigger_heavy_refresh()

    def reset_free_positions_ws2(self):
        self.parameter_frame.reset_free_positions(self._active_layer_indices())
        self._trigger_heavy_refresh()


    def _apply_trainable_selection(self):
        if hasattr(self.pot_layer, "apply_trainable_selection"):
            self.pot_layer.apply_trainable_selection()

    def _train_only_group(self, group_name):
        self.editor.train_only_group(group_name)
        self._apply_trainable_selection()

    def _free_positions_training(self):
        self.editor.free_positions_training()
        self._apply_trainable_selection()

    def _prepare_training_after_seeding(self):
        """Nach dem Seeding sollen die gesetzten Atome frei beweglich sein."""

        tm = self.train_method
        layers = [tm.layer(name) for name in self.pot_layer.config.layer_names]
        if not any(layer.free_positions.enabled for layer in layers):
            for layer in layers:
                layer.free_positions.enabled = True
                layer.intensities.enabled = True
            if hasattr(self.editor, "refresh_from_object"):
                self.editor.refresh_from_object()
        self._apply_trainable_selection()
        # das Seeding hat Parameter direkt beschrieben - alte Adam-Momente verwerfen
        self.optimizer = self.model.build_optimizer()
        self._seeded = False

    def start_training(self):
        if self.thread is not None and self.thread.isRunning():
            return
        if getattr(self, "_seeded", False):
            self._prepare_training_after_seeding()
        self._trained = True
        self._set_view(False)

        worker = TrainingWorker(
            model=self.model,
            optimizer=self.optimizer,
            target=self.target,
            pot_layer=self.pot_layer,
            ctf_layer=self.ctf_layer,
            train_method=self.train_method,
            plot_every=self.plot_every,
            heavy_every=self.heavy_every,
            max_steps=self.max_steps,
        )
        self._launch_worker(worker)

    def start_seeding(self):
        if self.seeder is None or (self.thread is not None and self.thread.isRunning()):
            return

        if getattr(self, "_trained", False):
            self.seeder.resync()
            self._trained = False
        self._seeded = True
        self._set_view(True)

        options = self.seeding
        worker = SeedingWorker(
            self.seeder,
            max_trials=options.get("trials", 100_000),
            tune_every=options.get("tune_every", 0),
            tune_steps=options.get("tune_steps", 3),
            save_callback=options.get("save_callback"),
            target=self.target,
            pot_layer=self.pot_layer,
            ctf_layer=self.ctf_layer,
            train_method=self.train_method,
        )
        self._launch_worker(worker)

    def _launch_worker(self, worker):
        self.thread = QtCore.QThread(self)
        self.worker = worker
        self.worker.moveToThread(self.thread)

        #self.worker.spin_values_changed.connect(self.on_spin_values_changed)
        queued = QtCore.Qt.ConnectionType.QueuedConnection
        self.worker.parameter_update.connect(self.on_parameter_update, queued)

        self.request_heavy_refresh_signal.connect(self.worker.request_heavy_refresh)

        self.thread.started.connect(self.worker.run)
        self.worker.progress_light.connect(self.on_progress_light, queued)
        self.worker.progress_heavy.connect(self.on_progress_heavy, queued)
        self.worker.status_changed.connect(self.on_status_changed, queued)
        self.worker.finished.connect(self.on_worker_finished, queued)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.start_btn.setEnabled(False)
        self.seed_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.thread.start()

    def _do_local_heavy_refresh(self):
        if self._heavy_refresh_running:
            self._heavy_refresh_pending = True
            return

        self._heavy_refresh_running = True


        print("do local refresh")
        try:
            simulated_image, target_image, p_diff, loss = self.model.loss_multislice(
                self.target
            )
            loss_item = float(loss.detach().cpu().item())
            print("BEFORE on_progress_heavy")
            payload = self._build_local_heavy_payload(
                step=0,
                loss_item=loss_item,
                simulated_image=simulated_image,
                target_image=target_image,
                p_diff=p_diff
            )
            self.on_progress_heavy(payload)
            print("AFTER on_progress_heavy")
        except Exception as exc:
            print("local heavy refresh failed:", exc)
        finally:
            self._heavy_refresh_running = False
            if self._heavy_refresh_pending:
                self._heavy_refresh_pending = False
                self._trigger_heavy_refresh()

    def _stem(self):
        return getattr(self.model.config, "imaging_mode", "tem") == "stem_haadf"

    def _strain_maps(self):
        if self._seeding_view:
            return None, None
        return strain_maps_for_active_group(self.pot_layer, self.train_method)

    def _build_local_heavy_payload(self, step, loss_item, simulated_image, target_image, p_diff):
        _, mask_img = self.model.get_current_target_and_mask_img(target_image)
        if mask_img is not None:
            mask_img = mask_img.detach().cpu().numpy()
        sim_img = simulated_image.detach().cpu().numpy()
        tgt_img = target_image.detach().cpu().numpy()
        diff_img = sim_img - tgt_img

        (pot_positions, pot_intensities,
         hist_counts, hist_edges) = collect_layer_visuals(self.pot_layer)

        ctf_phase, ring, probe_img = ctf_visuals(self.ctf_layer, self._stem())

        p_diff_np = None if p_diff is None else p_diff.detach().cpu().numpy()

        strain_maps, strain_group = self._strain_maps()

        pot_imgs = potential_images_for_display(self.pot_layer)

        return VizPayload(
            step=step,
            loss=loss_item,
            losses=np.array([], dtype=np.float32),
            sim_img=sim_img,
            diff_img=diff_img,
            target_img=tgt_img,
            mask_img=mask_img,
            ctf_phase=ctf_phase,
            probe_img=probe_img,
            ring_mask=ring,
            p_diff=p_diff_np,
            pot_positions=pot_positions,
            pot_intensities=pot_intensities,
            hist_counts=hist_counts,
            hist_edges=hist_edges,
            status="preview",
            train_method=self.train_method,
            strain_maps=strain_maps,
            strain_group=strain_group,
            potential_maps=pot_imgs,
            layer_names=self.pot_layer.layer_names,
            layer_groups=[l.group for l in self.pot_layer.layers],
            active_group=self.train_method.active_group,
        )

    def _trigger_heavy_refresh(self):
        print("TRIGGER heavy refresh")
        if self._heavy_refresh_running:
            self._heavy_refresh_pending = True
            return

        if self.worker is not None and not self.worker._pause:
            self.request_heavy_refresh_signal.emit()
        else:
            self._do_local_heavy_refresh()

    def closeEvent(self, event):
        # laufendes Seeding sauber beenden: der Worker speichert beim Stoppen
        if self.thread is not None and self.thread.isRunning():
            self.worker.stop()
            self.thread.quit()
            self.thread.wait(60_000)
        super().closeEvent(event)

    def pause_training(self):
        if self.worker is not None:
            self.worker.pause()
            self.worker.wait_until_parked()
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(True)

    def resume_training(self):
        if self.worker is not None:
            self.worker.resume()
            self.pause_btn.setEnabled(True)
            self.resume_btn.setEnabled(False)

    def stop_training(self):
        if self.worker is not None:
            self.worker.stop()
            self.status_label.setText("stopping...")
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

    @QtCore.Slot(int, float, object)
    def on_progress_light(self, step, loss, losses):
        # if self.updating_gui:
        #     print("Skipping light update because GUI is busy")
        #     return
        #
        # self.updating_gui = True


        self.step_label.setText(f"step: {step}")
        self.loss_label.setText(f"loss: {loss:.6g}")
        # update_light zeichnet nur noch das Loss-Panel neu (kein voller Redraw)
        self.canvas.update_light(step, loss, losses)
        self.canvas.flush_events()
        # self.updating_gui = False

    @QtCore.Slot(object)
    def on_progress_heavy(self, payload: VizPayload):
        if self.updating_gui:
            print("Skipping heavy update because GUI is busy")
            return

        self.updating_gui = True
        self.step_label.setText(f"step: {payload.step}")
        self.loss_label.setText(f"loss: {payload.loss:.6g}")
        self.canvas.update_heavy(payload, ctf_layer=self.ctf_layer)
        self.canvas.draw()
        self.canvas.flush_events()
        self.updating_gui = False

    @QtCore.Slot(str)
    def on_status_changed(self, status):
        self.status_label.setText(status)

    @QtCore.Slot()
    def on_worker_finished(self):
        self.start_btn.setEnabled(True)
        self.seed_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

        self.thread = None


def launch_gui_qt(
    model,
    target,
    pot_layer,
    ctf_layer,
    train_method,
    plot_every=5,
    heavy_every=10,
    max_steps=2_000_000,
    seeder=None,
    seeding=None,
    autostart_seeding=False,
):
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication([])

    window = LiveTrainWindow(
        model=model,
        target=target,
        pot_layer=pot_layer,
        ctf_layer=ctf_layer,
        train_method=train_method,
        plot_every=plot_every,
        heavy_every=heavy_every,
        max_steps=max_steps,
        seeder=seeder,
        seeding=seeding,
    )
    window.resize(1600, 950)
    window.show()
    if seeder is not None and autostart_seeding:
        QtCore.QTimer.singleShot(500, window.start_seeding)

    if owns_app:
        app.exec()

    return window