"""Zufaelliges Atom-fuer-Atom-Seeding als Alternative zum Gitter.

Statt die Atome aus einem Gitter (ImageJ-Plugin) zu starten, wird das Modell
Stueck fuer Stueck aufgebaut - wie der ``ATOM ADDER`` in ``test.py``:

1. Zufallspunkt ueber das Bild ziehen.
2. Kollisionsregel: Atome naeher als ``touch_A`` werden entfernt, Atome
   zwischen ``touch_A`` und ``merge_A`` werden mit dem neuen zusammengelegt
   (das neue wandert auf den Mittelpunkt, das alte verschwindet).
3. Simulieren und Loss gegen das Zielbild berechnen. Besser (oder gleich):
   behalten. Sonst: alles zurueck.

PyTorch kann einen ``nn.Parameter`` nicht in seiner Form aendern, ohne dass
Optimizer-Zustand, Speichern und Laden mitziehen muessen. Deshalb gibt es einen
**Pool fester Groesse**: die Schicht hat von Anfang an ``capacity`` Plaetze,
unbenutzte sind *geparkt* (Position weit ausserhalb des Bildes, Occupancy ~0).
Einfuegen und Entfernen schreiben nur Werte, keine Form aendert sich - Adam,
GUI und Speichern bleiben unberuehrt. Nach dem Seeding trainiert man das Modell
ganz normal weiter.

Der Pool arbeitet in der Identitaets-Geometrie der Gruppe (kein Shift,
Rotation, Scale, Strain): Positionen sind direkt Bildpixel (Zeile, Spalte).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from layers import LayerConfig, LayerGroupConfig, TYPICAL_DEBYE_WALLER_U

#: Position unbenutzter Plaetze: weit ausserhalb des gepolsterten Rasters, die
#: Rasterung verwirft solche Atome (siehe ``rasterize_atoms_bilinear``).
PARK_POSITION = -1.0e4
#: Occupancy-Logit unbenutzter Plaetze (sigmoid ~ 0)
INACTIVE_OCCUPANCY = -100.0

#: Flaechendichte von Graphen, Atome pro nm^2
GRAPHENE_ATOMS_PER_NM2 = 38.2


def graphene_capacity(shape, pixel_size_nm, reserve=1.5):
    """Poolgroesse fuer ein Graphenbild: erwartete Atomzahl mal ``reserve``."""

    area_nm2 = shape[0] * shape[1] * pixel_size_nm ** 2
    return int(math.ceil(area_nm2 * GRAPHENE_ATOMS_PER_NM2 * reserve))


def seed_config(shape, pixel_size_nm, capacity, voltage_kV=80.0, imaging_mode="stem_haadf",
                element="C", elements=(), group_name="Gr", padding=100,
                convergence_mrad=24.81, detector_mrad=(60.0, 200.0), source_size_A=0.3,
                aberrations=None, env_params=None, init_occupancy=100.0):
    """ParticlesConfig mit einer leeren Pool-Schicht (alle Plaetze geparkt)."""

    from ase.data import atomic_numbers
    from ctf import DEFAULT_ABERRATIONS
    from nn import ParticlesConfig

    parked = np.full((capacity, 2), PARK_POSITION, dtype=np.float64)
    atomic = np.full(capacity, float(atomic_numbers[element]), dtype=np.float64)

    layer = LayerConfig(name=group_name, element=element, z_nm=0.0, group=group_name,
                        init_occupancy=init_occupancy, occ_entropy=False,
                        elements=list(elements),
                        debye_waller_u=TYPICAL_DEBYE_WALLER_U.get(element, 0.0))
    group = LayerGroupConfig(name=group_name, control_points=[4, 4])

    kwargs = {}
    if imaging_mode == "stem_haadf":
        kwargs = dict(imaging_mode="stem_haadf", stem_convergence_mrad=float(convergence_mrad),
                      stem_detector_mrad=tuple(detector_mrad),
                      stem_source_size_A=float(source_size_A))

    return ParticlesConfig(
        init_positions=[parked], init_intensities=[atomic],
        shape=tuple(shape), pixel_size_nm=float(pixel_size_nm), voltage_kV=float(voltage_kV),
        init_aberrations=dict(aberrations if aberrations is not None else DEFAULT_ABERRATIONS),
        init_ctf_env_params=dict(env_params or {
            'source_size': 0.1e-3, 'Cc_um': 1.e3, 'deltaE': .35,
            'instab_defocus_nm': .01, 'use_detector': False}),
        init_amp_contrast=0.1, ctf_alpha_max_mrad=50., padding=padding,
        layers=[layer], groups=[group], **kwargs)


class AtomPool:
    """Plaetze einer Schicht: setzen, entfernen, zuruecknehmen - ohne Formaenderung."""

    def __init__(self, potential, layer=0):
        self.potential = potential
        self.layer = potential._layer_index(layer)
        tp = potential.tunable_parameters
        self.positions_buffer = potential.get_init_positions(self.layer)
        self.occupancy = tp.occupancies[self.layer]
        self.shifts = tp.position_shifts[self.layer]
        self.capacity = self.positions_buffer.shape[0]
        self.device = self.positions_buffer.device
        self.active_logit = float(potential.layers[self.layer].init_occupancy)
        self.sync()

    def sync(self):
        """Zustand aus dem Modell lesen (nach Training, Laden, ...)."""

        with torch.no_grad():
            positions = (self.positions_buffer + self.shifts).detach().cpu().numpy().astype(np.float64)
            occupied = (torch.sigmoid(self.occupancy) > 0.5).detach().cpu().numpy()
        parked = (positions < PARK_POSITION / 2).any(axis=1)
        self.active = occupied & ~parked
        self.positions = positions
        # nicht belegte Plaetze sauber parken, damit sie auch ohne Parameter-
        # datei nichts zum Bild beitragen
        stale = np.flatnonzero(~self.active & ~parked)
        stale = np.union1d(stale, np.flatnonzero(parked & occupied))
        if len(stale):
            self.set_slots(stale, np.full((len(stale), 2), PARK_POSITION), False)

    @property
    def n_active(self):
        return int(self.active.sum())

    def active_positions(self):
        return self.positions[self.active]

    def free_slot(self) -> Optional[int]:
        free = np.flatnonzero(~self.active)
        return int(free[0]) if len(free) else None

    def set_slots(self, slots, positions, active):
        """Plaetze setzen. ``active`` darf ein bool oder eine Liste sein."""

        slots = np.asarray(slots, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.float64).reshape(len(slots), 2)
        active = np.broadcast_to(np.asarray(active, dtype=bool), (len(slots),))

        idx = torch.as_tensor(slots, device=self.device)
        pos_t = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        occ_t = torch.where(torch.as_tensor(active, device=self.device),
                            torch.tensor(self.active_logit, device=self.device),
                            torch.tensor(INACTIVE_OCCUPANCY, device=self.device))
        with torch.no_grad():
            self.positions_buffer[idx] = pos_t
            self.shifts[idx] = 0.0
            self.occupancy[idx] = occ_t

        self.positions[slots] = positions
        self.active[slots] = active

    def snapshot(self, slots):
        slots = np.asarray(slots, dtype=np.int64)
        return slots, self.positions[slots].copy(), self.active[slots].copy()

    def restore(self, snapshot):
        slots, positions, active = snapshot
        self.set_slots(slots, positions, active)


def _join_snapshots(first, second):
    return tuple(np.concatenate([a, b]) for a, b in zip(first, second))


@dataclass
class SeedStats:
    trials: int = 0
    accepted: int = 0
    merged: int = 0
    removed: int = 0
    full: bool = False
    history: list = field(default_factory=list)


class RandomAtomSeeder:
    """Baut ein Modell durch zufaelliges Einfuegen auf (siehe Moduldoku).

    ``loss``: ``"model"`` nimmt die im Modell eingestellte Loss
    (``TrainMethod.loss_*``), ``"ncc"`` die negative Korrelation wie im
    urspruenglichen Skript.
    """

    def __init__(self, model, target, layer=0, touch_A=0.5, merge_A=1.0,
                 loss="model", margin_px=0.0, move_prob=0.0, remove_prob=0.0,
                 move_sigma_A=0.7, seed=None):
        if loss not in ("model", "ncc"):
            raise ValueError("loss muss 'model' oder 'ncc' sein")
        self.model = model
        self.target = target
        self.loss_kind = loss
        self.rng = np.random.default_rng(seed)
        self.pool = AtomPool(model.potential_layer, layer)

        pixel_A = model.config.pixel_size_nm * 10.0
        self.touch_px = touch_A / pixel_A
        self.merge_px = merge_A / pixel_A
        self.margin_px = float(margin_px)
        self.move_prob = float(move_prob)
        self.remove_prob = float(remove_prob)
        self.move_sigma_px = move_sigma_A / pixel_A
        self.shape = tuple(model.config.shape)

        self.stats = SeedStats()
        self._optimizer = None
        self.loss = self.evaluate()

    # -- Bewertung --------------------------------------------------------------
    def evaluate(self) -> float:
        # ein leeres Bild ist nach der z-Normierung konstant Null und sieht
        # damit "gut" aus - es darf nie als Vergleichswert dienen
        if self.pool.n_active == 0:
            return math.inf
        with torch.no_grad():
            if self.loss_kind == "ncc":
                value = self._negative_correlation()
            else:
                value = float(self.model.loss_multislice(self.target)[3])
        return value if math.isfinite(value) else math.inf

    def _negative_correlation(self):
        target, mask = self.model.get_current_target_and_mask_img(self.target)
        simulated, _aux = self.model.model(None)
        simulated, target = self.model.normalize_images(simulated, target)
        if mask is not None:
            keep = mask > 0
            simulated, target = simulated[keep], target[keep]
        return -float((simulated * target).mean())

    # -- Vorschlag --------------------------------------------------------------
    def resolve_collisions(self, x, y):
        """Kollisionsregel aus ``test.py``: liefert (neue Position, zu entfernende Plaetze)."""

        slots = np.flatnonzero(self.pool.active)
        coords = self.pool.positions[slots]
        graveyard = []
        merged = False

        for _ in range(len(slots) + 2):
            if len(slots) == 0:
                break
            distance = np.hypot(coords[:, 0] - x, coords[:, 1] - y)
            changed = False

            for n in np.flatnonzero(distance <= self.touch_px):
                if slots[n] not in graveyard:
                    graveyard.append(int(slots[n]))
                    changed = True

            near = np.flatnonzero((distance <= self.merge_px) & (distance > self.touch_px))
            if len(near):
                n = near[0]
                x = coords[n, 0] + (x - coords[n, 0]) / 2.0
                y = coords[n, 1] + (y - coords[n, 1]) / 2.0
                if slots[n] not in graveyard:
                    graveyard.append(int(slots[n]))
                    changed = True
                merged = True

            if not changed:
                break

        return (x, y), graveyard, merged

    def propose(self, exclude=None):
        height, width = self.shape
        x = self.rng.uniform(self.margin_px, height - self.margin_px)
        y = self.rng.uniform(self.margin_px, width - self.margin_px)
        return self.resolve_collisions(x, y)

    # -- Ein Versuch ------------------------------------------------------------
    def trial(self) -> bool:
        """Ein Versuch: meist Einfuegen, optional Verschieben/Entfernen.

        Das Einfuegen ist die Regel aus ``test.py``. Verschieben (Sprung um
        ``move_sigma_px`` oder an eine Zufallsstelle) und Entfernen sind
        Zusatzzuege fuer den Fall, dass reines Einfuegen in einem lokalen
        Minimum haengt (im TEM-Bild passiert das). Mit ``move_prob = remove_prob
        = 0`` (Standard) verhaelt sich der Seeder wie das Skript.
        """

        stats = self.stats
        pool = self.pool
        n = pool.n_active
        draw = self.rng.random()
        if n > 0 and draw < self.remove_prob:
            kind = "remove"
        elif n > 0 and draw < self.remove_prob + self.move_prob:
            kind = "move"
        else:
            kind = "add"

        if kind == "remove":
            stats.trials += 1
            slot = int(self.rng.choice(np.flatnonzero(pool.active)))
            undo = pool.snapshot([slot])
            pool.set_slots([slot], [[PARK_POSITION, PARK_POSITION]], False)
            return self._accept_or_undo(undo, 1, 0, False)

        if kind == "move":
            stats.trials += 1
            slot = int(self.rng.choice(np.flatnonzero(pool.active)))
            undo_self = pool.snapshot([slot])
            old = pool.positions[slot].copy()
            pool.set_slots([slot], [[PARK_POSITION, PARK_POSITION]], False)
            if self.rng.random() < 0.5:
                (x, y), graveyard, merged = self.resolve_collisions(
                    *(old + self.rng.normal(0.0, self.move_sigma_px, 2)))
            else:
                (x, y), graveyard, merged = self.propose()
            undo = _join_snapshots(undo_self, pool.snapshot(graveyard))
        else:
            slot = pool.free_slot()
            if slot is None:
                stats.full = True
                return False
            stats.trials += 1
            (x, y), graveyard, merged = self.propose()
            undo = pool.snapshot([slot] + graveyard)

        if graveyard:
            pool.set_slots(graveyard, np.full((len(graveyard), 2), PARK_POSITION), False)
        pool.set_slots([slot], [[x, y]], True)
        return self._accept_or_undo(undo, len(graveyard), 0, merged)

    def _accept_or_undo(self, undo, removed, _unused, merged):
        new_loss = self.evaluate()
        if new_loss <= self.loss:
            self.loss = new_loss
            self.stats.accepted += 1
            self.stats.removed += removed
            self.stats.merged += int(merged)
            return True
        self.pool.restore(undo)
        return False

    # -- Mikroskopparameter nachziehen ---------------------------------------------
    def tune(self, steps):
        """Ein paar Adam-Schritte auf den freigegebenen Mikroskopparametern.

        Die Atome selbst bleiben dabei unberuehrt (``free_positions`` etc.
        sollten im Seeding aus sein).
        """

        if steps <= 0:
            return
        if self._optimizer is None:
            self._optimizer = self.model.build_optimizer()

        pot = self.model.potential_layer
        with torch.enable_grad():
            for _ in range(steps):
                self._optimizer.zero_grad(set_to_none=True)
                loss = self.model.loss_multislice(self.target)[3]
                if not loss.requires_grad:
                    break
                loss.backward()
                pot.zero_grads_for_frozen()
                self._optimizer.step()
        self.loss = self.evaluate()

    # -- Schleife -----------------------------------------------------------------
    def run(self, trials, tune_every=0, tune_steps=0, log_every=200,
            save_every=0, on_save: Optional[Callable] = None,
            callback: Optional[Callable] = None, should_stop: Optional[Callable] = None):
        """``trials`` Versuche; liefert die Statistik.

        ``tune_every``: alle so viele Versuche ``tune_steps`` Adam-Schritte auf
        Aberrationen/Normierung. ``callback(seeder)`` wird bei jedem Log
        aufgerufen, ``on_save(seeder)`` bei jedem Speichern.
        """

        start = time.time()
        first = self.stats.trials
        for _ in range(trials):
            if should_stop is not None and should_stop():
                break
            self.trial()
            done = self.stats.trials - first

            if self.stats.full:
                print("[seed] Pool voll - Kapazitaet erhoehen")
                break
            if tune_every and tune_steps and done and done % tune_every == 0:
                self.tune(tune_steps)
            if log_every and done and done % log_every == 0:
                self.stats.history.append((self.stats.trials, self.pool.n_active, self.loss))
                rate = done / max(time.time() - start, 1e-9)
                print(f"[seed] Versuch {self.stats.trials}: {self.pool.n_active} Atome, "
                      f"Loss {self.loss:.5f}, angenommen {self.stats.accepted}, "
                      f"{rate:.0f} Versuche/s", flush=True)
                if callback is not None:
                    callback(self)
            if save_every and on_save is not None and done and done % save_every == 0:
                on_save(self)
        return self.stats

    # -- Ergebnis -----------------------------------------------------------------
    def positions_px(self):
        return self.pool.active_positions()

    def save(self, directory, name="seeded"):
        """config.json (nur die aktiven Atome) und .xyz ablegen.

        Die ``config.json`` ist ein normales Modell mit festen Atomen - sie
        laesst sich wie jede andere Config laden und weitertrainieren.
        """

        os.makedirs(directory, exist_ok=True)
        config = self.model.config
        layer = self.pool.layer
        active = self.pool.active
        positions = self.pool.positions[active]

        original_positions = config.init_positions
        original_intensities = config.init_intensities
        try:
            config.init_positions = [
                positions if i == layer else p for i, p in enumerate(original_positions)]
            config.init_intensities = [
                original_intensities[i][active] if i == layer else v
                for i, v in enumerate(original_intensities)]
            from mains.main import serialize
            with open(os.path.join(directory, f"{name}_config.json"), "w") as f:
                json.dump(serialize(config), f)
        finally:
            config.init_positions = original_positions
            config.init_intensities = original_intensities

        pixel_A = config.pixel_size_nm * 10.0
        try:
            from ase import Atoms
            from ase.io import write
            xyz = np.zeros((len(positions), 3))
            xyz[:, 0] = positions[:, 0] * pixel_A
            xyz[:, 1] = positions[:, 1] * pixel_A
            write(os.path.join(directory, f"{name}.xyz"),
                  Atoms(config.layers[layer].element + str(len(positions)), positions=xyz))
        except ImportError:
            pass
        np.save(os.path.join(directory, f"{name}_positions_px.npy"), positions)
        return positions
