"""Zufaelliges Atom-fuer-Atom-Seeding als Alternative zum Gitter.

Statt die Atome aus einem Gitter (ImageJ-Plugin) zu starten, wird das Modell
Stueck fuer Stueck aufgebaut - wie der ``ATOM ADDER`` in ``test.py``:

1. Zufallspunkt ueber das Bild ziehen.
2. Kollisionsregel: Atome naeher als ``touch_A`` werden entfernt, Atome
   zwischen ``touch_A`` und ``merge_A`` werden mit dem neuen zusammengelegt
   (das neue wandert auf den Mittelpunkt, das alte verschwindet).
3. Loss gegen das Zielbild berechnen. Besser (oder gleich): behalten.
   Sonst: alles zurueck.

Die Atom-Tensoren der Schicht (Positionen, Intensitaeten, Occupancies, Shifts)
**wachsen und schrumpfen** mit den angenommenen Versuchen (``AtomStore``).
Das Seeding steht am Anfang, danach beginnt das eigentliche Training - der
Optimizer wird erst dann gebaut, und die Parameter behalten ihre Identitaet.

Bewertet wird, wenn moeglich, **lokal** (``LocalSimulator``): nur das Fenster um
die geaenderten Atome wird neu simuliert und die Loss aus fortlaufenden Summen
aktualisiert. Das ist je nach Bildgroesse und Kernreichweite um ein bis zwei
Groessenordnungen schneller als ein Durchlauf des ganzen Bildes. Mit
``loss="model"`` oder bei Modellen, die der lokale Simulator nicht kennt, wird
wie frueher das ganze Bild simuliert.

Der Seeder arbeitet in der Identitaets-Geometrie der Gruppe (kein Shift,
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
from scipy.ndimage import distance_transform_edt

from layers import LayerConfig, LayerGroupConfig, TYPICAL_DEBYE_WALLER_U
from local_simulation import LocalSimulator

#: Occupancy unterhalb derer ein Atom beim Einlesen als nicht vorhanden gilt
OCCUPIED_THRESHOLD = 0.5
#: Positionen weit ausserhalb des Bildes (Altlast aus Pool-Configs) zaehlen als nicht vorhanden
PARK_POSITION = -1.0e4


def seed_config(shape, pixel_size_nm, voltage_kV=80.0, imaging_mode="stem_haadf",
                element="C", elements=(), group_name="Gr", padding=100,
                convergence_mrad=24.81, detector_mrad=(60.0, 200.0), source_size_A=0.3,
                aberrations=None, env_params=None, init_occupancy=100.0,
                probe_aberrations=None, amp_contrast=0.1, alpha_max_mrad=50.0,
                mask_file=None, target_file=None, start_positions=None, debye_waller_u=None):
    """ParticlesConfig mit einer Schicht, die leer startet.

    ``start_positions`` (n, 2) belegt sie gleich mit n Atomen (zum Weitermachen
    mit einem frueheren Ergebnis).
    """

    from ase.data import atomic_numbers
    from ctf import DEFAULT_ABERRATIONS
    from nn import ParticlesConfig

    if start_positions is None or len(start_positions) == 0:
        start = np.zeros((0, 2), dtype=np.float64)
    else:
        start = np.asarray(start_positions, dtype=np.float64).reshape(-1, 2)
    atomic = np.full(len(start), float(atomic_numbers[element]), dtype=np.float64)

    layer = LayerConfig(name=group_name, element=element, z_nm=0.0, group=group_name,
                        init_occupancy=init_occupancy, occ_entropy=False,
                        elements=list(elements),
                        debye_waller_u=(TYPICAL_DEBYE_WALLER_U.get(element, 0.0)
                                        if debye_waller_u is None else float(debye_waller_u)))
    group = LayerGroupConfig(name=group_name, control_points=[4, 4],
                             mask_file=mask_file, target_file=target_file)

    kwargs = {}
    if imaging_mode == "stem_haadf":
        kwargs = dict(imaging_mode="stem_haadf", stem_convergence_mrad=float(convergence_mrad),
                      stem_detector_mrad=tuple(detector_mrad),
                      stem_source_size_A=float(source_size_A),
                      init_probe_aberrations=dict(probe_aberrations) if probe_aberrations else None)

    return ParticlesConfig(
        init_positions=[start], init_intensities=[atomic],
        shape=tuple(shape), pixel_size_nm=float(pixel_size_nm), voltage_kV=float(voltage_kV),
        init_aberrations=dict(aberrations if aberrations is not None else DEFAULT_ABERRATIONS),
        init_ctf_env_params=dict(env_params or {
            'source_size': 0.1e-3, 'Cc_um': 1.e3, 'deltaE': .35,
            'instab_defocus_nm': .01, 'use_detector': False}),
        init_amp_contrast=float(amp_contrast), ctf_alpha_max_mrad=float(alpha_max_mrad),
        padding=padding,
        layers=[layer], groups=[group], **kwargs)


class AtomStore:
    """Atome einer Schicht: Zeilen der Modell-Tensoren anhaengen und entfernen.

    Die Tensoren selbst liegen im Modell (``init_positions_i`` und
    ``init_intensities_i`` als Buffer, ``position_shifts``, ``intensities`` und
    ``occupancies`` als Parameter). Zum Wachsen werden sie neu angelegt; die
    ``nn.Parameter``-Objekte bleiben dieselben (nur ``.data`` wechselt).

    ``positions`` (n, 2) und ``weights`` (n,) sind NumPy-Spiegel fuer die
    Nachbarsuche und den lokalen Simulator.
    """

    def __init__(self, potential, layer=0):
        from ase.data import atomic_numbers

        self.potential = potential
        self.layer = potential._layer_index(layer)
        self.device = potential.device
        config = potential.layers[self.layer]
        self.active_logit = float(config.init_occupancy)
        self.atomic_number = float(atomic_numbers[config.element])
        self.new_weight = float(torch.sigmoid(torch.tensor(self.active_logit)))
        self.sync()

    # -- Zugriff auf die Modell-Tensoren ------------------------------------------
    @property
    def _names(self):
        return f"init_positions_{self.layer}", f"init_intensities_{self.layer}"

    @property
    def _parameters(self):
        tp = self.potential.tunable_parameters
        return (tp.position_shifts[self.layer], tp.intensities[self.layer],
                tp.occupancies[self.layer])

    def tensors(self):
        pos, ints = self._names
        shifts, intensities, occupancy = self._parameters
        return (getattr(self.potential, pos), getattr(self.potential, ints),
                shifts.data, intensities.data, occupancy.data)

    def _assign(self, tensors):
        pos, ints = self._names
        setattr(self.potential, pos, tensors[0])
        setattr(self.potential, ints, tensors[1])
        for parameter, value in zip(self._parameters, tensors[2:]):
            parameter.data = value
            parameter.grad = None

    @property
    def trainable(self):
        return any(p.requires_grad for p in self._parameters)

    # -- Zustand ------------------------------------------------------------------
    @property
    def n(self):
        return len(self.positions)

    def sync(self):
        """Zustand aus dem Modell lesen (nach Training, Laden, ...).

        Atome mit Occupancy unter 0.5 gelten als nicht vorhanden und werden
        entfernt.
        """

        self._read_mirror()
        with torch.no_grad():
            occupied = (torch.sigmoid(self.tensors()[4]) > OCCUPIED_THRESHOLD).cpu().numpy()
        parked = (self.positions < PARK_POSITION / 2).any(axis=1)
        keep = occupied & ~parked
        if not keep.all():
            self._rebuild(np.flatnonzero(keep), np.zeros((0, 2)))

    def _read_mirror(self):
        with torch.no_grad():
            init_pos, _i, shifts, _j, _o = self.tensors()
            self.positions = (init_pos + shifts).detach().cpu().numpy().astype(np.float64)
            weights = self.potential.get_atom_weights()[self.layer]
            self.weights = weights.detach().cpu().numpy().astype(np.float64)

    def _rebuild(self, keep, add):
        """Zeilen ``keep`` behalten und ``add`` (m, 2) neue anhaengen; liefert die alten Tensoren."""

        keep = np.asarray(keep, dtype=np.int64)
        add = np.asarray(add, dtype=np.float32).reshape(-1, 2)
        m = len(add)
        old = self.tensors()
        keep_t = torch.as_tensor(keep, device=self.device)
        with torch.no_grad():
            add_t = torch.as_tensor(add, device=self.device)

            def full(value):
                return torch.full((m,), value, dtype=torch.float32, device=self.device)

            self._assign((
                torch.cat([old[0][keep_t], add_t]),
                torch.cat([old[1][keep_t], full(self.atomic_number)]),
                torch.cat([old[2][keep_t], torch.zeros((m, 2), device=self.device)]),
                torch.cat([old[3][keep_t], full(self.atomic_number)]),
                torch.cat([old[4][keep_t], full(self.active_logit)]),
            ))
        self.positions = np.concatenate([self.positions[keep], add.astype(np.float64)])
        self.weights = np.concatenate([self.weights[keep], np.full(m, self.new_weight)])
        return old

    def apply(self, remove, add):
        """Atome ``remove`` (Indizes) entfernen, ``add`` (m, 2) anhaengen; liefert ein Undo-Token."""

        keep = np.ones(self.n, dtype=bool)
        keep[np.asarray(remove, dtype=np.int64)] = False
        before = (self.positions, self.weights)
        tensors = self._rebuild(np.flatnonzero(keep), add)
        return tensors, before

    def undo(self, token):
        tensors, (self.positions, self.weights) = token
        self._assign(tensors)

    def active_positions(self):
        return self.positions.copy()

    def write_config(self):
        """Atome in ``config.init_*`` eintragen (Positionen = Start + Shift, aktuelle Intensitaeten)."""

        config = self.potential.config
        init_pos, _i, shifts, intensities, _o = self.tensors()
        positions = (init_pos + shifts).detach().cpu().numpy().astype(np.float64)
        values = intensities.detach().cpu().numpy().astype(np.float64)
        config.init_positions = [
            positions if i == self.layer else p for i, p in enumerate(config.init_positions)]
        config.init_intensities = [
            values if i == self.layer else v for i, v in enumerate(config.init_intensities)]


@dataclass
class SeedStats:
    trials: int = 0
    accepted: int = 0
    merged: int = 0
    removed: int = 0
    fallbacks: int = 0
    tune_rejected: int = 0
    refreshes: int = 0
    max_drift: float = 0.0
    full: bool = False
    history: list = field(default_factory=list)


class RandomAtomSeeder:
    """Baut ein Modell durch zufaelliges Einfuegen auf (siehe Moduldoku).

    ``loss``: ``"ncc"`` (Standard) ist die negative Korrelation wie im
    urspruenglichen Skript und wird, wenn moeglich, lokal berechnet
    (``local``). ``"model"`` nimmt die im Modell eingestellte Loss
    (``TrainMethod.loss_*``) und simuliert immer das ganze Bild.
    """

    def __init__(self, model, target, layer=0, touch_A=0.5, merge_A=1.0,
                 loss="ncc", margin_px=0.0, move_prob=0.0, remove_prob=0.0,
                 move_sigma_A=0.7, seed=None, region=None, local=True,
                 local_radius_px=None, refresh_every=300, max_atoms=None,
                 gradient_seeding=False, gradient_temperature=0.1,
                 neighbor_seeding=False, bond_A=1.42, bond_sigma_A=0.2,
                 neighbor_min_A=0.5, neighbor_max_A=2.0):
        if loss not in ("model", "ncc"):
            raise ValueError("loss muss 'model' oder 'ncc' sein")
        self.model = model
        self.target = target
        self.loss_kind = loss
        self.rng = np.random.default_rng(seed)
        self.atoms = AtomStore(model.potential_layer, layer)
        self.max_atoms = max_atoms

        pixel_A = model.config.pixel_size_nm * 10.0
        self.touch_px = touch_A / pixel_A
        self.merge_px = merge_A / pixel_A
        self.margin_px = float(margin_px)
        self.move_prob = float(move_prob)
        self.remove_prob = float(remove_prob)
        self.move_sigma_px = move_sigma_A / pixel_A
        self.shape = tuple(model.config.shape)
        # Nachbar-Seeding: statt gleichverteilt wird der Reihe nach um jedes vorhandene
        # Atom ein Punkt gezogen (Abstand ~ abgeschnittene Gauss-Verteilung um die Bindungslaenge)
        self.neighbor_seeding = bool(neighbor_seeding)
        self._neighbor_cursor = 0
        # Gradienten-Seeding (nur STEM): Positionen nach der Loss-Landschaft fuers Einfuegen
        # statt gleichverteilt bzw. nach Gauss; die Landschaft kommt aus dem lokalen Simulator
        self.gradient_seeding = bool(gradient_seeding)
        self.gradient_temperature = float(gradient_temperature)
        self._landscape = None
        self._landscape_version = -1
        self.bond_px = bond_A / pixel_A
        self.bond_sigma_px = bond_sigma_A / pixel_A
        self.neighbor_min_px = neighbor_min_A / pixel_A
        self.neighbor_max_px = neighbor_max_A / pixel_A
        self._region_mask = None

        # Bereich, in dem neue Atome gewuerfelt werden (bool-Maske, Zeile/Spalte in Pixeln)
        self._region_pixels = None
        if region is not None:
            region = np.asarray(region, dtype=bool)
            if region.shape != self.shape:
                raise ValueError(f"region {region.shape} passt nicht zum Bild {self.shape}")
            margin = int(math.ceil(self.margin_px))
            inner = np.zeros_like(region)
            inner[margin:self.shape[0] - margin, margin:self.shape[1] - margin] = True
            pixels = np.argwhere(region & inner)
            if len(pixels) == 0:
                raise ValueError("Die Maske laesst keinen Platz fuer Atome")
            self._region_pixels = pixels
            self._region_mask = region & inner

        self.stats = SeedStats()
        self._optimizer = None
        self._optimizer_version = -1
        self._version = 0
        self._want_local = bool(local) and loss == "ncc"
        self._local_radius = local_radius_px
        self._refresh_every = refresh_every
        self.local = None
        self.local_note = ""
        self.loss = math.inf
        self.resync(sync_store=False)

    # -- Zustand --------------------------------------------------------------
    @property
    def n_atoms(self) -> int:
        return self.atoms.n

    @property
    def pixel_A(self) -> float:
        return self.model.config.pixel_size_nm * 10.0

    # die drei Radien in Angstrom - zur Laufzeit aenderbar (GUI)
    @property
    def touch_A(self) -> float:
        return self.touch_px * self.pixel_A

    @touch_A.setter
    def touch_A(self, value):
        self.touch_px = float(value) / self.pixel_A

    @property
    def merge_A(self) -> float:
        return self.merge_px * self.pixel_A

    @merge_A.setter
    def merge_A(self, value):
        self.merge_px = float(value) / self.pixel_A

    @property
    def move_sigma_A(self) -> float:
        return self.move_sigma_px * self.pixel_A

    @move_sigma_A.setter
    def move_sigma_A(self, value):
        self.move_sigma_px = float(value) / self.pixel_A

    @property
    def bond_A(self) -> float:
        return self.bond_px * self.pixel_A

    @bond_A.setter
    def bond_A(self, value):
        self.bond_px = float(value) / self.pixel_A

    @property
    def bond_sigma_A(self) -> float:
        return self.bond_sigma_px * self.pixel_A

    @bond_sigma_A.setter
    def bond_sigma_A(self, value):
        self.bond_sigma_px = float(value) / self.pixel_A

    @property
    def neighbor_min_A(self) -> float:
        return self.neighbor_min_px * self.pixel_A

    @neighbor_min_A.setter
    def neighbor_min_A(self, value):
        self.neighbor_min_px = float(value) / self.pixel_A

    @property
    def neighbor_max_A(self) -> float:
        return self.neighbor_max_px * self.pixel_A

    @neighbor_max_A.setter
    def neighbor_max_A(self, value):
        self.neighbor_max_px = float(value) / self.pixel_A

    @property
    def gradient_supported(self) -> bool:
        return self.uses_local and self.local.stem

    @property
    def uses_local(self) -> bool:
        return self.local is not None and self.local.ok

    def resync(self, sync_store=True):
        """Nach Aenderungen von aussen (Training, Laden, Parameter) alles neu einlesen."""

        if sync_store:
            self.atoms.sync()
        self._version += 1
        self.local = None
        if self._want_local:
            local = LocalSimulator(self.model, self.target, self.atoms,
                                   radius_px=self._local_radius,
                                   refresh_every=self._refresh_every)
            if local.ok:
                local.refresh()
            self.local_note = local.reason
            self.local = local if local.ok else None
        self.loss = self.local.loss if self.uses_local else self.evaluate()
        return self.loss

    # -- Bewertung --------------------------------------------------------------
    def evaluate(self) -> float:
        """Loss aus einer vollstaendigen Simulation."""

        # ein leeres Bild ist nach der z-Normierung konstant Null und sieht
        # damit "gut" aus - es darf nie als Vergleichswert dienen
        if self.atoms.n == 0:
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
    def resolve_collisions(self, x, y, exclude=None):
        """Kollisionsregel aus ``test.py``: liefert (neue Position, zu entfernende Indizes, merged).

        ``exclude``: Indizes, die nicht mitspielen (z.B. das Atom, das gerade verschoben wird).
        """

        indices = np.arange(self.atoms.n)
        if exclude is not None and len(exclude):
            indices = np.setdiff1d(indices, np.asarray(exclude, dtype=np.int64))
        coords = self.atoms.positions[indices]
        graveyard = []
        merged = False

        for _ in range(len(indices) + 2):
            if len(indices) == 0:
                break
            distance = np.hypot(coords[:, 0] - x, coords[:, 1] - y)
            changed = False

            for n in np.flatnonzero(distance <= self.touch_px):
                if indices[n] not in graveyard:
                    graveyard.append(int(indices[n]))
                    changed = True

            near = np.flatnonzero((distance <= self.merge_px) & (distance > self.touch_px))
            if len(near):
                n = near[0]
                x = coords[n, 0] + (x - coords[n, 0]) / 2.0
                y = coords[n, 1] + (y - coords[n, 1]) / 2.0
                if indices[n] not in graveyard:
                    graveyard.append(int(indices[n]))
                    changed = True
                merged = True

            if not changed:
                break

        return (x, y), graveyard, merged

    def _uniform_point(self):
        height, width = self.shape
        if self._region_pixels is not None:
            row, col = self._region_pixels[self.rng.integers(len(self._region_pixels))]
            return row + self.rng.uniform(0.0, 1.0), col + self.rng.uniform(0.0, 1.0)
        return (self.rng.uniform(self.margin_px, height - self.margin_px),
                self.rng.uniform(self.margin_px, width - self.margin_px))

    def _inside(self, x, y):
        height, width = self.shape
        if not (self.margin_px <= x < height - self.margin_px
                and self.margin_px <= y < width - self.margin_px):
            return False
        return self._region_mask is None or bool(self._region_mask[int(x), int(y)])

    def _bases(self, exclude=None):
        """Die naechsten Atome der Reihe nach (hoechstens ein voller Durchgang)."""

        n = self.atoms.n
        skip = set(int(i) for i in exclude) if exclude is not None else set()
        for _ in range(n):
            index = self._neighbor_cursor % n
            self._neighbor_cursor = index + 1
            if index not in skip:
                yield self.atoms.positions[index]

    def _neighbor_point(self, exclude=None):
        """Punkt im Bindungsabstand um das naechste Atom der Reihe; None, wenn keiner gefunden wird.

        Jeder Aufruf nimmt das naechste Atom (Index fortlaufend, am Ende wieder von vorn;
        neu angenommene Atome stehen hinten und kommen im selben Durchgang dran).
        Mit Gradienten-Seeding kommt der Punkt aus der Loss-Landschaft im Ring um das Atom.
        """

        low, high = self.neighbor_min_px, self.neighbor_max_px
        if self.atoms.n == 0 or high <= low:
            return None
        gain = self._gain_map()
        if gain is not None and not np.isfinite(gain).any():
            return None
        for base in self._bases(exclude):
            if gain is not None:
                point = self._landscape_point(gain, base)
                if point is not None:
                    return point
                continue
            for _ in range(20):
                for _ in range(50):
                    radius = self.rng.normal(self.bond_px, self.bond_sigma_px)
                    if low <= radius <= high:
                        break
                else:
                    radius = min(max(self.bond_px, low), high)
                angle = self.rng.uniform(0.0, 2.0 * math.pi)
                x, y = base[0] + radius * math.cos(angle), base[1] + radius * math.sin(angle)
                if self._inside(x, y):
                    return x, y
        return None

    # -- Gradienten-Seeding ------------------------------------------------------
    def _gain_map(self):
        """Loss-Aenderung fuer ein neues Atom je Pixel; inf, wo es nicht verbessert oder kollidiert.

        None, wenn nicht verfuegbar. Auch wenn nirgends eine Verbesserung uebrig ist, bleibt das
        Ergebnis eine Karte; ``_landscape_point`` liefert dann None und es gilt die Zufallsauswahl.
        """

        if not self.gradient_seeding or not self.gradient_supported or self.atoms.n == 0:
            return None
        if self._landscape_version != self._version or self._landscape is None:
            gain = self.local.insertion_gain(self.atoms.new_weight)
            height, width = self.shape
            if gain is not None:
                valid = np.zeros(self.shape, dtype=bool)
                m = int(math.ceil(self.margin_px))
                valid[m:height - m, m:width - m] = True
                if self._region_mask is not None:
                    valid &= self._region_mask
                # Naehe zu bestehenden Atomen: die Kollisionsregel wuerde sie verschieben/entfernen,
                # die Landschaft gilt dort nicht. Nur echte Verbesserungen kommen in Frage.
                occupied = np.ones(self.shape, dtype=bool)
                positions = np.rint(self.atoms.positions).astype(int)
                occupied[np.clip(positions[:, 0], 0, height - 1), np.clip(positions[:, 1], 0, width - 1)] = False
                free = distance_transform_edt(occupied) > self.merge_px
                gain = np.where(valid & free & (gain < 0.0), gain, np.inf)
            self._landscape = (gain, None)
            self._landscape_version = self._version
        return self._landscape[0]

    def _gain_weights(self, gain):
        """Auswahlgewichte exp(-gain / (T * max|gain|)); Pixel ohne Verbesserung (inf) bekommen 0."""

        finite = np.isfinite(gain)
        if not finite.any():
            return None
        scale = max(float(np.abs(gain[finite]).max()), 1e-30)
        score = np.where(finite, -gain / (scale * max(self.gradient_temperature, 1e-3)), -np.inf)
        score -= score[finite].max()
        return np.exp(score)

    def _landscape_point(self, gain, base=None):
        """Punkt aus der Loss-Landschaft: ganzes Bild (``base`` None) oder Ring um ``base``."""

        if base is None:
            if self._landscape[1] is None:
                weights = self._gain_weights(gain)
                if weights is None:
                    return None
                self._landscape = (gain, np.cumsum(weights.ravel()))
            cdf = self._landscape[1]
            flat = int(np.searchsorted(cdf, self.rng.uniform(0.0, cdf[-1]), side="right"))
            row, col = divmod(min(flat, cdf.size - 1), self.shape[1])
        else:
            height, width = self.shape
            high = int(math.ceil(self.neighbor_max_px))
            r0, r1 = max(int(base[0]) - high, 0), min(int(base[0]) + high + 1, height)
            c0, c1 = max(int(base[1]) - high, 0), min(int(base[1]) + high + 1, width)
            if r1 <= r0 or c1 <= c0:
                return None
            window = gain[r0:r1, c0:c1]
            rows, cols = np.mgrid[r0:r1, c0:c1]
            distance = np.hypot(rows - base[0], cols - base[1])
            ring = (distance >= self.neighbor_min_px) & (distance <= self.neighbor_max_px)
            window = np.where(ring, window, np.inf)
            if not np.isfinite(window).any():
                return None
            # Skala vom ganzen Bild, damit die Temperatur ueberall dasselbe bedeutet
            finite = np.isfinite(gain)
            scale = max(float(np.abs(gain[finite]).max()), 1e-30)
            score = -window / (scale * max(self.gradient_temperature, 1e-3))
            score -= score[np.isfinite(window)].max()
            weights = np.where(np.isfinite(window), np.exp(score), 0.0).ravel()
            cdf = np.cumsum(weights)
            flat = int(np.searchsorted(cdf, self.rng.uniform(0.0, cdf[-1]), side="right"))
            flat = min(flat, cdf.size - 1)
            row, col = r0 + flat // (c1 - c0), c0 + flat % (c1 - c0)
        height, width = self.shape
        x = row + float(np.clip(self.rng.normal(0.0, 0.25), -0.5, 0.5))
        y = col + float(np.clip(self.rng.normal(0.0, 0.25), -0.5, 0.5))
        return (min(max(x, self.margin_px), height - self.margin_px - 1e-6),
                min(max(y, self.margin_px), width - self.margin_px - 1e-6))

    def propose(self, exclude=None):
        point = None
        if self.neighbor_seeding:
            point = self._neighbor_point(exclude)
        else:
            gain = self._gain_map()
            if gain is not None:
                point = self._landscape_point(gain)
        if point is None:
            point = self._uniform_point()
        return self.resolve_collisions(*point, exclude)

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
        atoms = self.atoms
        n = atoms.n
        draw = self.rng.random()
        if n > 0 and draw < self.remove_prob:
            kind = "remove"
        elif n > 0 and draw < self.remove_prob + self.move_prob:
            kind = "move"
        else:
            kind = "add"

        if kind == "add" and self.max_atoms is not None and n >= self.max_atoms:
            stats.full = True
            return False
        stats.trials += 1

        merged = False
        if kind == "remove":
            remove = [int(self.rng.integers(n))]
            add = np.zeros((0, 2))
        elif kind == "move":
            index = int(self.rng.integers(n))
            if self.rng.random() < 0.5:
                target_xy = atoms.positions[index] + self.rng.normal(0.0, self.move_sigma_px, 2)
                xy, graveyard, merged = self.resolve_collisions(*target_xy, exclude=[index])
            else:
                xy, graveyard, merged = self.propose(exclude=[index])
            remove = [index] + graveyard
            add = np.array([xy])
        else:
            xy, graveyard, merged = self.propose()
            remove = graveyard
            add = np.array([xy])
        removed = len(remove) - (1 if kind == "move" else 0)
        return self._accept_or_reject(remove, add, removed, merged)

    def _accept_or_reject(self, remove, add, removed, merged):
        stats = self.stats
        atoms = self.atoms
        local = self.local if self.uses_local else None

        proposal = local.propose(remove, add, atoms.new_weight) if local is not None else None
        if local is not None and proposal is None:
            stats.fallbacks += 1

        if proposal is not None:
            if not (math.isfinite(proposal.loss) and proposal.loss <= self.loss):
                return False
            atoms.apply(remove, add)
            local.commit(proposal)
            self.loss = proposal.loss
            self._version += 1
            self._count_accepted(removed, merged)
            if local.needs_refresh:
                self._refresh_local()
            return True

        token = atoms.apply(remove, add)
        new_loss = self.evaluate()
        if not (math.isfinite(new_loss) and new_loss <= self.loss):
            atoms.undo(token)
            return False
        self.loss = new_loss
        self._version += 1
        self._count_accepted(removed, merged)
        if local is not None:
            self._refresh_local()
        return True

    def _count_accepted(self, removed, merged):
        self.stats.accepted += 1
        self.stats.removed += removed
        self.stats.merged += int(merged)

    def _refresh_local(self):
        """Lokalen Simulator aus einer vollen Simulation neu aufbauen (Drift abbauen)."""

        before = self.loss
        after = self.local.refresh()
        self.stats.refreshes += 1
        if self.local.ok:
            self.stats.max_drift = max(self.stats.max_drift, abs(after - before))
            self.loss = after
        else:
            self.local_note = self.local.reason
            self.local = None
            self.loss = self.evaluate()

    # -- Mikroskopparameter nachziehen ---------------------------------------------
    def tune(self, steps):
        """Ein paar Adam-Schritte auf den freigegebenen Mikroskopparametern.

        Die Atome selbst bleiben dabei unberuehrt (``free_positions`` etc.
        sollten im Seeding aus sein).
        """

        if steps <= 0 or self.atoms.n == 0:
            return
        # Adam-Zustaende haben die alte Atomzahl - bei trainierbaren Atom-Tensoren neu aufbauen
        if self._optimizer is None or (self.atoms.trainable
                                       and self._optimizer_version != self._version):
            self._optimizer = self.model.build_optimizer()
            self._optimizer_version = self._version

        # Adam-Schritte auf dem Modell-Loss koennen die Korrelation verschlechtern:
        # dann werden die Parameter zurueckgesetzt, der Loss steigt nie
        saved = [(p, p.detach().clone()) for group in self._optimizer.param_groups
                 for p in group["params"]]
        before = self.resync(sync_store=False)

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
        self.resync(sync_store=False)
        if not self.loss <= before:
            with torch.no_grad():
                for p, value in saved:
                    p.copy_(value)
            self._optimizer = None
            self.resync(sync_store=False)
            self.stats.tune_rejected += 1

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
                print(f"[seed] {self.max_atoms} Atome erreicht", flush=True)
                break
            if tune_every and tune_steps and done and done % tune_every == 0:
                self.tune(tune_steps)
            if log_every and done and done % log_every == 0:
                self.stats.history.append((self.stats.trials, self.atoms.n, self.loss))
                rate = done / max(time.time() - start, 1e-9)
                print(f"[seed] Versuch {self.stats.trials}: {self.atoms.n} Atome, "
                      f"Loss {self.loss:.5f}, angenommen {self.stats.accepted}, "
                      f"{rate:.0f} Versuche/s", flush=True)
                if callback is not None:
                    callback(self)
            if save_every and on_save is not None and done and done % save_every == 0:
                on_save(self)
        return self.stats

    # -- Ergebnis -----------------------------------------------------------------
    def finish(self):
        """Atome in die Config uebernehmen und den lokalen Simulator freigeben."""

        self.atoms.write_config()
        self.local = None
        self._version += 1

    def positions_px(self):
        return self.atoms.active_positions()

    def save(self, directory, name="seeded", config_file=None):
        """config.json (die Atome als Startwerte) und .xyz ablegen.

        Die ``config.json`` ist ein normales Modell - sie laesst sich wie jede
        andere Config laden und weitertrainieren.
        """

        os.makedirs(directory, exist_ok=True)
        self.atoms.write_config()
        config = self.model.config
        from mains.main import serialize
        with open(os.path.join(directory, config_file or f"{name}_config.json"), "w") as f:
            json.dump(serialize(config), f)

        positions = config.init_positions[self.atoms.layer]
        pixel_A = config.pixel_size_nm * 10.0
        try:
            from ase import Atoms
            from ase.io import write
            xyz = np.zeros((len(positions), 3))
            xyz[:, 0] = positions[:, 0] * pixel_A
            xyz[:, 1] = positions[:, 1] * pixel_A
            write(os.path.join(directory, f"{name}.xyz"),
                  Atoms(config.layers[self.atoms.layer].element + str(len(positions)),
                        positions=xyz))
        except ImportError:
            pass
        np.save(os.path.join(directory, f"{name}_positions_px.npy"), positions)
        return positions
