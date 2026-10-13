r"""Part II, Section 6: random atom-by-atom seeding.

Motivation
==========
The occupancy relaxation of Section 1 can switch atoms off but cannot invent new
ones, and a lattice from a template may not fit a disordered sample at all.
Seeding builds the atom set by *discrete* search.  This is a Markov chain on
atom configurations :math:`\mathcal A` with a greedy (zero-temperature
Metropolis) acceptance: a trial :math:`\mathcal A\to\mathcal A'` is accepted if

.. math::

    \mathcal L(\mathcal A') \le \mathcal L(\mathcal A),
    \qquad (6.10)

and otherwise undone.  The loss never increases, the number of atoms is not
fixed, and the chain is reversible only through the move types below.

Trials
------
``add``
    a point from Section 6.3 is resolved against the present atoms by the
    collision rule of Section 6.2 (atoms that touch it vanish, close atoms are
    merged into it); the proposal is "remove these, add that".
``move``
    an atom is removed and added again, either after a Gaussian jump of width
    :math:`\sigma_{\rm move}` or at a fresh proposal.
``remove``
    a random atom is deleted.
``relax``
    (STEM) a promising site that is blocked by a neighbour :math:`\mathbf q` is
    found from the landscape; :math:`\mathbf q` is taken out, the new atom put in, and
    :math:`\mathbf q` re-inserted at the best site within :math:`r_{\rm relax}` of its old
    place *in the landscape of that intermediate state* (Section 6.4,
    :meth:`LocalLoss.hypothetical`).  Accepted only if the combined change
    satisfies Eq. (6.10).

With ``move_prob = remove_prob = relax_prob = 0`` only atoms are added.  The
relative weight of a new atom is chosen from ``weight_options`` as the one with the
lowest loss.

The loss is either the negative correlation of Eq. (6.2) from running sums
(``"ncc"``, local, Section 6.4) or the complete objective of Section 3
(``"objective"``, full image).  An empty configuration has loss :math:`\infty`: after
standardisation it would be a constant image, which looks deceptively good.

Tuning
------
Atom positions are discrete, the microscope parameters (defocus, ...) continuous.
Every ``tune_every`` trials :meth:`Seeder.tune` takes a few Adam steps of
Section 5 on the enabled global parameters; the result is kept only if
Eq. (6.10) holds for the correlation loss afterwards, otherwise all tensors are
restored (``tune_rejected``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from ...data.schema import Experiment, Group, Imaging, Layer, Stem, Structure, TYPICAL_DEBYE_WALLER_U
from ...physics.window import LocalImage
from ..model import Model
from ..trainer import Trainer
from .collision import resolve_point
from .local_loss import LocalLoss, Trial, negative_correlation
from .proposals import Landscape, NeighbourSampler, Region, distance_map
from .store import AtomStore, SeedingError


@dataclass
class SeedConfig:
    touch_A: float = 0.5
    merge_A: float = 1.0
    #: "ncc" (local running sums) or "objective" (complete loss, full image)
    loss: str = "ncc"
    margin_px: float = 0.0
    move_prob: float = 0.0
    remove_prob: float = 0.0
    relax_prob: float = 0.0
    move_sigma_A: float = 0.7
    relax_A: float = 2.0
    neighbour_seeding: bool = False
    bond_A: float = 1.42
    bond_sigma_A: float = 0.2
    neighbour_min_A: float = 0.5
    neighbour_max_A: float = 2.0
    gradient_seeding: bool = False
    gradient_temperature: float = 0.1
    weight_options: Sequence[float] = (1.0,)
    max_atoms: Optional[int] = None
    local: bool = True
    local_radius_px: Optional[int] = None
    refresh_every: int = 300
    seed: Optional[int] = None

    def __post_init__(self):
        if self.loss not in ("objective", "ncc"):
            raise ValueError("loss must be 'ncc' or 'objective'")
        options = [float(w) for w in self.weight_options]
        if not options or any(not 0.0 < w <= 1.5 for w in options):
            raise ValueError("weight_options must lie in (0, 1.5]")
        self.weight_options = tuple(sorted(set(options), reverse=True))


@dataclass
class SeedStats:
    trials: int = 0
    accepted: int = 0
    merged: int = 0
    removed: int = 0
    fallbacks: int = 0
    tune_rejected: int = 0
    refreshes: int = 0
    relaxed: int = 0
    max_drift: float = 0.0
    full: bool = False
    history: list = field(default_factory=list)


class Seeder:
    """Builds the atom set of one layer of ``model`` by random trials (module docstring)."""

    def __init__(self, model: Model, layer=0, config: Optional[SeedConfig] = None,
                 region: Optional[np.ndarray] = None, sync: bool = True):
        self.model = model
        self.config = config if config is not None else SeedConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.store = AtomStore(model.params, layer, sync=sync)
        self.pixel_A = model.experiment.imaging.pixel_size_nm * 10.0
        self.shape = tuple(model.experiment.imaging.shape)
        self.region = Region(self.shape, region)
        self.region.check(self.config.margin_px)
        self.neighbours = NeighbourSampler()
        self.stats = SeedStats()
        self.filter_report: Dict[str, int] = {}
        self.trainer: Optional[Trainer] = None
        self.local: Optional[LocalLoss] = None
        self.local_note = ""
        self.loss = math.inf
        self._version = 0
        self._raw = None
        self._landscape = None
        self._distance = (-1, None)
        self.resync(sync_store=False)

    # ------------------------------------------------------------------ radii in pixels
    def _px(self, value_A: float) -> float:
        return float(value_A) / self.pixel_A

    touch_px = property(lambda self: self._px(self.config.touch_A))
    merge_px = property(lambda self: self._px(self.config.merge_A))
    move_sigma_px = property(lambda self: self._px(self.config.move_sigma_A))
    relax_px = property(lambda self: self._px(self.config.relax_A))

    @property
    def margin(self) -> float:
        return float(self.config.margin_px)

    @property
    def n_atoms(self) -> int:
        return self.store.n

    @property
    def uses_local(self) -> bool:
        return self.local is not None

    @property
    def gradient_supported(self) -> bool:
        return self.uses_local and self.local.local.stem and not self.store.mixed

    # ------------------------------------------------------------------ state
    def resync(self, sync_store: bool = True) -> float:
        """Re-read everything after a change from outside (training, loading, parameters)."""

        if sync_store:
            self.store.sync()
        else:
            self.store._read_mirror()
        self._version += 1
        self.local = None
        self.local_note = ""
        model = self.model
        if self.config.local and self.config.loss == "ncc":
            local = LocalImage(model.simulator, model.layers(), model.params.optics_state(),
                               refresh_every=self.config.refresh_every, radius_px=self.config.local_radius_px)
            target, mask, reference = model.objective.regions()
            self.local = LocalLoss(local, target, reference, mask, model.params.normalisation, self.store.n)
        self.loss = self.local.loss if self.uses_local else self.evaluate()
        return self.loss

    def evaluate(self) -> float:
        """Loss of the current configuration from a complete simulation."""

        if self.store.n == 0:
            return math.inf
        with torch.no_grad():
            if self.config.loss == "ncc":
                target, mask, reference = self.model.objective.regions()
                value = negative_correlation(self.model.image(), target, reference, mask,
                                             self.model.params.normalisation)
            else:
                value = float(self.model.loss()[0])
        return value if math.isfinite(value) else math.inf

    # ------------------------------------------------------------------ proposals
    def resolve(self, x, y, exclude=None):
        return resolve_point(self.store.positions, x, y, self.touch_px, self.merge_px, exclude)

    def _option_list(self) -> List[float]:
        return list(self.config.weight_options) if self.store.supports_weights else [1.0]

    def _amplitude(self, option: float) -> float:
        spec, params, sim = self.store.spec, self.model.params, self.model.simulator
        return (option * self.store.new_weight * float(params.groups[spec.group].contrast)
                * float(sim.cross_sections(spec.element, spec.debye_waller_u)))

    def _min_gain(self):
        """Best loss change per pixel over all weight options, and the option that gives it."""

        gain = weight = None
        for option in self._option_list():
            one = self.local.insertion_gain(self._amplitude(option))
            if one is None:
                continue
            if gain is None:
                gain, weight = one, np.full(one.shape, option)
            else:
                better = one < gain
                gain, weight = np.where(better, one, gain), np.where(better, option, weight)
        return gain, weight

    def _raw_maps(self):
        key = (self._version, tuple(self.config.weight_options))
        if self._raw is None or self._raw[0] != key:
            gain, weight = self._min_gain()
            self._raw = (key, gain, weight)
        return self._raw[1], self._raw[2]

    def _atom_distance(self):
        if self._distance[0] != self._version:
            self._distance = (self._version, distance_map(self.shape, self.store.positions))
        return self._distance[1]

    def _landscape_map(self) -> Optional[Landscape]:
        if not self.config.gradient_seeding or not self.gradient_supported or self.store.n == 0:
            return None
        if self._landscape is None or self._landscape[0] != self._version:
            gain, _ = self._raw_maps()
            landscape = None
            if gain is not None:
                free = self._atom_distance() > self.merge_px
                landscape = Landscape(np.where(self.region.valid(self.margin) & free & (gain < 0.0), gain, np.inf))
            self._landscape = (self._version, landscape)
        return self._landscape[1]

    def propose(self, exclude=None):
        """A resolved proposal ``((x, y), removed, merged)``."""

        c = self.config
        landscape = self._landscape_map()
        point = None
        if c.neighbour_seeding:
            point = self.neighbours.point(
                self.rng, self.store.positions, self.region, self.margin, self._px(c.bond_A),
                self._px(c.bond_sigma_A), self._px(c.neighbour_min_A), self._px(c.neighbour_max_A),
                exclude, landscape, c.gradient_temperature)
        elif landscape is not None:
            pixel = landscape.pixel(self.rng, c.gradient_temperature)
            if pixel is not None:
                point = self.region.jitter(self.rng, pixel[0], pixel[1], self.margin)
        if point is None:
            point = self.region.uniform(self.rng, self.margin)
        return self.resolve(*point, exclude)

    # ------------------------------------------------------------------ evaluation of a change
    def _local_trial(self, remove, add, options) -> Optional[Trial]:
        store = self.store
        n_after = store.n - len(remove) + len(add)
        return self.local.propose(store.index, store.changes(remove, add, options), n_after)

    def _best_trial(self, remove, add):
        """Best trial over the weight options: ``(trial, options)`` or ``(None, None)``."""

        best, chosen = None, None
        for option in (self._option_list() if len(add) else [1.0]):
            options = np.full(len(add), option)
            trial = self._local_trial(remove, add, options)
            if trial is not None and (best is None or trial.loss < best.loss or not math.isfinite(best.loss)):
                best, chosen = trial, options
        return best, chosen

    def _accept_or_reject(self, remove, add, removed, merged, options=None) -> bool:
        stats, store = self.stats, self.store
        add = np.asarray(add, dtype=np.float64).reshape(-1, 2)
        trial = None
        if self.uses_local:
            if options is not None:
                options = np.asarray(options, dtype=np.float64)
                trial = self._local_trial(remove, add, options)
            else:
                trial, options = self._best_trial(remove, add)
            if trial is None:
                stats.fallbacks += 1

        if trial is not None:
            if not (math.isfinite(trial.loss) and trial.loss <= self.loss):
                return False
            store.apply(remove, add, options)
            self.local.commit(trial)
            self.loss = trial.loss
            self._version += 1
            self._count(removed, merged)
            if self.local.local.needs_refresh:
                self._refresh_local()
            return True

        token = store.apply(remove, add, options)
        new_loss = self.evaluate()
        if not (math.isfinite(new_loss) and new_loss <= self.loss):
            store.undo(token)
            return False
        self.loss = new_loss
        self._version += 1
        self._count(removed, merged)
        if self.uses_local:
            self._refresh_local()
        return True

    def _count(self, removed, merged):
        self.stats.accepted += 1
        self.stats.removed += removed
        self.stats.merged += int(merged)

    def _refresh_local(self):
        """Rebuild the local image from a full simulation (removes the drift of the window updates)."""

        before = self.loss
        model = self.model
        self.local.local.refresh(model.layers(), model.params.optics_state())
        after = self.local.reset(self.store.n)
        self.stats.refreshes += 1
        self.stats.max_drift = max(self.stats.max_drift, abs(after - before) if math.isfinite(before) else 0.0)
        self.loss = after

    # ------------------------------------------------------------------ trial types
    def _relax_trial(self):
        """Insert with evasion (module docstring); ``None`` if impossible, else accepted or not."""

        store, local = self.store, self.local
        if not self.gradient_supported or store.n < 2:
            return None
        raw, weight_map = self._raw_maps()
        if raw is None:
            return None
        temperature = self.config.gradient_temperature
        distance = self._atom_distance()
        simple = (distance > self.merge_px) & (raw < 0.0)
        candidates = self.region.valid(self.margin) & (distance <= self.relax_px) & ~simple & np.isfinite(raw)
        rows, cols = np.nonzero(candidates)
        if len(rows) == 0:
            return False
        row, col = Landscape(raw).draw_in(self.rng, rows, cols, temperature)
        p0 = np.array(self.region.jitter(self.rng, row, col, self.margin))
        w0 = float(weight_map[row, col])

        positions = store.positions
        d = np.hypot(positions[:, 0] - p0[0], positions[:, 1] - p0[1])
        q = int(np.argmin(d))
        if d[q] > self.relax_px or (d <= self.merge_px).sum() > 1:
            return False

        trial = self._local_trial([q], p0[None], np.array([w0]))
        if trial is None:
            return False
        with local.hypothetical(trial):
            gain, weight_map2 = self._min_gain()
        if gain is None:
            return False
        others = np.vstack([np.delete(positions, q, axis=0), p0[None]])
        yy, xx = np.mgrid[0:self.shape[0], 0:self.shape[1]]
        near = np.hypot(yy - positions[q, 0], xx - positions[q, 1]) <= self.relax_px
        allowed = (self.region.valid(self.margin) & near & (distance_map(self.shape, others) > self.merge_px)
                   & np.isfinite(gain))
        rows, cols = np.nonzero(allowed)
        if len(rows) == 0:
            return False
        row, col = Landscape(gain).draw_in(self.rng, rows, cols, temperature)
        q_new = np.array(self.region.jitter(self.rng, row, col, self.margin))
        w1 = float(weight_map2[row, col])
        ok = self._accept_or_reject([q], np.array([p0, q_new]), 0, False, options=np.array([w0, w1]))
        if ok:
            self.stats.relaxed += 1
        return ok

    def trial(self) -> bool:
        """One trial (module docstring); returns whether it was accepted."""

        c, stats, store = self.config, self.stats, self.store
        n = store.n
        draw = self.rng.random()
        if n > 0 and draw < c.remove_prob:
            kind = "remove"
        elif n > 0 and draw < c.remove_prob + c.move_prob:
            kind = "move"
        elif n > 1 and draw < c.remove_prob + c.move_prob + c.relax_prob and self.gradient_supported:
            kind = "relax"
        else:
            kind = "add"

        if kind in ("add", "relax") and c.max_atoms is not None and n >= c.max_atoms:
            stats.full = True
            return False
        stats.trials += 1

        if kind == "relax":
            return bool(self._relax_trial())

        merged = False
        if kind == "remove":
            remove = [int(self.rng.integers(n))]
            add = np.zeros((0, 2))
        elif kind == "move":
            index = int(self.rng.integers(n))
            if self.rng.random() < 0.5:
                jump = store.positions[index] + self.rng.normal(0.0, self.move_sigma_px, 2)
                xy, graveyard, merged = self.resolve(*jump, exclude=[index])
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

    # ------------------------------------------------------------------ given points
    def filter_points(self, points, commit=False, check_margin=True, check_collisions=True,
                      check_loss=True) -> np.ndarray:
        """Check given points in priority order against the criteria of an ``add`` trial.

        Per point: inside margin/region, ``max_atoms``, collision rule, and loss not worse.
        ``commit=False`` tests every point independently against the current state and changes
        nothing; ``commit=True`` inserts the accepted points one after the other.  Returns the
        accepted positions (after the collision rule); ``filter_report`` counts the reasons.
        """

        accepted = []
        count = self.store.n
        report = self.filter_report = {"outside": 0, "full": 0, "loss": 0, "accepted": 0}
        for point in np.asarray(points, dtype=np.float64).reshape(-1, 2):
            if self.config.max_atoms is not None and count >= self.config.max_atoms:
                self.stats.full = True
                report["full"] += 1
                continue
            if check_margin and not self.region.inside(*point, self.margin):
                report["outside"] += 1
                continue
            if check_collisions:
                xy, graveyard, merged = self.resolve(*point)
                if check_margin and not self.region.inside(*xy, self.margin):
                    report["outside"] += 1
                    continue
            else:
                xy, graveyard, merged = (float(point[0]), float(point[1])), [], False
            if not check_loss:
                ok = True
                if commit:
                    self.store.apply(graveyard, np.array([xy]))
                    self._version += 1
            elif commit:
                ok = self._accept_or_reject(graveyard, np.array([xy]), len(graveyard), merged)
            else:
                ok = self._would_accept(graveyard, xy)
            if ok:
                accepted.append(xy)
                count += 1 - len(graveyard)
                report["accepted"] += 1
            else:
                report["loss"] += 1
        return np.array(accepted, dtype=np.float64).reshape(-1, 2)

    def _would_accept(self, remove, xy) -> bool:
        add = np.array([xy])
        if self.uses_local:
            for option in self._option_list():
                trial = self._local_trial(remove, add, np.array([option]))
                if trial is not None and math.isfinite(trial.loss) and trial.loss <= self.loss:
                    return True
            return False
        token = self.store.apply(remove, add)
        try:
            new_loss = self.evaluate()
        finally:
            self.store.undo(token)
        return math.isfinite(new_loss) and new_loss <= self.loss

    # ------------------------------------------------------------------ tuning and loop
    def tune(self, steps: int) -> None:
        """A few optimiser steps on the enabled parameters; undone if the correlation gets worse."""

        if steps <= 0 or self.store.n == 0:
            return
        if self.trainer is None:
            self.trainer = Trainer(self.model)
        if not self.trainer.prepare():
            return
        settings, params = self.model.settings, self.model.params
        saved = [(t, t.detach().clone()) for key, p in settings.params().items() if p.enabled
                 for t in params.tensors(key, settings)]
        before = self.resync(sync_store=False)
        for _ in range(steps):
            self.trainer.step()
        self.resync(sync_store=False)
        if not self.loss <= before:
            with torch.no_grad():
                for tensor, value in saved:
                    tensor.copy_(value)
            self.trainer.optimiser = None
            self.resync(sync_store=False)
            self.stats.tune_rejected += 1

    def run(self, trials: int, tune_every: int = 0, tune_steps: int = 0, log_every: int = 200,
            save_every: int = 0, on_save: Optional[Callable] = None, callback: Optional[Callable] = None,
            should_stop: Optional[Callable[[], bool]] = None) -> SeedStats:
        """``trials`` trials.  ``callback(seeder)`` is called at every log, ``on_save(seeder)`` at every save."""

        first = self.stats.trials
        for _ in range(trials):
            if should_stop is not None and should_stop():
                break
            self.trial()
            done = self.stats.trials - first
            if self.stats.full:
                break
            if tune_every and tune_steps and done and done % tune_every == 0:
                self.tune(tune_steps)
            if log_every and done and done % log_every == 0:
                self.stats.history.append((self.stats.trials, self.store.n, self.loss))
                if callback is not None:
                    callback(self)
            if save_every and on_save is not None and done and done % save_every == 0:
                on_save(self)
        return self.stats

    def positions_px(self) -> np.ndarray:
        return self.store.positions.copy()

    def result(self):
        """``(Structure, ModelState)`` that reproduce the current atoms when loaded."""

        return self.model.params.to_structure(), self.model.params.to_state()


def empty_experiment(shape, pixel_size_nm, element="C", elements=(), group_name="Gr", mode="stem_haadf",
                     voltage_kV=80.0, padding=100, convergence_mrad=24.81, detector_mrad=(60.0, 200.0),
                     source_size_A=0.3, aberrations=None, probe_aberrations=None, amp_contrast=0.1,
                     alpha_max_mrad=50.0, init_occupancy=6.0, debye_waller_u=None, mask_file=None,
                     target_file=None, start_positions=None, start_weights=None,
                     envelope=None) -> Experiment:
    """An experiment with one empty layer, the starting point of a seeding run."""

    start = np.zeros((0, 2)) if start_positions is None or len(start_positions) == 0 \
        else np.asarray(start_positions, dtype=np.float64).reshape(-1, 2)
    from ...physics.structure import atomic_number_of
    layer = Layer(group_name, element, z_nm=0.0, group=group_name, init_occupancy=init_occupancy,
                  occ_entropy=False, elements=list(elements),
                  debye_waller_u=TYPICAL_DEBYE_WALLER_U.get(element, 0.0) if debye_waller_u is None
                  else float(debye_waller_u))
    group = Group(group_name, control_points=[4, 4], mask_file=mask_file, target_file=target_file)
    stem = Stem(float(convergence_mrad), tuple(detector_mrad), float(source_size_A),
                dict(probe_aberrations) if probe_aberrations else None)
    imaging = Imaging(tuple(shape), float(pixel_size_nm), mode=mode, voltage_kV=float(voltage_kV),
                      padding=padding, aberrations=dict(aberrations or {}),
                      envelope=dict(envelope or {}), amp_contrast=float(amp_contrast),
                      alpha_max_mrad=float(alpha_max_mrad), stem=stem)
    z = np.full(len(start), float(atomic_number_of(element)))
    factors = None
    if start_weights is not None and len(start_weights) == len(start):
        factors = [np.asarray(start_weights, dtype=np.float64)]
    return Experiment(imaging, [layer], [group], Structure([start], [z], factors))
