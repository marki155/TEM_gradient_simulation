r"""Part IV, Section 2: the session, the one owner of a reconstruction.

Theory
======

A reconstruction is a sequence of *states*; the optimiser (Part II, Section 5) and
the seeding (Part II, Section 6) move from one state to the next.  Everything that
a person does in the meantime -- switching a parameter, changing a weight,
deleting weak atoms, inserting candidate sites -- is a *change of state between
two steps*.  To make this well defined the model has exactly one owner, the
:class:`Session`, and every access goes through it.  The session is **not**
thread safe by design: the server (Section 4) calls it from one thread only and
interleaves the commands of the clients with the steps of the running job.

The session answers three kinds of requests.

**Queries** change nothing: :meth:`Session.info`, :meth:`Session.frame` (images,
atoms, optional extras), :meth:`Session.minima`.

**Settings** change the objective or the switches, never the parameters:
:meth:`~Session.set_param`, :meth:`~Session.set_loss`, ... Because the model
reads its settings at every step (Part II, Section 4) they act on the next step
without rebuilding anything.  The optimiser keeps the Adam moments of all
tensors that stay trainable (Part II, Section 5).

**Edits** change the parameters or the atom list (:meth:`~Session.edit`,
:meth:`~Session.insert_minima`).  After an edit the seeder re-reads the state
(:meth:`~temsim.learning.seeding.seeder.Seeder.resync`) because its running sums
(Part II, Eq. 6.4) are only valid for the state they were computed from.

The *frame* is the picture of the state that a client draws.  It contains

* the standardised simulated image :math:`x`, the standardised measured image :math:`y`
  and their difference :math:`\delta = x - y` (Part II, Eq. 2.2), each multiplied by the
  *loss mask* :math:`M`, so that what is not part of the loss is dark (as the images
  of the earlier interface were); with a target of its own for the active group there is
  no mask and the images are shown whole,
* the mask of the active group,
* for every layer the effective positions :math:`\mathbf r^{\rm eff}`
  (Part I, Eq. 1.2), the occupancies :math:`o` and the intensities :math:`I`,

and, on request, the optics (phase of the transfer function and the aperture ring
in TEM, the probe in STEM), the strain maps of the active group, the projected
potentials, and the gradient :math:`\partial\mathcal L/\partial\,\Delta\mathbf r` of the
loss with respect to the free position shifts.  Positions are in pixels
``(row, column)`` throughout.

Implementation
==============
"""

from __future__ import annotations

import copy
import dataclasses
import math
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..data import dataset as datasets
from ..data import store
from ..data.schema import Experiment, TrainSettings
from ..learning.model import Model
from ..learning.seeding import SeedConfig, Seeder, minima_positions
from ..learning.trainer import Trainer
from ..physics import optics, strain as strain_fields
from . import assemble, workspace as workspaces
from .assemble import Assembly

STRAIN_MAP_RESOLUTION = 256
CTF_MAX_ANGLE_MRAD = 42.0
PROBE_WINDOW_PX = 64

Arrays = Dict[str, np.ndarray]


@dataclasses.dataclass
class Capture:
    """What a training step leaves behind for a frame: tensors that nobody modifies any more
    (so that another thread can turn them into arrays) and plain numbers."""

    x: torch.Tensor
    y: torch.Tensor
    loss_mask: Optional[torch.Tensor]
    mask: Optional[np.ndarray]
    atoms: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    layers: List[Dict[str, Any]]
    data: Dict[str, Any]
    seq: int = 0


class SessionError(RuntimeError):
    """A request that cannot be fulfilled (reported to the client, the model is unchanged)."""


def _pick_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class Session:
    """One reconstruction: assembly, model, trainer and (lazily) seeder."""

    #: Seeding options that a running seeder reads afresh at every trial; any other change builds a new seeder.
    LIVE_SEED_OPTIONS = frozenset({"touch_A", "merge_A", "move_prob", "remove_prob", "relax_prob", "move_sigma_A",
                                   "relax_A", "neighbour_seeding", "bond_A", "bond_sigma_A", "neighbour_min_A",
                                   "neighbour_max_A", "gradient_seeding", "gradient_temperature", "weight_options",
                                   "max_atoms"})

    def __init__(self, device: str = "auto"):
        self.device = _pick_device(device)
        self.assembly: Optional[Assembly] = None
        self.model: Optional[Model] = None
        self.trainer: Optional[Trainer] = None
        self.seeder: Optional[Seeder] = None
        self.seed_config = SeedConfig()
        self.seed_layer = 0
        self.revision = 0
        self.last_save: Optional[str] = None
        #: the directory of ``--workspace``: relative paths are read from and named runs written to it
        self.workspace: Optional[str] = None
        self.candidates = np.zeros((0, 2))

    # ------------------------------------------------------------------ opening
    @property
    def is_open(self) -> bool:
        return self.model is not None

    def require_model(self) -> Model:
        if self.model is None:
            raise SessionError("nothing is open; send open_run or open_dataset first")
        return self.model

    def open_assembly(self, assembly: Assembly) -> Dict[str, Any]:
        model = Model(assembly.experiment, assembly.target, assembly.settings, assembly.state, self.device)
        self.assembly, self.model = assembly, model
        self.trainer, self.seeder = Trainer(model), None
        self.seed_layer = 0
        self.candidates = np.zeros((0, 2))
        self.revision += 1
        return self.info()

    def set_workspace(self, directory: Optional[str]) -> None:
        if directory:
            directory = os.path.abspath(directory)
            os.makedirs(directory, exist_ok=True)
        self.workspace = directory or None

    def workspace_entries(self) -> Dict[str, Any]:
        if self.workspace is None:
            raise SessionError("the core runs without a workspace (start it with --workspace DIR)")
        return {"root": self.workspace, "entries": workspaces.scan(self.workspace)}

    def open_run(self, directory: str, target_path: Optional[str] = None) -> Dict[str, Any]:
        return self.open_assembly(assemble.run_assembly(workspaces.resolve(self.workspace, directory), target_path))

    def open_dataset(self, path: str, route: str = "auto", resume: bool = True, **options) -> Dict[str, Any]:
        """``route``: ``"lattice"``, ``"seed"``, ``"run"`` or ``"auto"``.

        ``auto``: a directory with an experiment file is a run; an ``init.json`` with layers and a
        lattice starts the lattice route; everything else the seeding route.  ``resume``: continue
        from the autosave of the output folder if there is one.
        """

        path = workspaces.resolve(self.workspace, path)
        is_run = os.path.isdir(path) and any(os.path.exists(os.path.join(path, n)) for n in store.EXPERIMENT_FILES)
        if route == "run" or (route == "auto" and is_run):
            return self.open_run(path)
        data = datasets.load_dataset(path)
        spec = data.spec
        if route == "auto":
            route = "lattice" if spec is not None and spec.layers and spec.lattices else "seed"
        if route == "lattice":
            built = assemble.lattice_assembly(data)
        elif route == "seed":
            built = assemble.seeding_assembly(data, **options)
        else:
            raise SessionError(f"unknown route {route!r}")
        return self.open_assembly(assemble.resume_assembly(built) if resume else built)

    # ------------------------------------------------------------------ description
    def experiment(self) -> Experiment:
        """The specification with the atoms as they are now."""

        model = self.require_model()
        out = copy.copy(model.experiment)
        out.structure = model.params.to_structure()
        return out

    def info(self) -> Dict[str, Any]:
        if self.model is None:
            return {"open": False, "workspace": self.workspace}
        model, a = self.model, self.assembly
        imaging = model.experiment.imaging
        return {"open": True, "workspace": self.workspace, "kind": a.kind, "revision": self.revision, "device": self.device,
                "shape": list(imaging.shape), "pixel_size_nm": imaging.pixel_size_nm, "mode": imaging.mode,
                "voltage_kV": imaging.voltage_kV,
                "layers": [{"name": s.name, "group": s.group, "element": s.element, "z_nm": s.z_nm,
                            "atoms": int(n)} for s, n in zip(model.experiment.layers, model.params.counts())],
                "groups": [g.name for g in model.experiment.groups],
                "active_group": model.settings.active_group, "out_dir": a.out_dir, "notes": list(a.notes),
                "steps": self.trainer.steps if self.trainer else 0,
                "seeding": None if self.seeder is None else self.seeding_status()}

    # ------------------------------------------------------------------ settings
    def settings_changed(self) -> None:
        """Call after a change that the seeder's sums do not know about."""

        if self.seeder is not None:
            self.seeder.resync(sync_store=False)

    def set_train(self, new: TrainSettings) -> None:
        """Replace the settings *in place* (model, objective and optimiser hold a reference).

        Masks and group targets are kept if ``new`` carries none.
        """

        model = self.require_model()
        current = model.settings
        masks, targets = current.masks, current.targets
        for f in dataclasses.fields(TrainSettings):
            setattr(current, f.name, getattr(new, f.name))
        if not new.masks and not new.targets:
            current.masks, current.targets = masks, targets
        current.ensure_for(model.experiment)
        self.revision += 1
        self.settings_changed()

    def set_param(self, key: str, enabled: Optional[bool] = None, lr: Optional[float] = None) -> None:
        param = self.require_model().settings.param(key)
        if param is None:
            raise SessionError(f"unknown trainable key {key!r}")
        if enabled is not None:
            param.enabled = bool(enabled)
        if lr is not None:
            param.lr = float(lr)

    @staticmethod
    def _patch(target, values: Dict[str, Any], what: str) -> None:
        names = {f.name for f in dataclasses.fields(target)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise SessionError(f"unknown {what}: {unknown}; known: {sorted(names)}")
        for name, value in values.items():
            setattr(target, name, float(value))

    def set_loss(self, **weights) -> None:
        self._patch(self.require_model().settings.loss, weights, "loss weight")

    def set_regularisation(self, **weights) -> None:
        self._patch(self.require_model().settings.regularisation, weights, "prior weight")

    def set_active_group(self, name: Optional[str]) -> None:
        model = self.require_model()
        if name is not None and name not in model.experiment.group_names:
            raise SessionError(f"unknown group {name!r}; known: {model.experiment.group_names}")
        model.settings.active_group = name
        self.revision += 1
        self.settings_changed()

    def seeding_config(self, patch: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Read, and if ``patch`` is given change, the seeding configuration."""

        if patch:
            names = {f.name for f in dataclasses.fields(SeedConfig)}
            unknown = sorted(set(patch) - names)
            if unknown:
                raise SessionError(f"unknown seeding option: {unknown}")
            try:
                self.seed_config = SeedConfig(**{**dataclasses.asdict(self.seed_config), **patch})
            except ValueError as exc:
                raise SessionError(str(exc)) from exc
            if self.seeder is not None:
                if set(patch) <= self.LIVE_SEED_OPTIONS:
                    self.seeder.config = self.seed_config
                else:
                    self.seeder = None
        return dataclasses.asdict(self.seed_config)

    # ------------------------------------------------------------------ editable numbers
    def values(self) -> Dict[str, Any]:
        """The few numbers a person edits by hand, under flat keys.

        ``optics.amp_contrast``, ``optics.delta_E``, ``optics.Cc_um``, one ``optics.<key>``
        per aberration (always magnitude and angle, ``C12`` and ``phi12``, whatever the
        internal parametrisation is), ``normalisation``, and per group
        ``groups.<g>.shift|rotation|scale|contrast``.  Per-atom numbers are not included.
        """

        p = self.require_model().params
        out: Dict[str, Any] = {"optics.amp_contrast": float(p.amp_contrast), "optics.delta_E": float(p.delta_E),
                               "optics.Cc_um": float(p.Cc_um),
                               "normalisation": [float(v) for v in p.normalisation.detach().cpu()]}
        for key, value in self._polar_aberrations().items():
            out[f"optics.{key}"] = value
        for name, g in p.groups.items():
            out[f"groups.{name}.shift"] = [float(v) for v in g.shift.detach().cpu()]
            out[f"groups.{name}.rotation"] = float(g.rotation)
            out[f"groups.{name}.scale"] = [float(v) for v in g.scale.detach().cpu()]
            out[f"groups.{name}.contrast"] = float(g.contrast)
        return out

    def _polar_aberrations(self) -> Dict[str, float]:
        p = self.require_model().params
        raw = {k: float(v) for k, v in p.aberrations.items()}
        return optics.cartesian_to_polar(raw) if p.parametrization == "cartesian" else raw

    def set_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Write numbers of :meth:`values`; all keys are checked before anything is written."""

        p = self.require_model().params
        known = self.values()
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise SessionError(f"unknown value keys {unknown}")
        for key, value in values.items():
            if np.shape(value) != np.shape(known[key]):
                raise SessionError(f"{key!r} needs shape {np.shape(known[key])}, got {np.shape(value)}")
        scalar = {"optics.amp_contrast": p.amp_contrast, "optics.delta_E": p.delta_E, "optics.Cc_um": p.Cc_um,
                  "normalisation": p.normalisation}
        polar = self._polar_aberrations()
        polar_changed = False
        with torch.no_grad():
            for key, value in values.items():
                if key in scalar:
                    tensor = scalar[key]
                elif key.startswith("groups."):
                    _, name, attr = key.split(".")
                    tensor = getattr(p.groups[name], attr)
                else:
                    polar[key[len("optics."):]] = float(value)
                    polar_changed = True
                    continue
                tensor.copy_(torch.as_tensor(np.asarray(value, dtype=np.float32), device=tensor.device).reshape(tensor.shape))
            if polar_changed:
                for k, v in optics.convert(polar, p.parametrization).items():
                    if k in p.aberrations:
                        p.aberrations[k].fill_(float(v))
        self.settings_changed()
        return self.values()

    # ------------------------------------------------------------------ jobs
    def layer_index(self, layer: Any = None) -> int:
        model = self.require_model()
        if layer is None:
            group = model.settings.active_group
            for i, spec in enumerate(model.experiment.layers):
                if group is None or spec.group == group:
                    return i
            return 0
        if isinstance(layer, str):
            return model.params.layer_index(layer)
        return int(layer)

    def make_seeder(self, layer: Any = None, sync_store: bool = True) -> Seeder:
        """The seeder for ``layer`` (default: first layer of the active group), synchronised with the state.

        Synchronising drops the atoms with occupancy below one half from the model; a read-only caller
        passes ``sync_store=False``.
        """

        model = self.require_model()
        index = self.layer_index(layer)
        if self.seeder is None or self.seed_layer != index:
            region = self.assembly.region if self.assembly is not None else None
            self.seeder = Seeder(model, index, self.seed_config, region=region, sync=sync_store)
            self.seed_layer = index
        else:
            self.seeder.resync(sync_store=sync_store)
        return self.seeder

    def seeding_status(self) -> Dict[str, Any]:
        s = self.seeder
        return {"layer": self.seed_layer, "atoms": s.n_atoms, "loss": s.loss if math.isfinite(s.loss) else None,
                "trials": s.stats.trials, "accepted": s.stats.accepted, "merged": s.stats.merged,
                "removed": s.stats.removed, "relaxed": s.stats.relaxed, "full": s.stats.full,
                "local": s.uses_local}

    def refine(self, layer: Any = None, repulsion: float = 10.0, repulsion_A: float = 1.42,
               train_occupancies: bool = False) -> None:
        """Switch the settings to those of a refinement of a seeded layer (Section 3)."""

        model = self.require_model()
        name = model.experiment.layers[self.layer_index(layer)].name
        region = self.assembly.region if self.assembly is not None else None
        self.set_train(assemble.training_start(model.settings, name, region, repulsion, repulsion_A,
                                               train_occupancies))

    # ------------------------------------------------------------------ edits
    def _layers_of(self, group: Optional[str]) -> List[int]:
        model = self.require_model()
        group = group if group is not None else model.settings.active_group
        return [i for i, s in enumerate(model.experiment.layers) if group is None or s.group == group]

    def edit(self, name: str, **args) -> Dict[str, Any]:
        """Named edits between two steps; returns a small report.

        ``reset_occupancies``, ``reset_intensities``, ``reset_free_positions``,
        ``remove_low_occupancy`` (``threshold``), ``merge_close`` (``distance_A``, default the merge distance
        of the regularisation; ``min_occupancy``), ``resample_control`` (``control_points``);
        all act on the layers of ``group`` (default: the active group).  ``merge`` (``layer``, ``atoms``: a
        list of atom indices) fuses those atoms of one layer into one.
        """

        model = self.require_model()
        params, spec = model.params, model.experiment
        group = args.get("group")
        report: Dict[str, Any] = {}
        if name == "reset_occupancies":
            for i in self._layers_of(group):
                with torch.no_grad():
                    params.layers[i].logit.fill_(float(spec.layers[i].init_occupancy))
        elif name == "reset_intensities":
            for i in self._layers_of(group):
                with torch.no_grad():
                    params.layers[i].intensity.copy_(params.layers[i].reference)
        elif name == "reset_free_positions":
            for i in self._layers_of(group):
                with torch.no_grad():
                    params.layers[i].shifts.zero_()
        elif name == "remove_low_occupancy":
            threshold = float(args.get("threshold", 0.5))
            removed = 0
            for i in self._layers_of(group):
                with torch.no_grad():
                    keep = (params.occupancy(i) >= threshold).cpu().numpy()
                removed += int((~keep).sum())
                if not keep.all():
                    params.remove_atoms(i, keep)
            report["removed"] = removed
            self.revision += 1
        elif name == "merge_close":
            reg = model.settings.regularisation
            distance_A = float(args.get("distance_A", reg.merge_distance_A))
            distance_px = distance_A / (spec.imaging.pixel_size_nm * 10.0)
            merged = 0
            for i in self._layers_of(group):
                merged += params.merge_close(i, distance_px, float(args.get("min_occupancy", 0.5)))
            report["merged"] = merged
            self.revision += 1
        elif name == "merge":
            i = self.layer_index(args.get("layer"))
            report["merged"] = params.merge_atoms(i, [args["atoms"]])
            self.revision += 1
        elif name == "resample_control":
            report.update(self._resample_control(group, args.get("control_points")))
            self.revision += 1
        else:
            raise SessionError(f"unknown edit {name!r}")
        report["atoms"] = params.counts()
        self.settings_changed()
        return report

    def _resample_control(self, group: Optional[str], control_points) -> Dict[str, Any]:
        model = self.require_model()
        group = group if group is not None else model.settings.active_group
        if group is None or group not in model.params.groups:
            raise SessionError("resample_control needs a group")
        if control_points is None or len(control_points) != 2:
            raise SessionError("control_points must be [rows, columns]")
        gp = model.params.groups[group]
        flag = gp.strain.requires_grad
        new = strain_fields.resample_control(gp.strain.detach(), control_points)
        deviation = strain_fields.resample_deviation(gp.strain.detach(), new, model.params.shape)
        gp.strain = new.to(model.params.device).clone().requires_grad_(flag)
        model.experiment.group(group).control_points = [int(control_points[0]), int(control_points[1])]
        return {"max_deviation_px": float(deviation)}

    # ------------------------------------------------------------------ candidate sites
    def minima(self, threshold: float = 0.5, distance_px: float = 3.0, layer: Any = None,
               atom_distance: bool = True, verify: Optional[Dict[str, bool]] = None) -> np.ndarray:
        """Candidate sites of Part II, Eq. (7.1) of ``layer`` (``(n, 2)``), only inside the loss mask.

        ``atom_distance``: drop sites closer than ``distance_px`` to an existing atom.
        ``verify``: ``{check_margin, check_collisions, check_loss}``; if given, the sites are
        additionally tested by the criteria of an ``add`` trial (nothing is inserted).
        """

        model = self.require_model()
        diff = model.evaluate()["diff"]
        _, loss_mask, _ = model.objective.regions()
        if loss_mask is not None:
            diff = np.where(loss_mask.detach().cpu().numpy() > 0, diff, 0.0)
        index = self.layer_index(layer)
        with torch.no_grad():
            atoms = model.params.positions(index).cpu().numpy() if atom_distance else None
        points = minima_positions(diff, threshold, distance_px, atoms)
        if verify is not None and len(points):
            points = self.make_seeder(layer, sync_store=False).filter_points(points, commit=False, **verify)
        self.candidates = points
        return self.candidates

    def insert_minima(self, threshold: float = 0.5, distance_px: float = 3.0, layer: Any = None,
                      check_margin: bool = True, check_collisions: bool = True,
                      check_loss: bool = True, atom_distance: bool = True) -> Dict[str, Any]:
        """Insert the candidate sites that pass the criteria of an ``add`` trial."""

        points = self.minima(threshold, distance_px, layer, atom_distance)
        seeder = self.make_seeder(layer)
        before = seeder.n_atoms
        accepted = seeder.filter_points(points, commit=True, check_margin=check_margin,
                                        check_collisions=check_collisions, check_loss=check_loss)
        self.revision += 1
        return {"candidates": int(len(points)), "accepted": int(len(accepted)), "atoms_before": before,
                "atoms": seeder.n_atoms, "report": dict(seeder.filter_report)}

    # ------------------------------------------------------------------ frame
    def frame(self, extras=(), standardised: bool = False) -> Tuple[Dict[str, Any], Arrays]:
        """The picture of the state; see the module docstring.  ``extras``: subset of
        ``optics``, ``strain``, ``potentials``, ``gradients``."""

        model = self.require_model()
        ev = model.evaluate()
        _, loss_mask, _ = model.objective.regions()
        arrays = self._images(ev["standardised"], ev["standardised_target"], loss_mask, standardised)
        mask = model.settings.mask_for(model.settings.active_group)
        if mask is not None:
            arrays["mask"] = (np.asarray(mask) > 0).astype(np.uint8)
        layers = []
        with torch.no_grad():
            for i, spec in enumerate(model.experiment.layers):
                lp = model.params.layers[i]
                arrays[f"atoms.{i}.pos"] = model.params.positions(i).cpu().numpy().astype(np.float32)
                arrays[f"atoms.{i}.occ"] = model.params.occupancy(i).cpu().numpy().astype(np.float32)
                arrays[f"atoms.{i}.int"] = lp.intensity.detach().cpu().numpy().astype(np.float32)
                layers.append({"name": spec.name, "group": spec.group, "element": spec.element, "atoms": len(lp)})
        extras = set(extras or ())
        if "optics" in extras:
            self._optics(arrays)
        if "strain" in extras:
            self._strain(arrays)
        if "potentials" in extras:
            self._potentials(arrays)
        if "gradients" in extras:
            self._gradients(arrays)
        data = {"loss": ev["loss"], "data": ev["data"], "prior": ev["prior"], "layers": layers,
                "active_group": model.settings.active_group, "revision": self.revision,
                "steps": self.trainer.steps if self.trainer else 0, "mode": model.experiment.imaging.mode}
        if self.seeder is not None:
            data["seeding"] = self.seeding_status()
        return data, arrays

    @staticmethod
    def _images(x, y, loss_mask, standardised: bool = False) -> Arrays:
        """``sim``, ``target``, ``diff`` (times the loss mask); ``x`` and ``y`` are tensors or arrays."""

        x = x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        y = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
        if loss_mask is None:
            shown = 1.0
        else:
            shown = ((loss_mask.cpu().numpy() if torch.is_tensor(loss_mask) else np.asarray(loss_mask)) > 0)
            shown = shown.astype(np.float32)
        arrays: Arrays = {"sim": (x * shown).astype(np.float32, copy=False),
                          "target": (y * shown).astype(np.float32, copy=False),
                          "diff": ((x - y) * shown).astype(np.float32, copy=False)}
        if standardised:
            arrays["x"] = x.astype(np.float32, copy=False)
            arrays["y"] = y.astype(np.float32, copy=False)
        return arrays

    @torch.no_grad()
    def capture(self, kept: Dict[str, Any], result) -> Capture:
        """Take what a training step computed (``Trainer.step(keep=True)``); cheap, no synchronisation.

        The atoms are copied because the optimiser changes them in place; the images need no copy.
        """

        model = self.require_model()
        atoms, layers = [], []
        for i, spec in enumerate(model.experiment.layers):
            lp = model.params.layers[i]
            atoms.append((model.params.positions(i).detach().clone(), model.params.occupancy(i).detach().clone(),
                          lp.intensity.detach().clone()))
            layers.append({"name": spec.name, "group": spec.group, "element": spec.element, "atoms": len(lp)})
        data = {"loss": result.loss, "data": result.data, "prior": result.prior, "layers": layers,
                "active_group": model.settings.active_group, "revision": self.revision, "steps": result.step,
                "mode": model.experiment.imaging.mode, "partial": True}
        mask = model.settings.mask_for(model.settings.active_group)
        return Capture(kept["x"], kept["y"], kept["loss_mask"], mask, atoms, layers, data)

    @staticmethod
    def compose(capture: Capture) -> Tuple[Dict[str, Any], Arrays]:
        """A frame without extras from a :class:`Capture`; touches no model, so any thread may call it."""

        arrays = Session._images(capture.x, capture.y, capture.loss_mask)
        if capture.mask is not None:
            arrays["mask"] = (np.asarray(capture.mask) > 0).astype(np.uint8)
        for i, (pos, occ, intensity) in enumerate(capture.atoms):
            arrays[f"atoms.{i}.pos"] = pos.cpu().numpy().astype(np.float32, copy=False)
            arrays[f"atoms.{i}.occ"] = occ.cpu().numpy().astype(np.float32, copy=False)
            arrays[f"atoms.{i}.int"] = intensity.cpu().numpy().astype(np.float32, copy=False)
        return dict(capture.data), arrays

    @torch.no_grad()
    def _optics(self, arrays: Arrays) -> None:
        sim = self.model.simulator
        state = self.model.params.optics_state()
        if sim.stem_mode:
            probe = torch.fft.fftshift(sim.probe_intensity(state)).cpu().numpy()
            cy, cx = probe.shape[0] // 2, probe.shape[1] // 2
            r = PROBE_WINDOW_PX
            arrays["probe"] = probe[cy - r:cy + r, cx - r:cx + r].astype(np.float32)
            return
        grid = sim.grid
        n, m = grid.shape
        reach = CTF_MAX_ANGLE_MRAD * 1e-3 / grid.wavelength_nm            # 1/nm
        half_r = min(n // 2, int(math.ceil(reach * n * grid.pixel_size_nm)) + 1)
        half_c = min(m // 2, int(math.ceil(reach * m * grid.pixel_size_nm)) + 1)

        def cut(a):
            a = torch.fft.fftshift(a)
            return a[n // 2 - half_r:n // 2 + half_r, m // 2 - half_c:m // 2 + half_c].cpu().numpy()

        phase = torch.angle(sim.transfer(state))
        arrays["ctf_phase"] = cut(phase).astype(np.float32)
        arrays["ctf_ring"] = cut(sim.objective_aperture.to(phase.dtype)).astype(np.float32)

    def _strain(self, arrays: Arrays) -> None:
        model = self.require_model()
        group = model.settings.active_group or model.experiment.groups[-1].name
        shape = model.params.shape
        scale = STRAIN_MAP_RESOLUTION / max(shape)
        resolution = (max(32, int(round(shape[0] * scale))), max(32, int(round(shape[1] * scale))))
        reg = model.settings.regularisation
        maps = strain_fields.maps(model.params.groups[group].strain.detach(), shape,
                                  reg.strain_energy_lambda, reg.strain_energy_mu, resolution)
        for key, value in maps.items():
            arrays[f"strain.{key}"] = np.asarray(value, dtype=np.float32)

    @torch.no_grad()
    def _potentials(self, arrays: Arrays) -> None:
        model = self.require_model()
        p = model.simulator.padding
        for i, potential in enumerate(model.simulator.potentials(model.layers())):
            if potential is None:
                continue
            v = potential.detach()
            if v.is_complex():
                v = v.real
            if p:
                v = v[p:-p, p:-p]
            arrays[f"potential.{i}"] = v.cpu().numpy().astype(np.float32)

    def _gradients(self, arrays: Arrays) -> None:
        """Gradient of the loss with respect to the position shifts of every layer, ``(n, 2)`` px."""

        model = self.require_model()
        shifts = [lp.shifts for lp in model.params.layers]
        flags = [t.requires_grad for t in shifts]
        try:
            for t in shifts:
                t.requires_grad_(True)
            with torch.enable_grad():
                loss, _, _ = model.loss()
                grads = torch.autograd.grad(loss, shifts, allow_unused=True)
        finally:
            for t, flag in zip(shifts, flags):
                t.requires_grad_(flag)
        for i, g in enumerate(grads):
            if g is not None:
                arrays[f"grad.{i}"] = g.detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ saving
    def save(self, directory: Optional[str] = None, name: Optional[str] = None) -> str:
        """Write a run directory: ``name`` inside the workspace, else ``directory``, else the autosave folder."""

        model = self.require_model()
        if name:
            if self.workspace is None:
                raise SessionError("a name needs a workspace (start the core with --workspace DIR)")
            try:
                directory = workspaces.child(self.workspace, name)
            except ValueError as exc:
                raise SessionError(str(exc)) from None
            if os.path.isdir(directory) and workspaces.kind(directory) != "run":
                raise SessionError(f"{name!r} exists and is not a saved run; it is not overwritten")
        elif directory:
            directory = workspaces.resolve(self.workspace, directory)
        if not directory and self.assembly is not None and self.assembly.out_dir:
            directory = os.path.join(self.assembly.out_dir, assemble.AUTOSAVE_DIR)
        if not directory:
            raise SessionError("no directory given and the session has no default one")
        run = store.Run(self.experiment(), model.settings, model.state(), self.assembly.target,
                        list(self.assembly.notes))
        staging = directory.rstrip("/\\") + ".writing"
        shutil.rmtree(staging, ignore_errors=True)
        store.save_run(staging, run)
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        os.replace(staging, directory)
        self.last_save = directory
        return directory
