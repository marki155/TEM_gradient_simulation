r"""Part IV, Section 3: from a data folder (or a run) to a model.

Theory
======

Parts I-III describe the pieces; the user wants to start from *a folder with a
measurement*.  Three routes lead to a reconstruction problem.

**Lattice route** (the ImageJ plug-in).  The folder says which lattices exist,
which layer of which lattice fills which experiment layer, and what is trained.
Each lattice has to cover the image after the largest allowed rigid motion of the
groups that use it (Part III, Eq. 3.1: ``canvas_size_px``), so its reach is

.. math::

    \Delta_{\max}(\text{lattice}) = \max_{g \in \text{groups using it}}\ \Delta_g,
    \qquad
    \vartheta_{\max}(\text{lattice}) = \max_{g}\ \vartheta_g .

The height of an experiment layer is its ``z_nm`` if the folder gives one; the
data model cannot distinguish "not given" from ``0``, therefore the rule is: if
*all* layers of a lattice are at ``z_nm == 0`` while the lattice has layers at
different heights, the heights of the lattice (plus its ``z_offset_nm``) are used.

**Seeding route** (no lattice).  The model starts with one empty layer of the first
group; the seeding of Part II, Section 6 fills it.  Only the microscope is
trained meanwhile (aberrations, normalisation), and the loss counts the image
without a border of ``border_px`` pixels where the multislice and the probe are
cut off (Part I, Section 9).  When the seeded layer is saved for further
training the settings are changed to those of a *refinement*: positions and
intensities are released, a repulsion keeps neighbours apart, all other priors
are off, and only defocus and spherical aberration are trained.

**Run route.**  A run directory (any layout, Part III, Section 3) already contains
specification, settings and state; the target is the one stored with it or the
target of the folder that contains it.

Implementation
==============

This module is plain data wrangling (numpy and the lattice generator of Part I; the
seeding start imports one helper of Part II lazily).  Its result is an
:class:`Assembly`; the :class:`~temsim.service.session.Session` turns it into a
model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data import dataset as datasets
from ..data import store
from ..data.dataset import Dataset
from ..data.schema import Experiment, Layer, ModelState, Structure, TrainSettings
from ..physics import structure as lattices

DEFAULT_BORDER_PX = 10
DEFAULT_SEED_LOGIT = 6.0
SEED_OUT_DIR = "config_random"


@dataclass
class Assembly:
    """Everything a session needs to build a model."""

    experiment: Experiment
    settings: TrainSettings
    target: np.ndarray
    state: Optional[ModelState] = None
    kind: str = "lattice"                      # "lattice" | "seed" | "run"
    dataset: Optional[Dataset] = None
    #: Where the program writes its runs.
    out_dir: Optional[str] = None
    #: Seeding only: allowed region (no border) and border width.
    region: Optional[np.ndarray] = None
    border_px: int = 0
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# lattice route
# ---------------------------------------------------------------------------

def _reach(spec) -> Dict[str, Tuple[float, float]]:
    """Largest shift [px] and rotation [deg] per lattice, from the groups that use it."""

    default = spec.lattices[0].name if spec.lattices else None
    by_group = {g.name: g for g in spec.groups}
    sources = {s.layer: s for s in spec.layer_sources}
    reach: Dict[str, Tuple[float, float]] = {}
    for layer in spec.layers:
        source = sources.get(layer.name)
        name = (source.lattice if source and source.lattice else None) or default
        group = by_group.get(layer.group or layer.name)
        shift = float(group.max_shift or 0.0) if group else 0.0
        rotation = float(np.rad2deg(group.max_rotation or 0.0)) if group else 0.0
        previous = reach.get(name, (0.0, 0.0))
        reach[name] = (max(previous[0], shift), max(previous[1], rotation))
    return reach


def _numbers(symbols: List[str], fallback: str, n: int) -> np.ndarray:
    if symbols:
        return np.array([float(lattices.atomic_number_of(s)) for s in symbols], dtype=np.float64)
    return np.full(n, float(lattices.atomic_number_of(fallback)))


def lattice_assembly(data: Dataset) -> Assembly:
    """Experiment, settings and target of an ImageJ-style folder."""

    spec = data.spec
    if spec is None or not spec.layers:
        raise ValueError("the folder has no init.json with layers; use the seeding route instead")
    if not spec.lattices:
        raise ValueError("init.json has layers but no lattice")
    shape = data.shape
    imaging = spec.to_imaging(shape)
    reach = _reach(spec)

    built = {}
    for lattice in spec.lattices:
        shift, rotation = reach.get(lattice.name, (0.0, 0.0))
        built[lattice.name] = lattices.build_lattice(lattice, shape, spec.pixel_size_nm, shift, rotation)

    sources = {s.layer: s for s in spec.layer_sources}
    counters: Dict[str, int] = {}
    chosen = []
    for layer in spec.layers:
        source = sources.get(layer.name)
        name = (source.lattice if source and source.lattice else None) or spec.lattices[0].name
        if name not in built:
            raise ValueError(f"layer {layer.name!r} refers to the unknown lattice {name!r}; "
                             f"known: {sorted(built)}")
        index = source.lattice_layer if source and source.lattice_layer is not None else counters.get(name, 0)
        counters[name] = index + 1
        if index >= len(built[name]):
            raise ValueError(f"lattice {name!r} has {len(built[name])} layers, "
                             f"layer {layer.name!r} asks for number {index}")
        chosen.append((name, built[name][index]))

    layers, positions, numbers = [], [], []
    for layer, (name, generated) in zip(spec.layers, chosen):
        siblings = [l for l, (n, _) in zip(spec.layers, chosen) if n == name]
        heights = {round(g.z_nm, 6) for g in built[name]}
        use_lattice_z = all(l.z_nm == 0.0 for l in siblings) and len(heights) > 1
        element = layer.element or (generated.elements[0] if generated.elements else "C")
        layers.append(Layer(layer.name, element, z_nm=generated.z_nm if use_lattice_z else layer.z_nm,
                            group=layer.group, init_occupancy=layer.init_occupancy,
                            occ_entropy=layer.occ_entropy, elements=list(layer.elements),
                            debye_waller_u=layer.debye_waller_u))
        positions.append(np.asarray(generated.positions, dtype=np.float64))
        numbers.append(_numbers(generated.elements, element, len(generated.positions)))

    experiment = Experiment(imaging, layers, list(spec.groups), Structure(positions, numbers))
    settings = TrainSettings().ensure_for(experiment)
    unknown = datasets.apply_init_train(settings, spec.train)
    settings.ensure_for(experiment)
    notes = [f"unknown training key in init file: {key}" for key in unknown]
    settings.masks = {g.name: data.masks[g.name] for g in experiment.groups if g.name in data.masks}
    settings.targets = {g.name: data.group_targets[g.name] for g in experiment.groups
                        if g.name in data.group_targets}
    return Assembly(experiment, settings, data.target, None, "lattice", data,
                    os.path.join(data.folder, spec.config_dir), notes=notes)


# ---------------------------------------------------------------------------
# seeding route
# ---------------------------------------------------------------------------

def border_mask(shape, border_px: int) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    if border_px > 0:
        mask[:border_px] = mask[-border_px:] = False
        mask[:, :border_px] = mask[:, -border_px:] = False
    return mask


def seeding_settings(experiment: Experiment, region: Optional[np.ndarray], border_px: int,
                     lr_aberrations: float = 1e-2, lr_normalisation: float = 1e-3) -> TrainSettings:
    """Only the microscope is trained while atoms are seeded."""

    settings = TrainSettings().ensure_for(experiment)
    settings.aberrations.enabled, settings.aberrations.lr = True, lr_aberrations
    settings.normalisation.enabled, settings.normalisation.lr = True, lr_normalisation
    group = experiment.groups[0].name
    settings.active_group = group
    mask = border_mask(experiment.imaging.shape, border_px)
    if region is not None:
        mask &= np.asarray(region) > 0
    settings.masks = {group: mask}
    return settings


def seeding_assembly(data: Dataset, pixel_size_nm: Optional[float] = None, fov_nm: Optional[float] = None,
                     mode: Optional[str] = None, voltage_kV: Optional[float] = None,
                     border_px: int = DEFAULT_BORDER_PX, init_logit: float = DEFAULT_SEED_LOGIT,
                     convergence_mrad: Optional[float] = None, source_size_A: Optional[float] = None) -> Assembly:
    """An empty one-layer model for the folder (the first group and layer of ``init.json`` are used)."""

    from ..learning.seeding import empty_experiment          # the one place the service needs learning

    spec = data.spec
    shape = data.shape
    init = spec.imaging if spec else {}
    if pixel_size_nm is None and fov_nm is not None:
        pixel_size_nm = fov_nm / shape[1]
    if pixel_size_nm is None and spec is not None:
        pixel_size_nm = spec.pixel_size_nm
    if pixel_size_nm is None:
        raise ValueError("pixel size unknown: give pixel_size_nm or fov_nm (or put an init.json in the folder)")
    if mode is None:
        mode = init.get("mode", "stem_haadf")
    mode = "tem" if mode == "tem" else "stem_haadf"
    stem = dict(init.get("stem") or {})

    group = (spec.groups[0] if spec and spec.groups else None)
    layer = (spec.layers[0] if spec and spec.layers else None)
    if group is not None:
        name = group.name
    elif data.masks:
        name = next(iter(data.masks))
    else:
        name = datasets.DEFAULT_GROUP
    region = data.masks.get(name)
    if region is None and data.masks:
        region = next(iter(data.masks.values()))
    notes = []
    if spec and (len(spec.groups) > 1 or len(spec.layers) > 1):
        notes.append("only the first group/layer of init.json is seeded")
    if region is None:
        notes.append("no mask found: the whole image counts")

    element = (layer.element if layer else None) or "C"
    experiment = empty_experiment(
        shape, float(pixel_size_nm), element=element, elements=layer.elements if layer else (),
        group_name=name, mode=mode,
        voltage_kV=float(voltage_kV if voltage_kV is not None else (spec.voltage_kV if spec else 80.0)),
        padding=spec.padding if spec else 100,
        convergence_mrad=float(convergence_mrad if convergence_mrad is not None else stem.get("convergence_mrad", 24.81)),
        detector_mrad=tuple(stem.get("detector_mrad", (60.0, 200.0))),
        source_size_A=float(source_size_A if source_size_A is not None else stem.get("source_size_A", 0.3)),
        aberrations=init.get("aberrations") or None, probe_aberrations=stem.get("aberrations") or None,
        amp_contrast=float(init.get("amp_contrast", 0.1)), alpha_max_mrad=float(init.get("alpha_max_mrad", 50.0)),
        init_occupancy=float(init_logit),
        debye_waller_u=layer.debye_waller_u if layer and layer.debye_waller_u else None,
        mask_file=group.mask_file if group else None, envelope=init.get("envelope") or None)
    settings = seeding_settings(experiment, region, border_px)
    return Assembly(experiment, settings, data.target, None, "seed", data,
                    os.path.join(data.folder, SEED_OUT_DIR), region=region, border_px=int(border_px),
                    notes=notes)


def training_start(settings: TrainSettings, layer: str, region: Optional[np.ndarray] = None,
                   repulsion: float = 10.0, repulsion_A: float = 1.42,
                   train_occupancies: bool = False) -> TrainSettings:
    """Settings for refining a seeded layer (copy; the seeding settings stay untouched)."""

    import copy
    out = copy.deepcopy(settings)
    reg = out.regularisation
    reg.repulsion_a, reg.repulsion_distance_A = float(repulsion), float(repulsion_A)
    reg.position_shifts_a = reg.occ_entropy_a = reg.strain_energy_a = 0.0
    reg.strain_energy_lambda = reg.strain_energy_mu = 0.0
    out.aberration_flags = {k: k in ("C10", "C30") for k in out.aberration_flags}
    lt = out.layer(layer)
    lt.free_positions.enabled = True
    lt.intensities.enabled = not train_occupancies
    lt.occupancies.enabled = train_occupancies
    if region is not None and out.active_group is not None:
        out.masks[out.active_group] = np.asarray(region) > 0
    return out


# ---------------------------------------------------------------------------
# run route
# ---------------------------------------------------------------------------

def run_assembly(directory: str, target_path: Optional[str] = None) -> Assembly:
    run = store.load_run(directory)
    target = run.target
    if target is None:
        path = target_path or datasets.find_target(os.path.dirname(os.path.abspath(directory)))
        if path is None:
            raise FileNotFoundError(f"{directory} stores no target image and none was given")
        target = datasets.prepare_target(datasets.read_image(path))
    if tuple(target.shape) != tuple(run.experiment.imaging.shape):
        raise ValueError(f"target {tuple(target.shape)} does not fit the run {tuple(run.experiment.imaging.shape)}")
    return Assembly(run.experiment, run.train, np.asarray(target, dtype=np.float32), run.state, "run",
                    None, directory, notes=list(run.notes))
