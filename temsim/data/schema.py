r"""Part III - Data management.  Section 1: the data model.

Theory
======

A reconstruction task has four kinds of information, and keeping them apart is
the organising principle of the whole package.

1. **Specification** (:class:`Experiment`).  What is the sample made of and how
   is it imaged?  It consists of

   * the imaging setup :math:`\mathcal{I}` - mode (TEM or STEM-HAADF), image
     shape :math:`(H, W)`, pixel size :math:`\Delta x`, acceleration voltage
     :math:`U`, aberration coefficients, envelope, detector;
   * the layer stack - an ordered list of slices :math:`\ell = 1 \dots L` with
     element, height :math:`z_\ell`, and membership in a *group*;
   * the groups - rigid units that move together (global shift, rotation,
     scale, a smooth strain field);
   * the initial atomic structure, one list of positions per layer
     (Eq. 3.1: :math:`\mathbf{r}_{\ell,i} \in \mathbb{R}^2`, in pixels).

2. **Trainable state** (:class:`ModelState`).  The numbers the optimiser changes:
   per-atom position shifts :math:`\delta\mathbf{r}_{\ell,i}`, intensity (a
   continuous atomic number) :math:`Z_{\ell,i}`, occupancy logit
   :math:`o_{\ell,i}`, per-group transformation and strain control points, and
   the microscope parameters.  The *effective* position of an atom is

   .. math:: \mathbf{r}^{\rm eff}_{\ell,i} = \mathbf{r}_{\ell,i} + \delta\mathbf{r}_{\ell,i}
             \qquad (3.2)

   so the specification is never overwritten by training.

3. **Training settings** (:class:`TrainSettings`).  Which parts of the state may
   change, with which learning rate, how the loss terms are weighted, how strongly
   regularisers act, and which image regions count.  This is policy, not physics.

4. **Observations** (the target image and masks).  Bulk arrays; they live in
   ``.npy``/``.npz`` files next to the JSON, marked ``external`` below.

Implementation
==============

All classes are dataclasses without behaviour beyond validation and lookup.  They
import nothing but numpy, so that the user interface process can use them without
loading torch.  :mod:`temsim.data.codec` converts them to and from JSON;
:mod:`temsim.data.store` puts them on disk; :mod:`temsim.data.legacy` reads the
formats of earlier versions.

Units: lengths in the structure are *pixels* (origin at the image corner),
heights ``z_nm`` and the pixel size in nanometres, aberration coefficients in the
units of :mod:`temsim.physics.optics` (documented there), angles in mrad or rad as
named.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = 3

IMAGING_MODES = ("tem", "stem_haadf")

#: Typical one-dimensional thermal displacements u_rms [Angstrom] at room
#: temperature; ``u = sqrt(B / (8 pi^2))`` for a Debye-Waller factor B.
TYPICAL_DEBYE_WALLER_U = {
    "B": 0.070, "C": 0.060, "N": 0.060, "O": 0.070, "S": 0.070, "W": 0.055,
}


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------

@dataclass
class Layer:
    """One multislice slice."""

    name: str
    element: str
    z_nm: float = 0.0
    #: Name of the group this layer moves with (empty -> its own group).
    group: str = ""
    #: Initial occupancy *logit* o; the weight of an atom is sigmoid(o).
    init_occupancy: float = 2.0
    #: Apply the occupancy-entropy regulariser to this layer.
    occ_entropy: bool = True
    #: Elements the atomic number may interpolate between (e.g. B, C, N, O).
    #: Empty: one scattering kernel (``element``), intensity scales its amplitude.
    elements: List[str] = field(default_factory=list)
    #: Thermal displacement u_rms [Angstrom], 0 = frozen lattice.
    debye_waller_u: float = 0.0


@dataclass
class Group:
    """Atoms that move as one rigid unit plus a smooth strain field."""

    name: str
    #: Strain field control grid (rows, columns).
    control_points: List[int] = field(default_factory=lambda: [10, 10])
    #: Clamp on the global shift [px] / rotation [rad]; None = unbounded.
    max_shift: Optional[float] = None
    max_rotation: Optional[float] = None
    init_shift: List[float] = field(default_factory=lambda: [0.0, 0.0])
    init_rotation: float = 0.0
    init_scale: List[float] = field(default_factory=lambda: [1.0, 1.0])
    #: Empirical contrast factor of the group (Stobbs factor).  1 = untouched
    #: physical potential.  A documented fudge that makes the simulation/experiment
    #: discrepancy visible instead of hiding it in the occupancy.
    init_contrast_factor: float = 1.0
    #: Data sources, relative to the dataset folder.
    mask_file: Optional[str] = None
    target_file: Optional[str] = None
    #: While this group is active only its layers contribute to the image.
    isolate_when_active: bool = False


@dataclass
class Stem:
    """STEM-HAADF specific setup."""

    convergence_mrad: float = 25.0
    #: Inner and outer detector half-angle [mrad].
    detector_mrad: Tuple[float, float] = (60.0, 200.0)
    #: Gaussian source size, sigma [Angstrom].
    source_size_A: float = 0.5
    #: Probe aberrations; None -> use ``Imaging.aberrations``.
    aberrations: Optional[Dict[str, float]] = None


@dataclass
class Imaging:
    """How the sample is imaged."""

    shape: Tuple[int, int]
    pixel_size_nm: float
    mode: str = "tem"
    voltage_kV: float = 80.0
    #: Zero padding around the image for the propagation [px].
    padding: int = 100
    #: "+z": the beam travels from small to large z.
    beam_direction: str = "+z"
    #: Aberration coefficients, polar (C, phi) or cartesian (a, b) - the
    #: optics module converts either way.
    aberrations: Dict[str, float] = field(default_factory=dict)
    #: Envelope parameters (deltaE, Cc_um, source_size, instab_defocus_nm, ...).
    envelope: Dict[str, Any] = field(default_factory=dict)
    amp_contrast: float = 0.1
    alpha_max_mrad: float = 50.0
    #: Parametrisation used for optimisation: "cartesian" or "polar".
    ctf_parametrization: str = "cartesian"
    stem: Stem = field(default_factory=Stem)
    #: Atomic potential tables: "kirkland" or "lobato".
    potential_parametrization: str = "kirkland"
    #: Band limit of the propagator as a fraction of the Nyquist frequency
    #: (2/3 suppresses aliasing of the transmission function; 0 = off, as in
    #: all files written before the rebuild).
    band_limit: float = 2.0 / 3.0
    #: TEM image signal: "amplitude" (modulus of the wave, as before) or "intensity" (its square).
    tem_signal: str = "amplitude"

    def __post_init__(self):
        self.shape = (int(self.shape[0]), int(self.shape[1]))
        if self.potential_parametrization not in ("kirkland", "lobato"):
            raise ValueError("potential_parametrization must be 'kirkland' or 'lobato'")
        if self.tem_signal not in ("amplitude", "intensity"):
            raise ValueError("tem_signal must be 'amplitude' or 'intensity'")
        if self.mode not in IMAGING_MODES:
            raise ValueError(f"imaging mode must be one of {IMAGING_MODES}, got {self.mode!r}")


@dataclass
class Structure:
    """Initial atoms, one array per layer (same order as ``Experiment.layers``)."""

    #: (n, 2) positions in pixels, columns (y, x) as in the image array.
    positions: List[np.ndarray] = field(default_factory=list)
    #: (n,) continuous atomic number of each atom (the "intensity").
    atomic_numbers: List[np.ndarray] = field(default_factory=list)
    #: Optional (n,) weight of each atom relative to its atomic number (seeding).
    weight_factors: Optional[List[np.ndarray]] = None

    def __post_init__(self):
        self.positions = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in self.positions]
        self.atomic_numbers = [np.asarray(z, dtype=np.float64).reshape(-1) for z in self.atomic_numbers]
        if self.weight_factors is not None:
            self.weight_factors = [np.asarray(w, dtype=np.float64).reshape(-1)
                                   for w in self.weight_factors]
        for i, (p, z) in enumerate(zip(self.positions, self.atomic_numbers)):
            if len(p) != len(z):
                raise ValueError(f"layer {i}: {len(p)} positions but {len(z)} atomic numbers")

    @property
    def counts(self) -> List[int]:
        return [len(p) for p in self.positions]


@dataclass
class Experiment:
    """The complete specification of a reconstruction problem."""

    imaging: Imaging
    layers: List[Layer]
    groups: List[Group] = field(default_factory=list)
    structure: Structure = field(default_factory=Structure)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        self.layers, self.groups = resolve_layers_and_groups(self.layers, self.groups)
        if len(self.structure.positions) != len(self.layers):
            raise ValueError(f"structure has {len(self.structure.positions)} layers, "
                             f"specification has {len(self.layers)}")

    # -- convenient views ---------------------------------------------------
    @property
    def layer_names(self) -> List[str]:
        return [l.name for l in self.layers]

    @property
    def group_names(self) -> List[str]:
        return [g.name for g in self.groups]

    def group(self, name: str) -> Group:
        for g in self.groups:
            if g.name == name:
                return g
        raise KeyError(f"unknown group {name!r}, available: {self.group_names}")

    def layer_indices(self, group_name: str) -> List[int]:
        return [i for i, l in enumerate(self.layers) if l.group == group_name]


def resolve_layers_and_groups(layers: List[Layer], groups: List[Group]):
    """Validate names and add the groups that layers refer to but nobody defined.

    * layers without a group get a group of their own name,
    * names are unique and contain no ``.`` (they become dictionary keys such as
      ``groups.<name>.shift``),
    * groups without a layer are dropped.
    """

    layers, groups = list(layers), list(groups)
    if not layers:
        raise ValueError("at least one layer is required")
    names = [l.name for l in layers]
    if len(set(names)) != len(names):
        raise ValueError(f"layer names must be unique, got {names}")
    for name in names + [g.name for g in groups] + [l.group for l in layers if l.group]:
        if not name or "." in name:
            raise ValueError(f"invalid name {name!r}: must be non-empty and contain no '.'")
    for layer in layers:
        if not layer.group:
            layer.group = layer.name
    by_name = {g.name: g for g in groups}
    if len(by_name) != len(groups):
        raise ValueError("group names must be unique")
    for layer in layers:
        if layer.group not in by_name:
            by_name[layer.group] = Group(name=layer.group)
            groups.append(by_name[layer.group])
    used = {l.group for l in layers}
    return layers, [g for g in groups if g.name in used]


# ---------------------------------------------------------------------------
# Trainable state
# ---------------------------------------------------------------------------

@dataclass
class OpticsState:
    """Trained numbers of the aberration-bearing element (CTF in TEM, probe in STEM)."""

    amp_contrast: float = 0.1
    delta_E: float = 0.45
    Cc_um: float = 1000.0
    aberrations: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelState:
    """Everything the optimiser may change (Eq. 3.2 and the tables in Part II)."""

    #: Per layer, (n, 2) shifts added to the initial positions [px].
    position_shifts: List[np.ndarray] = field(default_factory=list)
    #: Per layer, (n,) effective atomic number.
    intensities: List[np.ndarray] = field(default_factory=list)
    #: Per layer, (n,) occupancy logits.
    occupancies: List[np.ndarray] = field(default_factory=list)
    #: Per group.
    shifts: Dict[str, np.ndarray] = field(default_factory=dict)
    rotations: Dict[str, float] = field(default_factory=dict)
    scales: Dict[str, np.ndarray] = field(default_factory=dict)
    contrast_factors: Dict[str, float] = field(default_factory=dict)
    #: Per group, (1, 2, Hc, Wc) strain control displacements.
    strain_control: Dict[str, np.ndarray] = field(default_factory=dict)
    #: "v3": nodes span the image area (Part I, Section 5); "legacy": earlier files
    #: (padded-image coordinates, transposed axes) - converted when the state is loaded.
    strain_convention: str = "v3"
    optics: Optional[OpticsState] = None
    #: Two numbers that correct brightness and contrast of the simulation.
    normalisation: Optional[np.ndarray] = None
    schema_version: int = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------

@dataclass
class TrainParam:
    enabled: bool = False
    lr: float = 1e-4


@dataclass
class GroupTrain:
    shift: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-3))
    rotation: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-5))
    scale: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))
    strain_field: TrainParam = field(default_factory=lambda: TrainParam(False, 0.1))
    #: Contrast factor and occupancy scale the same thing: release only one.
    contrast: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-3))


@dataclass
class LayerTrain:
    free_positions: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-3))
    intensities: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))
    occupancies: TrainParam = field(default_factory=lambda: TrainParam(False, 0.1))


@dataclass
class LossWeights:
    """Weights of the data-fidelity terms (see Part II, Section 1)."""

    var: float = 0.5
    local_var_100: float = 0.0
    mse: float = 0.0
    ncc: float = 0.5
    local_ncc_100: float = 0.0
    ssim: float = 0.0
    l1: float = 0.1


@dataclass
class Regularisation:
    """Weights and ranges of the priors (Part II, Section 3)."""

    occ_entropy_a: float = 1.0
    position_shifts_a: float = 1.0
    #: Pairwise repulsion: weight, and distance [Angstrom] below which it acts.
    repulsion_a: float = 0.0
    repulsion_distance_A: float = 1.0
    #: Elastic energy of the strain field: weight and Lame parameters.
    strain_energy_a: float = 0.0
    strain_energy_lambda: float = 1.0
    strain_energy_mu: float = 1.0


def _default_aberration_flags() -> Dict[str, bool]:
    return {k: True for k in ("C10", "C12", "C23", "C21", "C30", "C34", "C45")}


def _default_envelope_flags() -> Dict[str, bool]:
    return {"source_size": False, "Cc": True, "deltaE": True, "instab_defocus_nm": False}


@dataclass
class TrainSettings:
    """What is trained, how it is weighted, and where the image counts."""

    groups: Dict[str, GroupTrain] = field(default_factory=dict)
    layers: Dict[str, LayerTrain] = field(default_factory=dict)

    amp_contrast: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-5))
    #: Aberrations including their azimuth angles (C12 and phi12 belong together).
    aberrations: TrainParam = field(default_factory=lambda: TrainParam(True, 1e-5))
    envelope: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-5))
    normalisation: TrainParam = field(default_factory=lambda: TrainParam(True, 1e-5))
    aberration_flags: Dict[str, bool] = field(default_factory=_default_aberration_flags)
    envelope_flags: Dict[str, bool] = field(default_factory=_default_envelope_flags)

    loss: LossWeights = field(default_factory=LossWeights)
    regularisation: Regularisation = field(default_factory=Regularisation)

    #: Group that is currently being worked on (mask, target, isolation); None = all.
    active_group: Optional[str] = None

    #: Bulk arrays, stored in .npz next to the JSON.
    masks: Dict[str, np.ndarray] = field(default_factory=dict, metadata={"external": True})
    targets: Dict[str, np.ndarray] = field(default_factory=dict, metadata={"external": True})

    schema_version: int = SCHEMA_VERSION

    #: Names of the global (non group/layer) trainable switches.
    GLOBAL_PARAMS = ("amp_contrast", "aberrations", "envelope", "normalisation")

    # -- access --------------------------------------------------------------
    def group(self, name: str) -> GroupTrain:
        return self.groups.setdefault(name, GroupTrain())

    def layer(self, name: str) -> LayerTrain:
        return self.layers.setdefault(name, LayerTrain())

    def ensure_for(self, experiment: Experiment) -> "TrainSettings":
        for g in experiment.groups:
            self.group(g.name)
        for l in experiment.layers:
            self.layer(l.name)
        if self.active_group is not None and self.active_group not in self.groups:
            self.active_group = None
        return self

    def params(self) -> Dict[str, TrainParam]:
        """All switches by key: ``aberrations``, ``groups.<g>.shift``, ``layers.<l>.occupancies`` ..."""

        out = {name: getattr(self, name) for name in self.GLOBAL_PARAMS}
        for gname, gt in self.groups.items():
            for f in fields(gt):
                out[f"groups.{gname}.{f.name}"] = getattr(gt, f.name)
        for lname, lt in self.layers.items():
            for f in fields(lt):
                out[f"layers.{lname}.{f.name}"] = getattr(lt, f.name)
        return out

    def param(self, key: str) -> Optional[TrainParam]:
        return self.params().get(key)

    def enabled(self, key: str) -> bool:
        p = self.param(key)
        return bool(p is not None and p.enabled)

    def group_enabled(self, group: str, attr: str) -> bool:
        gt = self.groups.get(group)
        return bool(gt is not None and getattr(gt, attr).enabled)

    def layer_enabled(self, layer: str, attr: str) -> bool:
        lt = self.layers.get(layer)
        return bool(lt is not None and getattr(lt, attr).enabled)

    def mask_for(self, group: Optional[str]):
        return None if group is None else self.masks.get(group)

    def target_for(self, group: Optional[str]):
        return None if group is None else self.targets.get(group)


# ---------------------------------------------------------------------------
# Dataset description (the ImageJ plug-in and the data-folder convention)
# ---------------------------------------------------------------------------

@dataclass
class LatticeSpec:
    """Recipe for an initial atom lattice (executed by :mod:`temsim.physics.structure`)."""

    name: str = "lattice"
    type: str = "graphene"
    #: Lattice constants [Angstrom] and the angle between the lattice vectors.
    a: float = 2.46
    b: Optional[float] = None
    gamma_deg: float = 120.0
    #: ``type == "mx2"``: chemical formula; ``type == "file"``: structure file.
    formula: str = "WS2"
    file: Optional[str] = None
    #: ``type == "custom"``: rows ``(element, frac_a, frac_b, z_nm)``.
    basis: List[Any] = field(default_factory=list)
    rotation_deg: float = 0.0
    offset_px: Tuple[float, float] = (0.0, 0.0)
    z_offset_nm: float = 0.0


@dataclass
class LayerSource:
    """Which layer of which lattice fills a given experiment layer."""

    layer: str
    lattice: Optional[str] = None
    lattice_layer: Optional[int] = None


@dataclass
class InitTrain:
    """The ``train`` block of an ``init.json``: what to release at the start."""

    enable: List[str] = field(default_factory=list)
    learning_rates: Dict[str, float] = field(default_factory=dict)
    losses: Dict[str, float] = field(default_factory=dict)
    active_group: Optional[str] = None


@dataclass
class DatasetSpec:
    """Everything a data folder says about an experiment, before atoms exist.

    ``shape`` is optional because a bare folder only reveals it through the target
    image.  ``imaging`` holds the raw ``imaging`` block (mode, aberrations,
    envelope, stem ...); :meth:`to_imaging` merges it with the shape.
    """

    pixel_size_nm: float
    voltage_kV: float = 80.0
    shape: Optional[Tuple[int, int]] = None
    padding: int = 100
    imaging: Dict[str, Any] = field(default_factory=dict)
    layers: List[Layer] = field(default_factory=list)
    groups: List[Group] = field(default_factory=list)
    lattices: List[LatticeSpec] = field(default_factory=list)
    layer_sources: List[LayerSource] = field(default_factory=list)
    train: InitTrain = field(default_factory=InitTrain)
    target_file: Optional[str] = None
    target_zero_handling: str = "median"
    config_dir: str = "config"
    source: Dict[str, Any] = field(default_factory=dict)
    version: int = 2

    def to_imaging(self, shape: Optional[Tuple[int, int]] = None) -> Imaging:
        shape = shape or self.shape
        if shape is None:
            raise ValueError("image shape unknown: neither in the spec nor given")
        known = set(Imaging.__dataclass_fields__) - {"shape", "pixel_size_nm", "voltage_kV", "padding", "stem"}
        im = {k: v for k, v in self.imaging.items() if k in known}
        stem = dict(self.imaging.get("stem") or {})
        if "detector_mrad" in stem:
            stem["detector_mrad"] = tuple(stem["detector_mrad"])
        stem = {k: v for k, v in stem.items() if k in Stem.__dataclass_fields__}
        return Imaging(shape=shape, pixel_size_nm=self.pixel_size_nm, voltage_kV=self.voltage_kV,
                       padding=self.padding, stem=Stem(**stem), **im)
