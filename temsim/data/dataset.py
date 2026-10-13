"""Part III, Section 2: the data-folder convention.

Theory
======

A measurement is a folder.  Convention (every item optional unless stated)::

    <folder>/
        target.tif | target.npy       the measured image (required)
        mask_<group>.tif | <group>_mask.tif | mask.tif
                                      region in which atoms of <group> are expected
        init.json | init_random.json  setup (pixel size, voltage, mode, lattices ...)
        <config_dir>/                 run directories written by the programs

The image is dimensionless; Part II normalises it (Eq. 5.1), so the file's grey
values are irrelevant except for one defect: detectors write exact zeros for
pixels that were never read.  Those would pull the minimum of the image to zero and
distort the normalisation, therefore ``target_zero_handling == "median"`` (the
default, as in the ImageJ plug-in) replaces exact zeros by the image median.

``init.json`` is written by the ImageJ plug-in (``version`` 2) and, in reduced
form, by the headless seeding command (``init_random.json``).  Length units in it
follow the plug-in's conventions (shift in pixels, rotation in *degrees*); this
module converts to the internal convention of :mod:`temsim.data.schema` (radians).
Turning lattice recipes into atoms is physics and lives in
:mod:`temsim.physics.structure`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .legacy import LOSS_KEY_ALIASES, TRAIN_KEY_ALIASES
from .schema import (DatasetSpec, Group, InitTrain, Layer, LatticeSpec, LayerSource,
                     TrainSettings)

TARGET_NAMES = ("target.tif", "target.tiff", "target.npy")
INIT_NAMES = ("init.json", "init_random.json")
DEFAULT_GROUP = "Gr"


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def read_image(path: str) -> np.ndarray:
    """``.npy`` or ``.tif`` -> array (as stored)."""

    if path.lower().endswith(".npy"):
        return np.load(path)
    import tifffile                                   # imported lazily: optional dependency
    return tifffile.imread(path)


def prepare_target(image: np.ndarray, zero_handling: str = "median") -> np.ndarray:
    """Float32 copy; exact zeros replaced by the median if requested."""

    image = np.asarray(image, dtype=np.float32)
    if zero_handling == "median":
        zeros = image == 0.0
        if zeros.any():
            image = image.copy()
            image[zeros] = np.median(image)
    return image


def group_name_from_mask(filename: str) -> str:
    """``mask_Gr.tif`` -> ``Gr``, ``gr_mask.tif`` -> ``gr``, ``mask.tif`` -> default."""

    stem = os.path.splitext(os.path.basename(filename))[0]
    if stem.startswith("mask_"):
        return stem[len("mask_"):]
    if stem.endswith("_mask"):
        return stem[:-len("_mask")]
    return DEFAULT_GROUP


def find_masks(folder: str) -> List[str]:
    """Mask files of a folder, sorted, ``.tif``/``.tiff``/``.npy`` only."""

    found = []
    for name in sorted(os.listdir(folder)):
        low = name.lower()
        if not low.endswith((".tif", ".tiff", ".npy")):
            continue
        stem = os.path.splitext(low)[0]
        if stem.startswith("mask") or stem.endswith("_mask"):
            found.append(os.path.join(folder, name))
    return found


def find_target(folder: str) -> Optional[str]:
    for name in TARGET_NAMES:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------------
# init.json
# ---------------------------------------------------------------------------

def _group_from_init(entry: dict) -> Group:
    return Group(
        name=entry["name"],
        control_points=list(entry.get("control_points", [10, 10])),
        max_shift=None if entry.get("max_shift_px") is None else float(entry["max_shift_px"]),
        max_rotation=None if entry.get("max_rotation_deg") is None
        else float(np.deg2rad(entry["max_rotation_deg"])),
        init_shift=list(entry.get("init_shift_px", [0.0, 0.0])),
        init_rotation=float(np.deg2rad(entry.get("init_rotation_deg", 0.0))),
        init_scale=list(entry.get("init_scale", [1.0, 1.0])),
        init_contrast_factor=float(entry.get("init_contrast_factor",
                                             entry.get("init_potential_factor", 1.0))),
        mask_file=entry.get("mask_file"), target_file=entry.get("target_file"),
        isolate_when_active=bool(entry.get("isolate_when_active", False)))


def _layer_from_init(entry: dict) -> Layer:
    from .schema import TYPICAL_DEBYE_WALLER_U
    element = entry.get("element") or "C"
    return Layer(
        name=entry["name"], element=element, z_nm=float(entry.get("z_nm", 0.0)),
        group=entry.get("group") or entry["name"],
        init_occupancy=float(entry.get("init_occupancy", 2.0)),
        occ_entropy=bool(entry.get("occ_entropy", True)),
        elements=list(entry.get("elements", [])),
        debye_waller_u=float(entry.get("debye_waller_u", TYPICAL_DEBYE_WALLER_U.get(element, 0.0))))


def _lattice_from_init(entry: dict) -> LatticeSpec:
    spec = LatticeSpec(name=entry.get("name", "lattice"))
    for key in ("type", "formula", "file"):
        if entry.get(key) is not None:
            setattr(spec, key, entry[key])
    for key in ("a", "gamma_deg", "rotation_deg", "z_offset_nm"):
        if entry.get(key) is not None:
            setattr(spec, key, float(entry[key]))
    spec.b = None if entry.get("b") is None else float(entry["b"])
    spec.basis = [list(b) for b in entry.get("basis", [])]
    spec.offset_px = tuple(float(v) for v in entry.get("offset_px", (0.0, 0.0)))
    return spec


def train_key(key: str) -> str:
    """Switch name written by the plug-in (old names allowed) -> key of ``TrainSettings.params()``."""

    return TRAIN_KEY_ALIASES.get(key, key)


def apply_init_train(settings: TrainSettings, init: InitTrain) -> List[str]:
    """Apply the ``train`` block of an init file; returns the keys that were not understood."""

    unknown = []
    for key, value in init.losses.items():
        name = LOSS_KEY_ALIASES.get(key, key)
        if hasattr(settings.loss, name):
            setattr(settings.loss, name, float(value))
        else:
            unknown.append(key)
    for key in init.enable:
        param = settings.param(train_key(key))
        if param is None:
            unknown.append(key)
        else:
            param.enabled = True
    for key, lr in init.learning_rates.items():
        param = settings.param(train_key(key))
        if param is None:
            unknown.append(key)
        else:
            param.lr = float(lr)
    if init.active_group is not None:
        settings.active_group = init.active_group
    return unknown


def spec_from_init(init: dict) -> DatasetSpec:
    """Parsed ``init.json`` / ``init_random.json`` -> :class:`DatasetSpec`."""

    version = init.get("version", 1)
    if version > 2:
        raise ValueError(f"init.json version {version} is newer than this program understands (2)")

    groups = [_group_from_init(g) for g in init.get("groups", [])]
    layers = [_layer_from_init(l) for l in init.get("layers", [])]

    entries = init.get("lattices")
    if entries is None:
        single = dict(init.get("lattice") or {})
        if single:
            single.setdefault("name", groups[0].name if groups else "lattice")
            entries = [single]
    lattices = [_lattice_from_init(e) for e in entries or []]
    default_lattice = lattices[0].name if lattices else None

    counters: Dict[str, int] = {}
    sources = []
    for raw in init.get("layers", []):
        lat = raw.get("lattice", default_lattice)
        index = raw.get("lattice_layer")
        if index is None and lat is not None:
            index = counters.get(lat, 0)
            counters[lat] = index + 1
        sources.append(LayerSource(layer=raw["name"], lattice=lat, lattice_layer=index))

    t = init.get("train") or {}
    train = InitTrain(enable=list(t.get("enable", [])),
                      learning_rates={k: float(v) for k, v in (t.get("learning_rates") or {}).items()},
                      losses={k: float(v) for k, v in (t.get("losses") or {}).items()},
                      active_group=t.get("active_group"))

    shape = init.get("shape")
    return DatasetSpec(
        pixel_size_nm=float(init["pixel_size_nm"]), voltage_kV=float(init.get("voltage_kV", 80.0)),
        shape=None if shape is None else (int(shape[0]), int(shape[1])),
        padding=int(init.get("padding", 100)), imaging=dict(init.get("imaging") or {}),
        layers=layers, groups=groups, lattices=lattices, layer_sources=sources, train=train,
        target_file=init.get("target_file"),
        target_zero_handling=init.get("target_zero_handling", "median"),
        config_dir=init.get("config_dir", "config"), source=dict(init.get("source") or {}),
        version=int(version))


# ---------------------------------------------------------------------------
# The folder as a whole
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """A loaded data folder."""

    folder: str
    spec: Optional[DatasetSpec]
    #: Raw target (zero handling applied), not yet normalised.
    target: np.ndarray
    #: Boolean masks by group name.
    masks: Dict[str, np.ndarray] = field(default_factory=dict)
    #: Group targets (when ``group.target_file`` is set).
    group_targets: Dict[str, np.ndarray] = field(default_factory=dict)
    init_path: Optional[str] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.target.shape)


def load_dataset(path: str) -> Dataset:
    """Load a data folder or an ``init.json`` path.

    The spec is ``None`` when no init file exists; the caller then has to supply
    pixel size and voltage.
    """

    if os.path.isdir(path):
        folder, init_path = path, None
        for name in INIT_NAMES:
            if os.path.exists(os.path.join(folder, name)):
                init_path = os.path.join(folder, name)
                break
    elif path.lower().endswith(".json"):
        folder, init_path = os.path.dirname(os.path.abspath(path)), path
    else:
        raise ValueError(f"{path} is neither a folder nor an init file")

    spec = None
    if init_path is not None:
        with open(init_path, "r", encoding="utf-8") as f:
            spec = spec_from_init(json.load(f))

    zero = spec.target_zero_handling if spec else "median"
    target_path = os.path.join(folder, spec.target_file) if spec and spec.target_file else find_target(folder)
    if target_path is None or not os.path.exists(target_path):
        raise FileNotFoundError(f"no target image in {folder} (expected one of {TARGET_NAMES})")
    target = prepare_target(read_image(target_path), zero)

    masks: Dict[str, np.ndarray] = {}
    group_targets: Dict[str, np.ndarray] = {}
    if spec:
        for g in spec.groups:
            if g.mask_file:
                masks[g.name] = np.asarray(read_image(os.path.join(folder, g.mask_file))) > 0
            if g.target_file:
                group_targets[g.name] = prepare_target(read_image(os.path.join(folder, g.target_file)))
    if not masks:
        for mask_path in find_masks(folder):
            masks.setdefault(group_name_from_mask(mask_path), np.asarray(read_image(mask_path)) > 0)

    for name, m in masks.items():
        if m.shape != target.shape:
            raise ValueError(f"mask '{name}' has shape {m.shape}, target has {target.shape}")
    if spec and spec.shape is not None and tuple(spec.shape) != tuple(target.shape):
        raise ValueError(f"init file says shape {tuple(spec.shape)}, target has {tuple(target.shape)}")
    return Dataset(folder=folder, spec=spec, target=target, masks=masks,
                   group_targets=group_targets, init_path=init_path)
