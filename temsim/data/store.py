"""Part III, Section 3: persistence and versioning.

Theory
======

A *run directory* is the unit of reproducibility.  It holds exactly the objects of
the data model (:mod:`temsim.data.schema`)::

    experiment.json   Experiment    the specification, incl. initial atoms
    train.json        TrainSettings policy (without bulk arrays)
    state.json        ModelState    trained numbers (optional)
    masks.npz         masks by group name          (optional)
    targets.npz       group targets by group name  (optional)
    target.npy        the measured image           (optional)

Together they determine a run up to the random seed (Section 4: seeds are part of
``train.json`` once a seeding run has used them).

Versioning follows one rule.  Every JSON file carries ``schema_version``; files of
the current version are decoded by the generic codec, everything else is routed to
:mod:`temsim.data.legacy`.  *Legacy detection is structural*, not by file name,
because the old programs wrote files without any version marker:

* experiment: ``"imaging"`` key present -> current, ``"init_positions"`` -> legacy;
* train settings: nested ``"loss"`` object -> current, otherwise legacy;
* model state: ``"position_shifts"`` at the top level -> current,
  ``"potential_parameters"`` -> legacy.

For reading, the file names of the old programs are also understood
(``config.json``, ``train_method.json``, ``area_masks.npz/.npy``,
``group_targets.npz``, ``target_gr.npy``); on writing only the new names are used.
Writing is atomic (temporary file, then rename) so that an interrupted save never
leaves half a file behind.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import codec, legacy
from .schema import SCHEMA_VERSION, Experiment, ModelState, TrainSettings

EXPERIMENT_FILES = ("experiment.json", "config.json")
TRAIN_FILES = ("train.json", "train_method.json")
STATE_FILES = ("state.json",)
MASK_FILES = ("masks.npz", "area_masks.npz")
TARGET_FILES = ("targets.npz", "group_targets.npz")
IMAGE_FILES = ("target.npy",)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_existing(directory: str, names) -> Optional[str]:
    for name in names:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None


def _check_version(raw: dict, what: str) -> None:
    version = raw.get("schema_version", 0)
    if version > SCHEMA_VERSION:
        raise ValueError(f"{what} has schema_version {version}, this program reads up to "
                         f"{SCHEMA_VERSION}; please update")


# ---------------------------------------------------------------------------
# single objects (dict level)
# ---------------------------------------------------------------------------

def experiment_from_dict(raw: dict) -> Experiment:
    raw = legacy.unwrap(raw)
    if "imaging" in raw:
        _check_version(raw, "experiment")
        return codec.decode(Experiment, raw)
    if "init_positions" in raw:
        return legacy.experiment_from_legacy(raw)
    raise ValueError("not an experiment file: neither 'imaging' nor 'init_positions' found")


def train_from_dict(raw: dict) -> TrainSettings:
    raw = legacy.unwrap(raw)
    if isinstance(raw.get("loss"), dict):
        _check_version(raw, "train settings")
        return codec.decode(TrainSettings, raw)
    return legacy.train_settings_from_legacy(raw)


def state_from_dict(raw: dict, experiment: Experiment) -> ModelState:
    raw = legacy.unwrap(raw)
    if "potential_parameters" in raw or "ctf_parameters" in raw:
        return legacy.state_from_legacy(raw, experiment)
    _check_version(raw, "model state")
    return codec.decode(ModelState, raw)


# ---------------------------------------------------------------------------
# single objects (file level)
# ---------------------------------------------------------------------------

def save_experiment(path: str, experiment: Experiment) -> None:
    _write_json(path, codec.encode(experiment))


def load_experiment(path: str) -> Experiment:
    return experiment_from_dict(_read_json(path))


def save_train(path: str, settings: TrainSettings) -> None:
    _write_json(path, codec.encode(settings))


def load_train(path: str) -> TrainSettings:
    return train_from_dict(_read_json(path))


def save_state(path: str, state: ModelState) -> None:
    _write_json(path, codec.encode(state))


def load_state(path: str, experiment: Experiment) -> ModelState:
    return state_from_dict(_read_json(path), experiment)


def save_arrays(path: str, arrays: Dict[str, np.ndarray]) -> None:
    tmp = path + ".tmp.npz"
    np.savez(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
    os.replace(tmp, path)


def load_arrays(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# run directories
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """A run directory in memory."""

    experiment: Experiment
    train: TrainSettings
    state: Optional[ModelState] = None
    #: Measured image used for the run, if stored with it.
    target: Optional[np.ndarray] = None
    #: Remarks about legacy conversions that happened while loading.
    notes: List[str] = field(default_factory=list)


def save_run(directory: str, run: Run) -> None:
    os.makedirs(directory, exist_ok=True)
    save_experiment(os.path.join(directory, "experiment.json"), run.experiment)
    save_train(os.path.join(directory, "train.json"), run.train)
    if run.state is not None:
        save_state(os.path.join(directory, "state.json"), run.state)
    if run.train.masks:
        save_arrays(os.path.join(directory, "masks.npz"), run.train.masks)
    if run.train.targets:
        save_arrays(os.path.join(directory, "targets.npz"), run.train.targets)
    if run.target is not None:
        np.save(os.path.join(directory, "target.npy"), run.target)


def _legacy_stacked_masks(path: str, experiment: Experiment) -> Dict[str, np.ndarray]:
    """``area_masks.npy`` is a stack in the order of the old group list (Gr, WS2)."""

    stack = np.load(path)
    if stack.ndim == 2:
        stack = stack[None]
    names = legacy.LEGACY_GROUP_NAMES
    return {names[i]: stack[i] > 0 for i in range(min(len(stack), len(names)))
            if names[i] in experiment.group_names}


def load_run(directory: str) -> Run:
    """Load a run directory of the current or any legacy layout."""

    exp_path = _first_existing(directory, EXPERIMENT_FILES)
    if exp_path is None:
        raise FileNotFoundError(f"no experiment.json / config.json in {directory}")
    experiment = load_experiment(exp_path)

    notes: List[str] = []
    train_path = _first_existing(directory, TRAIN_FILES)
    train = load_train(train_path) if train_path else TrainSettings()
    train.ensure_for(experiment)

    mask_path = _first_existing(directory, MASK_FILES)
    if mask_path:
        train.masks = {k: np.asarray(v) > 0 for k, v in load_arrays(mask_path).items()}
    elif os.path.exists(os.path.join(directory, "area_masks.npy")):
        train.masks = _legacy_stacked_masks(os.path.join(directory, "area_masks.npy"), experiment)
        notes.append("masks read from legacy area_masks.npy")
    target_path = _first_existing(directory, TARGET_FILES)
    if target_path:
        train.targets = {k: v.astype(np.float32) for k, v in load_arrays(target_path).items()}
    elif os.path.exists(os.path.join(directory, "target_gr.npy")):
        first = legacy.LEGACY_GROUP_NAMES[0]
        train.targets = {first: np.load(os.path.join(directory, "target_gr.npy")).astype(np.float32)}
        notes.append("group target read from legacy target_gr.npy")

    state_path = _first_existing(directory, STATE_FILES)
    state = load_state(state_path, experiment) if state_path else None

    image_path = _first_existing(directory, IMAGE_FILES)
    target = np.load(image_path).astype(np.float32) if image_path else None
    return Run(experiment=experiment, train=train, state=state, target=target, notes=notes)
