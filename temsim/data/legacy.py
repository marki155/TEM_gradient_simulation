"""Part III, Section 3: reading the formats of earlier versions.

Theory
======

A file format is a contract that outlives the code that wrote it.  Four formats
exist in the wild, and all of them must stay readable:

========================  ======================================================
File                      Generations
========================  ======================================================
``config.json``           (a) fixed four layers ``Gr/S1/W/S2`` with ``z_positions_nm``
                          and ``control_points_*``; (b) ``layers``/``groups``;
                          each either written with ``asdict`` or wrapped as
                          ``{"__type__": "dataclass", ...}``.
``train_method.json``     (a) flat flags ``graphen_shift`` ...; (b) groups/layers
                          dictionaries; plain or wrapped.
parameter dumps           ``scale_Gr``, ``shift_WS2``, ``graphene_potential_factor``
(``best.json`` ...)       or the group dictionaries; polar or cartesian aberrations.
``init.json``             written by the ImageJ plug-in (input, version 1 and 2).
========================  ======================================================

Strategy (Eq. 3.3): a migration is a pure function from the raw JSON value to the
current schema, ``new = M_k(M_{k-1}(...(old)))``.  Nothing in the rest of the code
knows an old name.  Every function here has a test that loads real files.

Aberration coefficients are passed through unchanged: converting between the polar
and the cartesian form is optics, not data management (see
:mod:`temsim.physics.optics`).  Likewise, resampling a strain control grid to a
different resolution happens where the grid is applied.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .schema import (
    Experiment, Group, GroupTrain, Imaging, Layer, LayerTrain, ModelState,
    OpticsState, Stem, Structure, TrainParam, TrainSettings,
)

log = logging.getLogger("temsim.data")

# ---------------------------------------------------------------------------
# Generic unwrapping of  {"__type__": ...}  envelopes
# ---------------------------------------------------------------------------


def unwrap(value: Any) -> Any:
    """Strip the ``__type__`` envelopes written by the old serialisers.

    ``dataclass`` -> its ``data`` object, ``ndarray`` and ``tensor`` -> nested list,
    ``tuple`` -> list.  Plain values pass through.
    """

    if isinstance(value, dict):
        kind = value.get("__type__")
        if kind in ("dataclass", "ndarray", "tensor", "tuple"):
            return unwrap(value["data"])
        return {k: unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unwrap(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# config.json
# ---------------------------------------------------------------------------

LEGACY_LAYER_NAMES: Tuple[str, ...] = ("Gr", "S1", "W", "S2")
LEGACY_LAYER_ELEMENTS: Tuple[str, ...] = ("C", "S", "W", "S")
LEGACY_LAYER_GROUP: Dict[str, str] = {"Gr": "Gr", "S1": "WS2", "W": "WS2", "S2": "WS2"}
LEGACY_GROUP_NAMES: Tuple[str, ...] = ("Gr", "WS2")
LEGACY_GROUP_INIT = {
    "Gr": dict(init_scale=[1.12, 1.14], init_rotation=0.025, init_shift=[-9.5, -11.0]),
    "WS2": dict(init_scale=[1.05, 1.09], init_rotation=0.0, init_shift=[15.9, -5.0]),
}
LEGACY_INIT_OCCUPANCY = {"Gr": 100.0, "S1": 2.0, "W": 2.0, "S2": 2.0}

_GROUP_RENAMES = {"init_potential_factor": "init_contrast_factor"}


def legacy_layers_and_groups(z_positions_nm, control_points_graphen=(10, 10),
                             control_points_ws2=(10, 10), max_shifts=None,
                             max_rotations=None, n_layers=None):
    """The original fixed constellation ``Gr = [Gr]``, ``WS2 = [S1, W, S2]``."""

    z = list(z_positions_nm) if z_positions_nm is not None else []
    n_layers = len(z) if n_layers is None else n_layers
    z += [0.0] * (n_layers - len(z))

    if n_layers != len(LEGACY_LAYER_NAMES):
        group = Group(name="all", control_points=list(control_points_ws2),
                      max_shift=None if max_shifts is None else float(max_shifts[-1]),
                      max_rotation=None if max_rotations is None else float(max_rotations[-1]))
        return ([Layer(name=f"L{i}", element="C", z_nm=float(z[i]), group="all")
                 for i in range(n_layers)], [group])

    groups = []
    for gi, gname in enumerate(LEGACY_GROUP_NAMES):
        groups.append(Group(
            name=gname,
            control_points=list(control_points_graphen if gname == "Gr" else control_points_ws2),
            max_shift=None if max_shifts is None else float(max_shifts[gi]),
            max_rotation=None if max_rotations is None else float(max_rotations[gi]),
            isolate_when_active=(gname == "Gr"),
            **LEGACY_GROUP_INIT[gname]))
    layers = [Layer(name=name, element=LEGACY_LAYER_ELEMENTS[i], z_nm=float(z[i]),
                    group=LEGACY_LAYER_GROUP[name],
                    init_occupancy=LEGACY_INIT_OCCUPANCY[name],
                    occ_entropy=(name != "Gr"))
              for i, name in enumerate(LEGACY_LAYER_NAMES)]
    return layers, groups


def _dataclass_from(cls, raw: dict, renames: Optional[dict] = None):
    raw = dict(raw)
    for old, new in (renames or {}).items():
        if old in raw:
            raw.setdefault(new, raw.pop(old))
    known = set(cls.__dataclass_fields__)
    dropped = sorted(set(raw) - known)
    if dropped:
        log.info("%s: ignoring unknown fields %s", cls.__name__, dropped)
    return cls(**{k: v for k, v in raw.items() if k in known})


def experiment_from_legacy(raw: dict) -> Experiment:
    """``config.json`` (any generation) -> :class:`Experiment`."""

    c = unwrap(raw)
    positions = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in c["init_positions"]]
    atomic = [np.asarray(z, dtype=np.float64).reshape(-1) for z in c["init_intensities"]]
    weights = c.get("init_weight_factors")

    if c.get("layers"):
        layers = [_dataclass_from(Layer, l) for l in c["layers"]]
        groups = [_dataclass_from(Group, g, _GROUP_RENAMES) for g in c.get("groups") or []]
    else:
        layers, groups = legacy_layers_and_groups(
            c.get("z_positions_nm") or [],
            c.get("control_points_graphen") or (10, 10),
            c.get("control_points_ws2") or (10, 10),
            c.get("max_shifts"), c.get("max_rotations"), n_layers=len(positions))

    stem = Stem(
        convergence_mrad=float(c.get("stem_convergence_mrad", 25.0)),
        detector_mrad=tuple(c.get("stem_detector_mrad") or (60.0, 200.0)),
        source_size_A=float(c.get("stem_source_size_A", 0.5)),
        aberrations=dict(c["init_probe_aberrations"]) if c.get("init_probe_aberrations") else None)

    imaging = Imaging(
        shape=tuple(c["shape"]), pixel_size_nm=float(c["pixel_size_nm"]),
        mode=c.get("imaging_mode", "tem"), voltage_kV=float(c["voltage_kV"]),
        padding=int(c.get("padding", 100)), beam_direction=c.get("beam_direction", "+z"),
        aberrations=dict(c.get("init_aberrations") or {}),
        envelope=dict(c.get("init_ctf_env_params") or {}),
        amp_contrast=float(c.get("init_amp_contrast", 0.1)),
        alpha_max_mrad=float(c.get("ctf_alpha_max_mrad", 50.0)),
        ctf_parametrization=c.get("ctf_parametrization", "cartesian"), stem=stem,
        potential_parametrization=c.get("potential_parametrization", "kirkland"),
        band_limit=float(c.get("band_limit", 0.0)), tem_signal=c.get("tem_signal", "amplitude"))

    return Experiment(imaging=imaging, layers=layers, groups=groups,
                      structure=Structure(positions, atomic, weights))


# ---------------------------------------------------------------------------
# train_method.json
# ---------------------------------------------------------------------------

#: switch names of the old TrainMethod -> TrainSettings key
TRAIN_KEY_ALIASES = {
    "train_amplitude_contrast": "amp_contrast",
    "train_ctf_aberrations": "aberrations",
    "train_env_parameters": "envelope",
    "train_normalisation_parameters": "normalisation",
}

#: old flat loss name -> field of :class:`LossWeights`
LOSS_KEY_ALIASES = {f"loss_{n}": n for n in
                    ("var", "local_var_100", "mse", "ncc", "local_ncc_100", "ssim", "l1")}
_LOSS_NAMES = LOSS_KEY_ALIASES
_REG_NAMES = {
    "occ_entropy_a": "occ_entropy_a",
    "position_shifts_regularization_a": "position_shifts_a",
    "repulsion_a": "repulsion_a", "repulsion_distance_A": "repulsion_distance_A",
    "strain_energy_a": "strain_energy_a", "strain_energy_lambda": "strain_energy_lambda",
    "strain_energy_mu": "strain_energy_mu",
}

_FLAT_GROUP_FIELDS = {
    "graphen_shift": ("Gr", "shift"), "graphen_rotation": ("Gr", "rotation"),
    "graphen_scale": ("Gr", "scale"), "Gr_strain_field": ("Gr", "strain_field"),
    "WS2_shift": ("WS2", "shift"), "WS2_rotation": ("WS2", "rotation"),
    "WS2_scale": ("WS2", "scale"), "WS2_strain_field": ("WS2", "strain_field"),
}
_FLAT_LAYER_FIELDS = {
    f"train_{'graphen' if n == 'Gr' else n}_{what}": (n, what)
    for n in LEGACY_LAYER_NAMES for what in ("intensities", "occupancies")
}
_FLAT_GROUP_TO_LAYER = {"graphen_free_positions": "Gr", "WS2_free_positions": "WS2"}


def _train_param(raw) -> TrainParam:
    raw = raw or {}
    return TrainParam(enabled=bool(raw.get("enabled", False)), lr=float(raw.get("lr", 1e-4)))


def _flat_to_nested(data: dict) -> dict:
    """Very old flat ``graphen_shift`` style flags -> ``groups``/``layers`` dictionaries."""

    data = dict(data)
    groups: Dict[str, dict] = dict(data.get("groups") or {})
    layers: Dict[str, dict] = dict(data.get("layers") or {})
    for old, (g, attr) in _FLAT_GROUP_FIELDS.items():
        if old in data:
            groups.setdefault(g, {})[attr] = data.pop(old)
    for old, (l, attr) in _FLAT_LAYER_FIELDS.items():
        if old in data:
            layers.setdefault(l, {})[attr] = data.pop(old)
    for old, g in _FLAT_GROUP_TO_LAYER.items():
        if old in data:
            value = data.pop(old)
            for l in LEGACY_LAYER_NAMES:
                if LEGACY_LAYER_GROUP[l] == g:
                    layers.setdefault(l, {})["free_positions"] = value
    area = data.pop("only_train_small_area", None)
    if area is not None and 0 <= int(area) < len(LEGACY_GROUP_NAMES):
        data["active_group"] = LEGACY_GROUP_NAMES[int(area)]
    data["groups"], data["layers"] = groups, layers
    return data


def train_settings_from_legacy(raw: dict) -> TrainSettings:
    """``train_method.json`` (any generation) -> :class:`TrainSettings`."""

    d = _flat_to_nested(unwrap(raw))
    ts = TrainSettings()

    for gname, gp in d["groups"].items():
        g = GroupTrain()
        for attr, value in (gp or {}).items():
            if hasattr(g, attr):
                setattr(g, attr, _train_param(value))
        ts.groups[gname] = g
    for lname, lp in d["layers"].items():
        l = LayerTrain()
        for attr, value in (lp or {}).items():
            if hasattr(l, attr):
                setattr(l, attr, _train_param(value))
        ts.layers[lname] = l

    for old, new in TRAIN_KEY_ALIASES.items():
        if old in d:
            setattr(ts, new, _train_param(d[old]))
    for old, new in _LOSS_NAMES.items():
        if old in d:
            setattr(ts.loss, new, float(d[old]))
    for old, new in _REG_NAMES.items():
        if old in d:
            setattr(ts.regularisation, new, float(d[old]))
    if isinstance(d.get("train_aberrations"), dict):
        ts.aberration_flags = {k: bool(v) for k, v in d["train_aberrations"].items()}
    if isinstance(d.get("train_env_params"), dict):
        ts.envelope_flags = {k: bool(v) for k, v in d["train_env_params"].items()}
    ts.active_group = d.get("active_group") or None
    return ts


# ---------------------------------------------------------------------------
# parameter dumps (best.json ...)
# ---------------------------------------------------------------------------

_FLAT_POTENTIAL_FIELDS = {
    "scale_Gr": ("scales", "Gr"), "rotation_Gr": ("rotations", "Gr"), "shift_Gr": ("shifts", "Gr"),
    "scale_WS2": ("scales", "WS2"), "rotation_WS2": ("rotations", "WS2"),
    "shift_WS2": ("shifts", "WS2"),
    "graphene_potential_factor": ("contrast_factors", "Gr"),
    "ws2_potential_factor": ("contrast_factors", "WS2"),
}


def _per_layer(value, experiment: Experiment, name: str, width: Optional[int]):
    """List with one array per layer; remap old four-layer files by layer name."""

    if not isinstance(value, list):
        return []
    names = experiment.layer_names
    if len(value) != len(names):
        remapped = []
        for n in names:
            if n in LEGACY_LAYER_NAMES and LEGACY_LAYER_NAMES.index(n) < len(value):
                remapped.append(value[LEGACY_LAYER_NAMES.index(n)])
            else:
                log.warning("parameter file: '%s' (%d layers) does not fit layers %s - ignored",
                            name, len(value), names)
                return []
        value = remapped
    return [np.asarray(v, dtype=np.float64).reshape((-1, width) if width else (-1,)) for v in value]


def state_from_legacy(raw: dict, experiment: Experiment) -> ModelState:
    """Parameter dump (``best.json`` & co.) -> :class:`ModelState`."""

    d = unwrap(raw)
    pot = dict(d.get("potential_parameters") or {})
    names = experiment.group_names
    state = ModelState()

    buckets: Dict[str, Dict[str, Any]] = {k: dict(pot.get(k) or {}) for k in
                                          ("scales", "rotations", "shifts", "contrast_factors")}
    if "potential_factors" in pot:                       # intermediate version
        buckets["contrast_factors"].update(pot["potential_factors"])
    for old, (bucket, gname) in _FLAT_POTENTIAL_FIELDS.items():
        if old in pot and gname in names:
            buckets[bucket][gname] = pot[old]
    state.scales = {k: np.asarray(v, dtype=np.float64) for k, v in buckets["scales"].items()}
    state.shifts = {k: np.asarray(v, dtype=np.float64) for k, v in buckets["shifts"].items()}
    state.rotations = {k: float(v) for k, v in buckets["rotations"].items()}
    state.contrast_factors = {k: float(v) for k, v in buckets["contrast_factors"].items()}

    strain = pot.get("strain_fields")
    if isinstance(strain, list):                         # positional: Gr, WS2
        strain = {LEGACY_GROUP_NAMES[i]: s for i, s in enumerate(strain)
                  if i < len(LEGACY_GROUP_NAMES) and LEGACY_GROUP_NAMES[i] in names}
    for gname, entry in (strain or {}).items():
        if isinstance(entry, dict) and "control" in entry:
            state.strain_control[gname] = np.asarray(entry["control"], dtype=np.float64)
            state.strain_convention = "legacy"

    state.position_shifts = _per_layer(pot.get("position_shifts"), experiment, "position_shifts", 2)
    state.intensities = _per_layer(pot.get("intensities"), experiment, "intensities", None)
    state.occupancies = _per_layer(pot.get("occupancies"), experiment, "occupancies", None)

    ctf = d.get("ctf_parameters")
    if isinstance(ctf, dict):
        state.optics = OpticsState(
            amp_contrast=float(ctf.get("amp_contrast", 0.1)),
            delta_E=float(ctf.get("deltaE", 0.45)), Cc_um=float(ctf.get("Cc_um", 1000.0)),
            aberrations={k: float(v) for k, v in (ctf.get("aberrations") or {}).items()})
    if d.get("normalisation_parameters") is not None:
        state.normalisation = np.asarray(d["normalisation_parameters"], dtype=np.float64)
    return state
