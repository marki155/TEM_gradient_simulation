from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, asdict, fields
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import json
import os

from layers import (
    LEGACY_GROUP_NAMES,
    LEGACY_LAYER_GROUP,
    LEGACY_LAYER_NAMES,
)


@dataclass
class TrainState:
    running: bool = False
    paused: bool = False
    step: int = 0
    loss: float = 0.0


@dataclass
class TrainParam:
    enabled: bool = False
    lr: float = 1e-4


@dataclass
class GroupTrainParams:
    """Trainierbare Groessen einer Struktur-Gruppe (starre Einheit)."""

    shift: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-3))
    rotation: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-5))
    scale: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))
    strain_field: TrainParam = field(default_factory=lambda: TrainParam(False, 0.1))
    #: empirischer Kontrastfaktor der Gruppe (Stobbs-Faktor). Skaliert dasselbe
    #: wie die Occupancy - immer nur eins von beidem freigeben.
    contrast: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-3))


@dataclass
class LayerTrainParams:
    """Trainierbare Groessen einer einzelnen Schicht."""

    free_positions: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-3))
    intensities: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-4))
    occupancies: TrainParam = field(default_factory=lambda: TrainParam(False, 0.1))


# ---------------------------------------------------------------------------
# (De)Serialisierung
# ---------------------------------------------------------------------------

def serialize(obj):
    if isinstance(obj, torch.Tensor):
        return {"__type__": "tensor", "data": obj.detach().cpu().tolist()}

    elif is_dataclass(obj):
        d = {
            "__type__": "dataclass",
            "__class__": type(obj).__name__,
            "data": {}
        }
        # fields() gibt nur dataclass-Felder zurück, obj bleibt erhalten
        for f in fields(obj):
            d["data"][f.name] = serialize(getattr(obj, f.name))
        # Klassenattribute manuell
        for k in ["train_aberrations", "train_env_params"]:
            if hasattr(obj, k):
                d["data"][k] = serialize(getattr(obj, k))
        return d

    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [serialize(v) for v in obj]

    elif isinstance(obj, tuple):
        return {"__type__": "tuple", "data": [serialize(v) for v in obj]}

    else:
        return obj


def deserialize(obj, class_registry: dict):
    if isinstance(obj, dict):
        t = obj.get("__type__")

        if t == "tensor":
            return torch.tensor(obj["data"])

        elif t == "tuple":
            return tuple(deserialize(v, class_registry) for v in obj["data"])

        elif t == "dataclass":
            cls = class_registry[obj["__class__"]]
            # Klassenattribute rausfiltern, die kein __init__-Parameter sind
            init_fields = {f.name for f in fields(cls)}
            kwargs = {}
            class_attrs = {}
            for k, v in obj["data"].items():
                if k in init_fields:
                    kwargs[k] = deserialize(v, class_registry)
                else:
                    class_attrs[k] = deserialize(v, class_registry)
            instance = cls(**kwargs)
            # Klassenattribute setzen
            for k, v in class_attrs.items():
                setattr(instance, k, v)
            return instance

        return {k: deserialize(v, class_registry) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [deserialize(v, class_registry) for v in obj]

    else:
        return obj


# ---------------------------------------------------------------------------
# TrainMethod
# ---------------------------------------------------------------------------

#: Praefixe der Schluessel, die :meth:`TrainMethod.iter_train_params` erzeugt.
GROUP_KEY = "groups"
LAYER_KEY = "layers"


@dataclass
class TrainMethod:
    """Was trainiert wird - dynamisch ueber Gruppen und Schichten.

    ``groups`` und ``layers`` sind Dicts, deren Schluessel den Namen aus der
    :class:`~layers.LayerGroupConfig` / :class:`~layers.LayerConfig` in der
    ``ParticlesConfig`` entsprechen.  Damit haengt die Anzahl der Schichten
    nirgends mehr im Code fest.

    ``active_group`` waehlt aus, an welcher Gruppe gerade gearbeitet wird
    (Maske/Target/Isolation).  ``None`` bedeutet "alle Gruppen" (Main-Training).
    """

    # --- Trainierbare Parameter ------------------------------------------------
    groups: Dict[str, GroupTrainParams] = field(default_factory=dict)
    layers: Dict[str, LayerTrainParams] = field(default_factory=dict)

    train_amplitude_contrast: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-5))
    #: Aberrationen inkl. ihrer Azimutwinkel (C12 und phi12 gehoeren zusammen)
    train_ctf_aberrations: TrainParam = field(default_factory=lambda: TrainParam(True, 1e-5))
    train_env_parameters: TrainParam = field(default_factory=lambda: TrainParam(False, 1e-5))
    train_normalisation_parameters: TrainParam = field(default_factory=lambda: TrainParam(True, 1e-5))

    # --- Loss-Gewichte ---------------------------------------------------------
    loss_var: float = 0.5
    loss_local_var_100: float = 0.0
    loss_mse: float = 0.0
    loss_ncc: float = 0.5
    loss_local_ncc_100: float = 0.0
    loss_ssim: float = 0.0
    loss_l1: float = 0.1

    occ_entropy_a: float = 1.0
    position_shifts_regularization_a: float = 1.0
    #: Abstossung zwischen Atomen einer Schicht: Gewicht und Abstand [A], unter dem sie wirkt (0 = aus)
    repulsion_a: float = 0.0
    repulsion_distance_A: float = 1.0
    strain_energy_a: float = 0.0
    strain_energy_lambda: float = 1.0
    strain_energy_mu: float = 1.0

    # --- Aktive Gruppe ---------------------------------------------------------
    #: Name der aktiven Gruppe oder None fuer "alle Gruppen".
    active_group: Optional[str] = None

    # --- Daten (nicht in der GUI) ----------------------------------------------
    #: Flaechenmasken je Gruppe.
    masks: Dict[str, Optional[torch.Tensor]] = field(default_factory=dict)
    #: Optionales eigenes Zielbild je Gruppe.
    targets: Dict[str, Optional[torch.Tensor]] = field(default_factory=dict)

    #: Legacy-Eingaben (alte Aufrufe ``TrainMethod(area_masks=[...], target_gr=...)``)
    area_masks: Optional[List] = None
    target_gr: Optional[torch.Tensor] = None

    train_aberrations = {"C10": True,
                         'C12': True,
                         'C23': True,
                         'C21': True,
                         "C30": True,
                         'C34': True,
                         'C45': True}  # um
    train_env_params = {
        'source_size': False,
        'Cc': True,
        'deltaE': True,
        'instab_defocus_nm': False
    }

    #: Felder, die der GUI-Editor nicht anfassen soll.
    NON_GUI_FIELDS = ("masks", "targets", "area_masks", "target_gr", "groups", "layers", "active_group")

    def __post_init__(self):
        # Alte Aufrufform: Liste von Masken in der Reihenfolge [Gr, WS2].
        if self.area_masks:
            for i, mask in enumerate(self.area_masks):
                if mask is None:
                    continue
                name = LEGACY_GROUP_NAMES[i] if i < len(LEGACY_GROUP_NAMES) else f"group{i}"
                self.masks.setdefault(name, mask)
        if self.target_gr is not None:
            self.targets.setdefault(LEGACY_GROUP_NAMES[0], self.target_gr)

    # -- Zugriff ---------------------------------------------------------------
    def group(self, name: str) -> GroupTrainParams:
        if name not in self.groups:
            self.groups[name] = GroupTrainParams()
        return self.groups[name]

    def layer(self, name: str) -> LayerTrainParams:
        if name not in self.layers:
            self.layers[name] = LayerTrainParams()
        return self.layers[name]

    def mask_for(self, group_name: Optional[str]):
        if group_name is None:
            return None
        return self.masks.get(group_name)

    def target_for(self, group_name: Optional[str]):
        if group_name is None:
            return None
        return self.targets.get(group_name)

    def set_active_group(self, name: Optional[str]):
        self.active_group = name or None

    def ensure_for_config(self, config) -> "TrainMethod":
        """Legt fehlende Eintraege fuer alle Gruppen/Schichten der Config an."""

        for group in getattr(config, "groups", []):
            self.group(group.name)
        for layer in getattr(config, "layers", []):
            self.layer(layer.name)
        if self.active_group is not None and self.active_group not in self.groups:
            self.active_group = None
        return self

    # -- Iteration ueber alle TrainParam-Eintraege -------------------------------
    def iter_train_params(self) -> List[Tuple[str, TrainParam]]:
        """Liefert ``(key, TrainParam)`` fuer alles, was trainiert werden kann.

        Schluessel-Format: ``groups.<gruppe>.<feld>``, ``layers.<schicht>.<feld>``
        oder der Feldname selbst fuer globale Parameter.
        """

        out: List[Tuple[str, TrainParam]] = []
        for f in fields(self):
            value = getattr(self, f.name)
            if is_train_param(value):
                out.append((f.name, value))

        for gname, gparams in self.groups.items():
            for f in fields(gparams):
                out.append((f"{GROUP_KEY}.{gname}.{f.name}", getattr(gparams, f.name)))

        for lname, lparams in self.layers.items():
            for f in fields(lparams):
                out.append((f"{LAYER_KEY}.{lname}.{f.name}", getattr(lparams, f.name)))

        return out

    def get_train_param(self, key: str) -> Optional[TrainParam]:
        parts = key.split(".")
        if len(parts) == 1:
            value = getattr(self, key, None)
            return value if is_train_param(value) else None
        if len(parts) != 3:
            return None
        kind, name, attr = parts
        holder = self.groups.get(name) if kind == GROUP_KEY else self.layers.get(name)
        if holder is None:
            return None
        return getattr(holder, attr, None)

    def is_enabled(self, key: str) -> bool:
        tp = self.get_train_param(key)
        return bool(tp is not None and tp.enabled)

    # -- Bequeme Abfragen fuer den Rest des Codes -------------------------------
    def group_enabled(self, group_name: str, attr: str) -> bool:
        gp = self.groups.get(group_name)
        return bool(gp is not None and getattr(gp, attr).enabled)

    def layer_enabled(self, layer_name: str, attr: str) -> bool:
        lp = self.layers.get(layer_name)
        return bool(lp is not None and getattr(lp, attr).enabled)


def is_train_param(value) -> bool:
    return is_dataclass(value) and hasattr(value, "enabled") and hasattr(value, "lr")


registry = {
    "TrainMethod": TrainMethod,
    "TrainParam": TrainParam,
    "GroupTrainParams": GroupTrainParams,
    "LayerTrainParams": LayerTrainParams,
}


# ---------------------------------------------------------------------------
# Migration alter train_method.json Dateien
# ---------------------------------------------------------------------------

#: alte Feldnamen -> (Gruppe, Feld) der neuen Struktur
LEGACY_GROUP_FIELDS = {
    "graphen_shift": ("Gr", "shift"),
    "graphen_rotation": ("Gr", "rotation"),
    "graphen_scale": ("Gr", "scale"),
    "Gr_strain_field": ("Gr", "strain_field"),
    "WS2_shift": ("WS2", "shift"),
    "WS2_rotation": ("WS2", "rotation"),
    "WS2_scale": ("WS2", "scale"),
    "WS2_strain_field": ("WS2", "strain_field"),
}

#: alte Feldnamen -> (Schicht, Feld); "Gr"/"S1"/... aus LEGACY_LAYER_NAMES
LEGACY_LAYER_FIELDS = {
    "train_graphen_intensities": ("Gr", "intensities"),
    "train_S1_intensities": ("S1", "intensities"),
    "train_W_intensities": ("W", "intensities"),
    "train_S2_intensities": ("S2", "intensities"),
    "train_graphen_occupancies": ("Gr", "occupancies"),
    "train_S1_occupancies": ("S1", "occupancies"),
    "train_W_occupancies": ("W", "occupancies"),
    "train_S2_occupancies": ("S2", "occupancies"),
}

#: alte Gruppen-Flags, die auf alle Schichten der Gruppe wirkten
LEGACY_GROUP_TO_LAYER_FIELDS = {
    "graphen_free_positions": ("Gr", "free_positions"),
    "WS2_free_positions": ("WS2", "free_positions"),
}


def _is_legacy_train_method(raw: dict) -> bool:
    data = raw.get("data", raw)
    return any(k in data for k in LEGACY_GROUP_FIELDS) or "only_train_small_area" in data


def migrate_train_method_dict(raw: dict) -> dict:
    """Wandelt eine alte (flache) train_method.json in die neue Struktur um."""

    data = dict(raw.get("data", raw))

    groups: Dict[str, dict] = {}
    layers: Dict[str, dict] = {}

    def _group(name):
        return groups.setdefault(name, {
            "__type__": "dataclass", "__class__": "GroupTrainParams", "data": {}})["data"]

    def _layer(name):
        return layers.setdefault(name, {
            "__type__": "dataclass", "__class__": "LayerTrainParams", "data": {}})["data"]

    for old_name, (gname, attr) in LEGACY_GROUP_FIELDS.items():
        if old_name in data:
            _group(gname)[attr] = data.pop(old_name)

    for old_name, (lname, attr) in LEGACY_LAYER_FIELDS.items():
        if old_name in data:
            _layer(lname)[attr] = data.pop(old_name)

    for old_name, (gname, attr) in LEGACY_GROUP_TO_LAYER_FIELDS.items():
        if old_name not in data:
            continue
        value = data.pop(old_name)
        for lname in LEGACY_LAYER_NAMES:
            if LEGACY_LAYER_GROUP[lname] == gname:
                _layer(lname)[attr] = value

    # Aktive Gruppe aus only_train_small_area
    area_idx = data.pop("only_train_small_area", None)
    if area_idx is not None:
        area_idx = int(area_idx)
        if 0 <= area_idx < len(LEGACY_GROUP_NAMES):
            data["active_group"] = LEGACY_GROUP_NAMES[area_idx]
        else:
            data["active_group"] = None

    # Felder, die es nicht mehr gibt, still verwerfen
    known = {f.name for f in fields(TrainMethod)} | {"train_aberrations", "train_env_params"}
    dropped = [k for k in data if k not in known]
    for k in dropped:
        data.pop(k)
    if dropped:
        print(f"[train_method migration] ignorierte alte Felder: {dropped}")

    data["groups"] = groups
    data["layers"] = layers

    out = dict(raw)
    if "data" in raw:
        out["data"] = data
    else:
        out = data
    return out


# ---------------------------------------------------------------------------
# Speichern / Laden
# ---------------------------------------------------------------------------

def _tensor_to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_train_methods(tm: TrainMethod, save_path: str):
    data = serialize(tm)
    for key in ("masks", "targets", "area_masks", "target_gr"):
        data["data"].pop(key, None)
    with open(save_path, "w") as f:
        json.dump(data, f)

    config_dir = os.path.dirname(save_path)

    masks = {k: _tensor_to_numpy(v) for k, v in tm.masks.items() if v is not None}
    targets = {k: _tensor_to_numpy(v) for k, v in tm.targets.items() if v is not None}

    if masks:
        np.savez(os.path.join(config_dir, "area_masks.npz"), **masks)
    if targets:
        np.savez(os.path.join(config_dir, "group_targets.npz"), **targets)


def _load_legacy_mask_files(tm: TrainMethod, config_dir: str, group_names_ordered):
    """Liest die alten ``area_masks.npy`` / ``target_gr.npy`` Dateien ein."""

    legacy_masks = os.path.join(config_dir, "area_masks.npy")
    if os.path.exists(legacy_masks):
        masks = np.load(legacy_masks)
        for i in range(masks.shape[0]):
            name = group_names_ordered[i] if i < len(group_names_ordered) else f"group{i}"
            tm.masks.setdefault(name, torch.tensor(masks[i]))

    legacy_target = os.path.join(config_dir, "target_gr.npy")
    if os.path.exists(legacy_target) and group_names_ordered:
        tm.targets.setdefault(
            group_names_ordered[0],
            torch.tensor(np.load(legacy_target), dtype=torch.float32),
        )


def load_train_method(save_path: str, config=None) -> TrainMethod:
    """Laedt eine train_method.json - alte (flache) Dateien werden migriert.

    ``config`` (optional) ist die zugehoerige ``ParticlesConfig``; damit werden
    fehlende Gruppen-/Schicht-Eintraege ergaenzt und die Masken den richtigen
    Gruppennamen zugeordnet.
    """

    with open(save_path, "r") as f:
        raw = json.load(f)

    if _is_legacy_train_method(raw):
        print(f"[train_method] migriere altes Format: {save_path}")
        raw = migrate_train_method_dict(raw)

    tm = deserialize(raw, registry)

    if config is not None:
        tm.ensure_for_config(config)
        group_names_ordered = [g.name for g in config.groups]
    else:
        group_names_ordered = list(LEGACY_GROUP_NAMES)

    config_dir = os.path.dirname(save_path)

    masks_npz = os.path.join(config_dir, "area_masks.npz")
    if os.path.exists(masks_npz):
        with np.load(masks_npz) as data:
            for name in data.files:
                tm.masks.setdefault(name, torch.tensor(data[name]))

    targets_npz = os.path.join(config_dir, "group_targets.npz")
    if os.path.exists(targets_npz):
        with np.load(targets_npz) as data:
            for name in data.files:
                tm.targets.setdefault(name, torch.tensor(data[name], dtype=torch.float32))

    _load_legacy_mask_files(tm, config_dir, group_names_ordered)

    return tm


def load_group_data(tm: TrainMethod, config, base_dir: str, dtype=torch.float32) -> TrainMethod:
    """Laedt Maske und Zielbild je Gruppe aus den in der Config hinterlegten Pfaden."""

    import tifffile

    def _read(path):
        full = path if os.path.isabs(path) else os.path.join(base_dir, path)
        if full.endswith(".npy"):
            return np.load(full)
        return tifffile.imread(full)

    for group in config.groups:
        if group.mask_file and group.name not in tm.masks:
            tm.masks[group.name] = torch.tensor(_read(group.mask_file) > 0)
        if group.target_file and group.name not in tm.targets:
            image = np.asarray(_read(group.target_file), dtype=np.float32)
            image[np.where(image == 0.0)] = np.median(image)
            tm.targets[group.name] = torch.tensor(image, dtype=dtype)

    return tm
