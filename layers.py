"""Dynamische Schicht-Konfiguration fuer das Multislice-Modell.

Frueher waren genau vier Schichten fest verdrahtet (Gr, S1, W, S2) und die
Trainierbarkeit war in zwei Gruppen gesplittet (``graphen_*`` / ``WS2_*``).
Hier steht jetzt die generische Beschreibung:

* :class:`LayerConfig`      - eine physikalische Schicht (Element, z-Position)
* :class:`LayerGroupConfig` - eine Struktur-Gruppe, die sich gemeinsam bewegt
  (gemeinsamer Shift / Rotation / Scale / Strain-Feld / Maske / Target)

Eine Gruppe kann beliebig viele Schichten enthalten. Das alte Verhalten ist
exakt die Konstellation ``Gr`` = [Gr] und ``WS2`` = [S1, W, S2] und wird von
:func:`legacy_layers_and_groups` erzeugt (siehe Migration alter configs).

Das Modul haengt bewusst von nichts ab (kein torch, kein nn.py), damit es von
allen anderen Modulen importiert werden kann.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LayerGroupConfig:
    """Eine Gruppe von Schichten, die sich als starre Einheit bewegt."""

    name: str

    # Strain-Feld (Spline-Kontrollpunkte)
    control_points: List[int] = field(default_factory=lambda: [10, 10])

    # Limits fuer die globale Transformation (None -> kein Clamping)
    max_shift: Optional[float] = None
    max_rotation: Optional[float] = None

    # Startwerte der globalen Transformation
    init_shift: List[float] = field(default_factory=lambda: [0.0, 0.0])
    init_rotation: float = 0.0
    init_scale: List[float] = field(default_factory=lambda: [1.0, 1.0])
    init_potential_factor: float = 1.0

    # Datenquellen fuer diese Gruppe (Pfade relativ zum Datensatz-Ordner)
    mask_file: Optional[str] = None
    target_file: Optional[str] = None

    # Wenn diese Gruppe die aktive ist: nur ihre Schichten simulieren,
    # alle anderen Schichten liefern ein Null-Potential.
    isolate_when_active: bool = False


@dataclass
class LayerConfig:
    """Eine einzelne Schicht (ein Multislice-Slice)."""

    name: str
    element: str
    z_nm: float = 0.0
    group: str = ""              # leer -> eigene Gruppe mit gleichem Namen
    init_occupancy: float = 2.0  # Logit, wird spaeter durch sigmoid() geschickt
    occ_entropy: bool = True     # Occupancy-Entropie-Regularisierung anwenden?


# ---------------------------------------------------------------------------
# Legacy-Definition (Gr / S1 / W / S2)
# ---------------------------------------------------------------------------

#: Namen der vier fest verdrahteten Schichten der alten Implementierung.
LEGACY_LAYER_NAMES: Tuple[str, ...] = ("Gr", "S1", "W", "S2")
LEGACY_LAYER_ELEMENTS: Tuple[str, ...] = ("C", "S", "W", "S")
#: Zuordnung alte Schicht -> alte Gruppe.
LEGACY_LAYER_GROUP: Dict[str, str] = {"Gr": "Gr", "S1": "WS2", "W": "WS2", "S2": "WS2"}
LEGACY_GROUP_NAMES: Tuple[str, ...] = ("Gr", "WS2")

#: Startwerte, die frueher als Default in ``TunablePotentialParameters`` standen.
LEGACY_GROUP_INIT = {
    "Gr": dict(init_scale=[1.12, 1.14], init_rotation=0.025, init_shift=[-9.5, -11.0]),
    "WS2": dict(init_scale=[1.05, 1.09], init_rotation=0.0, init_shift=[15.9, -5.0]),
}

#: Occupancy-Startwerte der alten ``initialize_intensities_parameters``.
LEGACY_INIT_OCCUPANCY = {"Gr": 100.0, "S1": 2.0, "W": 2.0, "S2": 2.0}


def legacy_layers_and_groups(
        z_positions_nm: Sequence[float],
        control_points_graphen: Sequence[int] = (10, 10),
        control_points_ws2: Sequence[int] = (10, 10),
        max_shifts: Optional[Sequence[float]] = None,
        max_rotations: Optional[Sequence[float]] = None,
        n_layers: Optional[int] = None,
) -> Tuple[List[LayerConfig], List[LayerGroupConfig]]:
    """Baut die alte 4-Schicht-Konstellation als Layer-/Group-Liste.

    Wird fuer die Migration alter ``config.json`` Dateien benutzt, die noch
    kein ``layers``-Feld haben.  Fuer eine abweichende Schichtzahl wird auf
    generische Namen (``L0`` ... ) mit einer einzigen Gruppe ausgewichen.
    """

    z_positions_nm = list(z_positions_nm) if z_positions_nm is not None else []
    if n_layers is None:
        n_layers = len(z_positions_nm)
    while len(z_positions_nm) < n_layers:
        z_positions_nm.append(0.0)

    if n_layers != len(LEGACY_LAYER_NAMES):
        # Unbekannte Konstellation: eine Gruppe, generische Schichtnamen.
        group = LayerGroupConfig(
            name="all",
            control_points=list(control_points_ws2),
            max_shift=None if max_shifts is None else float(max_shifts[-1]),
            max_rotation=None if max_rotations is None else float(max_rotations[-1]),
        )
        layers = [
            LayerConfig(name=f"L{i}", element="C", z_nm=float(z_positions_nm[i]), group="all")
            for i in range(n_layers)
        ]
        return layers, [group]

    groups: List[LayerGroupConfig] = []
    for gi, gname in enumerate(LEGACY_GROUP_NAMES):
        groups.append(LayerGroupConfig(
            name=gname,
            control_points=list(control_points_graphen if gname == "Gr" else control_points_ws2),
            max_shift=None if max_shifts is None else float(max_shifts[gi]),
            max_rotation=None if max_rotations is None else float(max_rotations[gi]),
            mask_file=None,
            target_file=None,
            isolate_when_active=(gname == "Gr"),
            **LEGACY_GROUP_INIT[gname],
        ))

    layers = [
        LayerConfig(
            name=name,
            element=LEGACY_LAYER_ELEMENTS[i],
            z_nm=float(z_positions_nm[i]),
            group=LEGACY_LAYER_GROUP[name],
            init_occupancy=LEGACY_INIT_OCCUPANCY[name],
            # Graphen war frueher von der Occupancy-Entropie ausgenommen
            # (aux_loss lief ueber range(1, N)).
            occ_entropy=(name != "Gr"),
        )
        for i, name in enumerate(LEGACY_LAYER_NAMES)
    ]
    return layers, groups


# ---------------------------------------------------------------------------
# Helfer
# ---------------------------------------------------------------------------

def _as_dataclass(value, cls):
    if isinstance(value, cls):
        return value
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        # Serialisierte Form aus main.serialize() ({"__type__": "dataclass", ...})
        if value.get("__type__") == "dataclass":
            value = value["data"]
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(value) - known
        if unknown:
            raise KeyError(f"Unbekannte Felder fuer {cls.__name__}: {sorted(unknown)}")
        return cls(**value)
    raise TypeError(f"Kann {value!r} nicht als {cls.__name__} interpretieren")


def coerce_layers(raw) -> List[LayerConfig]:
    return [_as_dataclass(item, LayerConfig) for item in (raw or [])]


def coerce_groups(raw) -> List[LayerGroupConfig]:
    return [_as_dataclass(item, LayerGroupConfig) for item in (raw or [])]


def resolve_layers_and_groups(layers, groups) -> Tuple[List[LayerConfig], List[LayerGroupConfig]]:
    """Validiert Schichten/Gruppen und ergaenzt fehlende Gruppen.

    * Schichten ohne ``group`` bekommen eine eigene Gruppe mit ihrem Namen.
    * Gruppen, die von einer Schicht referenziert werden aber nicht definiert
      sind, werden mit Defaults angelegt.
    """

    layers = coerce_layers(layers)
    groups = coerce_groups(groups)

    if not layers:
        raise ValueError("Es muss mindestens eine Schicht definiert sein.")

    names = [l.name for l in layers]
    if len(set(names)) != len(names):
        raise ValueError(f"Schichtnamen muessen eindeutig sein, bekommen: {names}")

    # Namen landen als Schluessel in nn.ModuleDict und in den TrainMethod-Keys
    # ("groups.<name>.shift"), deshalb kein Punkt und nicht leer.
    for name in names + [g.name for g in groups] + [l.group for l in layers if l.group]:
        if not name or "." in name:
            raise ValueError(
                f"Ungueltiger Name {name!r}: darf nicht leer sein und keinen Punkt enthalten.")

    for layer in layers:
        if not layer.group:
            layer.group = layer.name

    by_name = {g.name: g for g in groups}
    if len(by_name) != len(groups):
        raise ValueError("Gruppennamen muessen eindeutig sein.")

    for layer in layers:
        if layer.group not in by_name:
            group = LayerGroupConfig(name=layer.group)
            by_name[group.name] = group
            groups.append(group)

    used = {l.group for l in layers}
    groups = [g for g in groups if g.name in used]
    return layers, groups


def layer_indices_by_group(layers: Sequence[LayerConfig]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for i, layer in enumerate(layers):
        out.setdefault(layer.group, []).append(i)
    return out


def layer_names(layers: Sequence[LayerConfig]) -> List[str]:
    return [l.name for l in layers]


def group_names(groups: Sequence[LayerGroupConfig]) -> List[str]:
    return [g.name for g in groups]


def z_positions_nm(layers: Sequence[LayerConfig]) -> List[float]:
    return [float(l.z_nm) for l in layers]


def graphene_only(
        z_nm: float = 0.0,
        control_points: Sequence[int] = (10, 10),
        max_shift: Optional[float] = None,
        max_rotation: Optional[float] = None,
        mask_file: Optional[str] = None,
        target_file: Optional[str] = None,
) -> Tuple[List[LayerConfig], List[LayerGroupConfig]]:
    """Bequeme Kurzform: ein Modell aus genau einer Graphen-Schicht."""

    group = LayerGroupConfig(
        name="Gr",
        control_points=list(control_points),
        max_shift=max_shift,
        max_rotation=max_rotation,
        mask_file=mask_file,
        target_file=target_file,
        isolate_when_active=True,
        **LEGACY_GROUP_INIT["Gr"],
    )
    layer = LayerConfig(
        name="Gr", element="C", z_nm=float(z_nm), group="Gr",
        init_occupancy=LEGACY_INIT_OCCUPANCY["Gr"], occ_entropy=False,
    )
    return [layer], [group]
