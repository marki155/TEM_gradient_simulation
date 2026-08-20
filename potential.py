from dataclasses import dataclass, field
from typing import Dict, List, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from ase.data import atomic_numbers
from matplotlib import pyplot as plt

from displacement_field import SplineDisplacementField2D, StrainTunableParameters
from layers import (
    LEGACY_GROUP_NAMES,
    LEGACY_LAYER_NAMES,
    layer_indices_by_group,
)

from torch.nn.utils.rnn import pad_sequence

from kirkland_supersampled import KirklandPrecalcPotential
from lobato_potential import projected_atomic_potential_kernel_lobato
from physics_helper import relativistic_wavelength_nm

from dataclasses import fields, is_dataclass, asdict
from typing import Any, Type

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)

PROJECTED_POTENTIAL_PARAMS = {
    "C": [
        (18.2, 0.020),
        (7.6,  0.055),
        (2.1,  0.120),
    ],
    "S": [
        (55.0, 0.018),
        (22.0, 0.050),
        (6.5,  0.120),
    ],
    "W": [
        (210.0, 0.015),
        (85.0,  0.040),
        (25.0,  0.100),
    ],
}


PROJECTED_POTENTIAL_PARAMS = {
    "C": [
        (0.020, 0.020),
        (0.030, 0.055),
        (0.020, 0.120),
    ],
    "S": [
        (0.050, 0.018),
        (0.070, 0.050),
        (0.050, 0.120),
    ],
    "W": [
        (0.120, 0.015),
        (0.180, 0.040),
        (0.120, 0.100),
    ],
}

PROJECTED_POTENTIAL_PARAMS = {
    "C": [
        (0.0008, 0.020),
        (0.0012, 0.050),
        (0.0008, 0.120),
    ],
    "S": [
        (0.0015, 0.018),
        (0.0020, 0.050),
        (0.0015, 0.120),
    ],
    "W": [
        (0.0008, 0.015),
        (0.0012, 0.040),
        (0.0008, 0.100),
    ],
}

DOYLE_TURNER_PARAMS = {
    'C': [(0.489918, 0.206593), (0.262003, 0.0774039), (0.196767, 0.495519)],
    'W': [(0.489918, 0.35384), (0.262003, 0.32576), (0.196767, 0.1488)],
    'S': [(0.489918, 0.02051), (0.262003, 0.01164)]
}

DOYLE_TURNER_PARAMS = {
    "C": [(0.489918, 20.6593), (0.262003, 7.74039), (0.196767, 49.5519)],
    "W": [(0.489918, 35.384), (0.262003, 32.576), (0.196767, 14.880)],
    "S": [(0.489918, 2.051), (0.262003, 1.164)]
}

DOYLE_TURNER_PARAMS_NM = {
    "C": [
        (0.489918, 0.00206593),
        (0.262003, 0.000774039),
        (0.196767, 0.00495519),
    ],
    "W": [
        (0.489918, 0.00353840),
        (0.262003, 0.00325760),
        (0.196767, 0.00148800),
    ],
    "S": [
        (0.489918, 0.00020510),
        (0.262003, 0.00011640),
    ],
}


DOYLE_TURNER_PARAMS = {
    "C": [
        (2980.009564, 19109.271662),
        (4253.567068, 51003.137574),
        (499.001460,   7967.084532),
    ],
    "W": [
        (1739.908195, 11157.138143),
        (1010.690938, 12118.865915),
        (1661.725166, 26531.194627),
    ],
    "S": [
        (30017.021739, 192483.752337),
        (28285.453609, 339161.663268),
    ],
}


def update_from_serialized(target: Any, source: Any):
    # Dataclass rekursiv durchlaufen
    if is_dataclass(target):
        if not isinstance(source, dict):
            raise TypeError(f"Für Dataclass {type(target).__name__} wurde dict erwartet, bekam {type(source).__name__}")
        for f in fields(target):
            name = f.name
            if name not in source:
                continue
            update_from_serialized(getattr(target, name), source[name])
        return

    # Einzelner Parameter
    if isinstance(target, nn.Parameter):
        tensor = torch.tensor(source, dtype=target.dtype, device=target.device)
        with torch.no_grad():
            target.copy_(tensor)
        return

    # Normaler Tensor
    if isinstance(target, torch.Tensor):
        tensor = torch.tensor(source, dtype=target.dtype, device=target.device)
        with torch.no_grad():
            target.copy_(tensor)
        return

    # ParameterList
    if isinstance(target, nn.ParameterList):
        if len(target) != len(source):
            raise ValueError(f"ParameterList-Länge passt nicht: {len(target)} != {len(source)}")
        for t, s in zip(target, source):
            update_from_serialized(t, s)
        return

    # ParameterDict
    if isinstance(target, nn.ParameterDict):
        for key, value in source.items():
            if key not in target:
                raise KeyError(f"Key {key!r} fehlt in ParameterDict")
            update_from_serialized(target[key], value)
        return

    # list
    if isinstance(target, list):
        if len(target) != len(source):
            raise ValueError(f"Listenlänge passt nicht: {len(target)} != {len(source)}")
        for t, s in zip(target, source):
            update_from_serialized(t, s)
        return

    # dict
    if isinstance(target, dict):
        for key, value in source.items():
            if key not in target:
                raise KeyError(f"Key {key!r} fehlt im dict")
            update_from_serialized(target[key], value)
        return


@dataclass
class TunablePotentialParameters:
    """Alle trainierbaren Groessen des Potentials.

    Pro Schicht (Reihenfolge = ``config.layers``): ``position_shifts``,
    ``intensities``, ``occupancies``.
    Pro Gruppe (Schluessel = Gruppenname): ``shifts``, ``rotations``,
    ``scales``, ``contrast_factors``, ``strain_fields``.
    """

    strain_fields: Dict[str, StrainTunableParameters]

    position_shifts: nn.ParameterList
    intensities: nn.ParameterList
    occupancies: nn.ParameterList

    shifts: nn.ParameterDict
    rotations: nn.ParameterDict
    scales: nn.ParameterDict
    #: empirischer Kontrastfaktor je Gruppe (Stobbs-Faktor), siehe LayerGroupConfig
    contrast_factors: nn.ParameterDict


# ---------------------------------------------------------------------------
# Migration alter Parameter-Dateien (best.json & Co.)
# ---------------------------------------------------------------------------

#: alte Feldnamen -> (neues Dict, Gruppenname)
LEGACY_PARAMETER_FIELDS = {
    "scale_Gr": ("scales", "Gr"),
    "rotation_Gr": ("rotations", "Gr"),
    "shift_Gr": ("shifts", "Gr"),
    "scale_WS2": ("scales", "WS2"),
    "rotation_WS2": ("rotations", "WS2"),
    "shift_WS2": ("shifts", "WS2"),
    "graphene_potential_factor": ("contrast_factors", "Gr"),
    "ws2_potential_factor": ("contrast_factors", "WS2"),
}


def is_legacy_potential_parameters(source) -> bool:
    if not isinstance(source, dict):
        return False
    # "potential_factors" hiess der Kontrastfaktor in einer Zwischenversion
    return any(k in source for k in LEGACY_PARAMETER_FIELDS) or "potential_factors" in source


def migrate_serialized_potential_parameters(source, config):
    """Bringt eine alte (Gr/WS2-)Parameter-Datei in die neue Gruppen-Struktur.

    Schichtbezogene Listen werden ueber die alten Schichtnamen zugeordnet, damit
    z.B. eine reine Graphen-Config die Graphen-Eintraege einer alten
    4-Schicht-Datei uebernehmen kann.
    """

    if not is_legacy_potential_parameters(source):
        return source

    group_names = [g.name for g in config.groups]
    layer_names_new = [l.name for l in config.layers]

    out = dict(source)

    # Zwischenversion: der Kontrastfaktor hiess potential_factors
    if "potential_factors" in out:
        renamed = out.pop("potential_factors")
        out.setdefault("contrast_factors", renamed)

    buckets: Dict[str, Dict[str, object]] = {
        "scales": {}, "rotations": {}, "shifts": {}, "contrast_factors": {}}

    for old_key, (bucket, gname) in LEGACY_PARAMETER_FIELDS.items():
        if old_key in out:
            value = out.pop(old_key)
            if gname in group_names:
                buckets[bucket][gname] = value

    for bucket, values in buckets.items():
        if values:
            out[bucket] = values

    strain = out.get("strain_fields")
    if isinstance(strain, list):
        out["strain_fields"] = {
            name: strain[i]
            for i, name in enumerate(LEGACY_GROUP_NAMES)
            if i < len(strain) and name in group_names
        }

    # Schichtlisten auf die neuen Schichten umsortieren
    for key in ("position_shifts", "intensities", "occupancies"):
        value = out.get(key)
        if not isinstance(value, list) or len(value) == len(layer_names_new):
            continue
        remapped = []
        ok = True
        for name in layer_names_new:
            if name in LEGACY_LAYER_NAMES:
                idx = LEGACY_LAYER_NAMES.index(name)
                if idx < len(value):
                    remapped.append(value[idx])
                    continue
            ok = False
            break
        if ok:
            print(f"[potential migration] '{key}' von {len(value)} auf "
                  f"{len(remapped)} Schichten gemappt: {layer_names_new}")
            out[key] = remapped
        else:
            print(f"[potential migration] '{key}' passt nicht zu den neuen "
                  f"Schichten {layer_names_new} und wird ignoriert.")
            out.pop(key)

    return out


class DifferentiablePotential(nn.Module):
    """Projiziertes Potential aller Schichten.

    Die Schichten und ihre Gruppen kommen komplett aus ``config.layers`` /
    ``config.groups`` - im Code ist nichts mehr auf Gr/S1/W/S2 festgelegt.
    """

    def __init__(self, config, train_methods, device):
        super().__init__()
        self.pixel_size_nm = config.pixel_size_nm
        self.wavelength_nm = relativistic_wavelength_nm(config.voltage_kV)
        self.device = device

        self.H, self.W = config.shape

        self.layers = list(config.layers)
        self.groups = list(config.groups)
        self.groups_by_name = {g.name: g for g in self.groups}
        self.layer_indices_by_group = layer_indices_by_group(self.layers)
        self.group_of_layer = [self.groups_by_name[l.group] for l in self.layers]
        self.layer_names = [l.name for l in self.layers]
        self.group_names = [g.name for g in self.groups]
        self.N_l = len(self.layers)

        if len(config.init_positions) != self.N_l:
            raise ValueError(
                f"init_positions hat {len(config.init_positions)} Eintraege, "
                f"die Config definiert aber {self.N_l} Schichten "
                f"({self.layer_names}).")

        self.train_methods = train_methods
        self.train_method = train_methods  # Alias, altes Attribut
        self.config = config

        self.strain_fields = nn.ModuleDict({
            group.name: SplineDisplacementField2D(
                config=config, control_shape=list(group.control_points),
                device=device, dtype=torch.float32, init_scale=0.0)
            for group in self.groups
        })

        strain_parameters = {name: fieldd.get_tunable_parameters()
                             for name, fieldd in self.strain_fields.items()}
        position_shifts_ = self.initialize_position_shifts_parameters(config.init_positions)
        intensities_, occupancies_ = self.initialize_intensities_parameters(config.init_intensities)
        shifts_, rotations_, scales_, factors_ = self.initialize_group_parameters()

        self.tunable_parameters = TunablePotentialParameters(
            strain_fields=strain_parameters,
            position_shifts=position_shifts_,
            intensities=intensities_,
            occupancies=occupancies_,
            shifts=shifts_,
            rotations=rotations_,
            scales=scales_,
            contrast_factors=factors_,
        )

        # Elemente je Schicht: entweder genau eins oder eine Mischreihe
        self.layer_elements = [self._resolve_elements(layer) for layer in self.layers]
        self.layer_atomic_numbers = [
            torch.tensor([float(atomic_numbers[s]) for s in symbols],
                         dtype=torch.float32, device=device)
            for symbols in self.layer_elements
        ]

        self.potential_pad = config.padding
        self.kirkland_precalc_pot = KirklandPrecalcPotential(
            self.H + 2 * self.potential_pad,
            self.W + 2 * self.potential_pad,
            self.pixel_size_nm, self.device,
            atom_types=sorted({s for symbols in self.layer_elements for s in symbols},
                              key=lambda s: atomic_numbers[s]),
            debye_waller=self._collect_debye_waller())

        self.apply_trainable_selection()

        for name, value in vars(self.tunable_parameters).items():
            if isinstance(value, (nn.Parameter, nn.ParameterList, nn.ParameterDict)):
                setattr(self, name, value)

    # -- Konstruktion ---------------------------------------------------------
    @classmethod
    def from_config_and_parameters(cls, config, train_methods, tunable_parameters, device="cpu"):
        instance = cls(config, train_methods, device)
        if isinstance(tunable_parameters, dict):
            tunable_parameters = migrate_serialized_potential_parameters(tunable_parameters, config)
        update_from_serialized(instance.tunable_parameters, tunable_parameters)
        instance.apply_trainable_selection()
        for name, value in vars(instance.tunable_parameters).items():
            if isinstance(value, (nn.Parameter, nn.ParameterList, nn.ParameterDict)):
                setattr(instance, name, value)
        return instance

    def get_tunable_parameters(self):
        return self.tunable_parameters

    def _collect_debye_waller(self):
        """Thermische Auslenkung je Element (u_rms in nm) aus den Schichten.

        Die Kernel werden pro Element vorberechnet, deshalb muss ein Element,
        das in mehreren Schichten vorkommt, dort dieselbe Auslenkung haben.
        """

        per_element = {}
        for layer, symbols in zip(self.layers, self.layer_elements):
            u_A = float(getattr(layer, "debye_waller_u", 0.0) or 0.0)
            if u_A < 0:
                raise ValueError(f"Schicht {layer.name}: debye_waller_u darf nicht negativ sein")
            for symbol in symbols:
                previous = per_element.get(symbol)
                if previous is not None and abs(previous - u_A) > 1e-12:
                    raise ValueError(
                        f"Element {symbol} bekommt in verschiedenen Schichten "
                        f"unterschiedliche debye_waller_u ({previous} vs {u_A} A). "
                        f"Der Kernel wird pro Element vorberechnet, das geht nicht.")
                per_element[symbol] = u_A

        return {symbol: u_A / 10.0 for symbol, u_A in per_element.items() if u_A > 0}

    @staticmethod
    def _resolve_elements(layer):
        """Elemente einer Schicht, nach Ordnungszahl sortiert."""

        symbols = list(getattr(layer, "elements", None) or [])
        if not symbols:
            return [layer.element]

        if layer.element not in symbols:
            raise ValueError(
                f"Schicht {layer.name}: element={layer.element!r} muss in "
                f"elements={symbols} enthalten sein (es liefert den Startwert "
                f"der Intensitaeten).")

        unknown = [s for s in symbols if s not in atomic_numbers]
        if unknown:
            raise ValueError(f"Schicht {layer.name}: unbekannte Elemente {unknown}")

        return sorted(set(symbols), key=lambda s: atomic_numbers[s])

    def mixes_elements(self, i: int) -> bool:
        return len(self.layer_elements[i]) > 1

    def element_mixture_weights(self, i: int, intensities):
        """Zerlegt die Intensitaet (= kontinuierliche Ordnungszahl) auf die Elemente.

        Stueckweise lineare Zeltfunktionen ueber die Ordnungszahlen: bei I = Z_k
        bekommt Element k das Gewicht 1, dazwischen wird linear gemischt.  Die
        Gewichte summieren sich immer zu 1, die Gesamtzahl der Atome bleibt also
        erhalten - anders als beim blossen Skalieren der Amplitude.
        """

        numbers = self.layer_atomic_numbers[i]
        clamped = torch.clamp(intensities, float(numbers[0]), float(numbers[-1]))

        weights = []
        for k in range(len(numbers)):
            left = (torch.ones_like(clamped) if k == 0
                    else (clamped - numbers[k - 1]) / (numbers[k] - numbers[k - 1]))
            right = (torch.ones_like(clamped) if k == len(numbers) - 1
                     else (numbers[k + 1] - clamped) / (numbers[k + 1] - numbers[k]))
            weights.append(torch.clamp(torch.minimum(left, right), 0.0, 1.0))

        return weights

    def _warn_intensities_out_of_range(self, i, intensities):
        if not hasattr(self, "_range_warned"):
            self._range_warned = set()
        numbers = self.layer_atomic_numbers[i]
        name = self.layers[i].name
        outside = bool(((intensities < numbers[0]) | (intensities > numbers[-1])).any())
        if outside and name not in self._range_warned:
            print(f"[Warnung] Schicht {name}: Intensitaeten liegen ausserhalb von "
                  f"[{float(numbers[0]):.0f}, {float(numbers[-1]):.0f}] und werden geklemmt - "
                  f"dort ist der Gradient null. Elementliste erweitern?")
            self._range_warned.add(name)
        elif not outside:
            self._range_warned.discard(name)

    def group_for_layer(self, i: int):
        return self.group_of_layer[i]

    def layer_indices(self, group_name: str):
        return self.layer_indices_by_group.get(group_name, [])

    def initialize_group_parameters(self):
        shifts = nn.ParameterDict()
        rotations = nn.ParameterDict()
        scales = nn.ParameterDict()
        factors = nn.ParameterDict()

        for group in self.groups:
            shifts[group.name] = nn.Parameter(
                torch.tensor(list(group.init_shift), dtype=torch.float32, device=self.device))
            rotations[group.name] = nn.Parameter(
                torch.tensor(float(group.init_rotation), dtype=torch.float32, device=self.device))
            scales[group.name] = nn.Parameter(
                torch.tensor(list(group.init_scale), dtype=torch.float32, device=self.device))
            factors[group.name] = nn.Parameter(
                torch.tensor(float(group.init_contrast_factor), dtype=torch.float32, device=self.device))

        return shifts, rotations, scales, factors

    def initialize_position_shifts_parameters(self, init_positions):
        positions_ = [torch.tensor(np.asarray(pos), dtype=torch.float32, device=self.device)
                      for pos in init_positions]
        for i, t in enumerate(positions_):
            self.register_buffer(f"init_positions_{i}", t)

        position_shifts_ = nn.ParameterList(
            [torch.zeros_like(pos) for pos in positions_])

        for i, layer in enumerate(self.layers):
            position_shifts_[i].requires_grad = self.train_methods.layer_enabled(
                layer.name, "free_positions")

        return position_shifts_

    def initialize_intensities_parameters(self, init_intensities):
        init_intensities = [np.asarray(I) for I in init_intensities]

        intensities_ = nn.ParameterList(
            [torch.tensor(I, dtype=torch.float32, device=self.device) for I in init_intensities])

        # Startwerte merken: die Intensitaeten wirken relativ dazu (siehe
        # get_atom_weights), damit der Startzustand Gewicht = Occupancy bleibt.
        for i, I in enumerate(init_intensities):
            self.register_buffer(f"init_intensities_{i}",
                                 torch.tensor(I, dtype=torch.float32, device=self.device))

        occupancies_ = nn.ParameterList([
            torch.ones(init_intensities[i].shape, dtype=torch.float32, device=self.device)
            * float(self.layers[i].init_occupancy)
            for i in range(self.N_l)
        ])

        return intensities_, occupancies_

    def get_init_positions(self, i):
        return getattr(self, f"init_positions_{i}")

    def get_init_intensities(self, i):
        return getattr(self, f"init_intensities_{i}")

    # -- Trainierbarkeit ------------------------------------------------------
    def apply_trainable_selection(self):
        tp = self.tunable_parameters
        tm = self.train_methods

        self._warn_degenerate_weights()

        for i, layer in enumerate(self.layers):
            tp.intensities[i].requires_grad = tm.layer_enabled(layer.name, "intensities")
            tp.occupancies[i].requires_grad = tm.layer_enabled(layer.name, "occupancies")
            tp.position_shifts[i].requires_grad = tm.layer_enabled(layer.name, "free_positions")

        for group in self.groups:
            tp.shifts[group.name].requires_grad = tm.group_enabled(group.name, "shift")
            tp.rotations[group.name].requires_grad = tm.group_enabled(group.name, "rotation")
            tp.scales[group.name].requires_grad = tm.group_enabled(group.name, "scale")
            tp.contrast_factors[group.name].requires_grad = tm.group_enabled(group.name, "contrast")

            strain_enabled = tm.group_enabled(group.name, "strain_field")
            for p in self.strain_fields[group.name].parameters():
                p.requires_grad = strain_enabled

    def _warn_degenerate_weights(self):
        """Intensitaet und Occupancy multiplizieren dasselbe Gewicht.

        Beide gleichzeitig freizugeben ist exakt entartet - sie koennen sich
        beliebig gegeneinander verschieben, ohne dass sich das Bild aendert.
        """

        if not hasattr(self, "_degenerate_warned"):
            self._degenerate_warned = set()

        tm = self.train_methods
        for layer in self.layers:
            if self.mixes_elements(self.layer_names.index(layer.name)):
                # dort waehlt die Intensitaet das Element, sie skaliert nicht
                conflicts = []
            else:
                conflicts = [("intensities", "occupancies")] if (
                    tm.layer_enabled(layer.name, "intensities")
                    and tm.layer_enabled(layer.name, "occupancies")) else []

            if tm.group_enabled(layer.group, "contrast") and tm.layer_enabled(layer.name, "occupancies"):
                conflicts.append((f"Gruppe {layer.group}: contrast", "occupancies"))

            if conflicts and layer.name not in self._degenerate_warned:
                for a, b in conflicts:
                    print(f"[Warnung] Schicht {layer.name}: {a} UND {b} sind gleichzeitig "
                          f"trainierbar - beide skalieren dasselbe Potential und sind "
                          f"gegeneinander entartet. Besser nur eins davon freigeben.")
                self._degenerate_warned.add(layer.name)
            elif not conflicts:
                self._degenerate_warned.discard(layer.name)

    def zero_grads_for_frozen(self):
        tp = self.tunable_parameters
        tm = self.train_methods

        for i, layer in enumerate(self.layers):
            if not tm.layer_enabled(layer.name, "intensities"):
                tp.intensities[i].grad = None
            if not tm.layer_enabled(layer.name, "occupancies"):
                tp.occupancies[i].grad = None
            if not tm.layer_enabled(layer.name, "free_positions"):
                tp.position_shifts[i].grad = None

        for group in self.groups:
            if not tm.group_enabled(group.name, "shift"):
                tp.shifts[group.name].grad = None
            if not tm.group_enabled(group.name, "rotation"):
                tp.rotations[group.name].grad = None
            if not tm.group_enabled(group.name, "scale"):
                tp.scales[group.name].grad = None
            if not tm.group_enabled(group.name, "contrast"):
                tp.contrast_factors[group.name].grad = None
            if not tm.group_enabled(group.name, "strain_field"):
                for p in self.strain_fields[group.name].parameters():
                    p.grad = None

    # -- Geometrie ------------------------------------------------------------
    def get_positions(self):
        tp = self.tunable_parameters
        tm = self.train_methods
        center = torch.tensor((self.H / 2, self.W / 2), device=self.device)

        ret_positions = []
        for i, layer in enumerate(self.layers):
            group = self.group_of_layer[i]
            gname = group.name

            theta = tp.rotations[gname]
            if not tm.group_enabled(gname, "rotation"):
                theta = theta.detach()
            if group.max_rotation is not None:
                theta = torch.clamp(theta, -group.max_rotation, group.max_rotation)
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            shift = tp.shifts[gname]
            if not tm.group_enabled(gname, "shift"):
                shift = shift.detach()
            if group.max_shift is not None:
                shift = torch.clamp(shift, -group.max_shift, group.max_shift)

            position_shifts = tp.position_shifts[i]
            if not tm.layer_enabled(layer.name, "free_positions"):
                position_shifts = position_shifts.detach()

            positions = self.get_init_positions(i) + position_shifts
            centered_positions = positions - center

            rotated_positions = torch.zeros_like(centered_positions)
            rotated_positions[:, 0] = cos_theta * centered_positions[:, 0] - sin_theta * centered_positions[:, 1]
            rotated_positions[:, 1] = sin_theta * centered_positions[:, 0] + cos_theta * centered_positions[:, 1]

            if tm.group_enabled(gname, "scale"):
                scales = torch.clamp(tp.scales[gname], min=0.80, max=1.2)
            else:
                scales = tp.scales[gname].detach()

            rotated_positions = rotated_positions * scales
            new_positions = rotated_positions + center + shift
            new_positions = self.strain_fields[gname](new_positions)

            ret_positions.append(new_positions)

        return ret_positions

    def group_elastic_loss(self, group_name: str, lambda_=1.0, mu=1.0):
        """Elastische Energie des Verschiebungsfeldes einer Gruppe."""

        indices = self.layer_indices(group_name)
        if not indices:
            return torch.zeros((), device=self.device)

        positions = self.get_positions()
        coords = torch.cat([positions[i].detach() for i in indices], dim=0)
        coords = coords.clone().detach().requires_grad_(True)

        energy = self.strain_fields[group_name].elastic_energy_density(
            coords, lambda_=lambda_, mu=mu)
        return energy.mean()

    def occupy_layers_in_area(self, mask, layers=None):
        """Setzt die Occupancy aller Atome innerhalb ``mask`` auf ~0.

        ``layers`` sind Schicht-Indizes oder -Namen; ohne Angabe alle Schichten.
        """

        if layers is None:
            layers = range(self.N_l)
        indices = [self.layer_names.index(l) if isinstance(l, str) else int(l) for l in layers]

        positions = self.get_positions()
        for l in indices:
            pos = positions[l]
            xs, ys = pos[:, 0], pos[:, 1]
            for i in range(pos.shape[0]):
                x, y = int(xs[i]), int(ys[i])
                if 0 <= x < self.H and 0 <= y < self.W:
                    if mask[x, y]:
                        self.tunable_parameters.occupancies[l][i] = -100

    # -- Intensitaeten --------------------------------------------------------
    def get_atom_weights(self):
        """Gewicht je Atom fuer die Rasterung des Potentials.

        ``Gewicht = (I / I_start) * sigmoid(occupancy)``

        Die Elementstaerke steckt bereits im Kirkland-Kernel der Schicht,
        deshalb wirkt die Intensitaet *relativ* zu ihrem Startwert (typisch der
        Ordnungszahl): am Anfang ist der Quotient exakt 1, das Gewicht also
        genau die Occupancy wie bisher.  Ein Wert von 6 -> 7 heisst dann
        "streut wie Stickstoff statt Kohlenstoff" - so lassen sich Fremdatome
        modellieren, ohne die Skalierung des ganzen Potentials zu verschieben.
        """

        tp = self.tunable_parameters
        tm = self.train_methods

        weights = []
        for i, layer in enumerate(self.layers):
            occupancy = tp.occupancies[i].sigmoid()
            if not tm.layer_enabled(layer.name, "occupancies"):
                occupancy = occupancy.detach()

            if self.mixes_elements(i):
                # dort waehlt die Intensitaet das Element aus (Summe der
                # Mischgewichte ist 1), sie skaliert nicht die Amplitude
                weights.append(occupancy)
                continue

            relative = tp.intensities[i] / self.get_init_intensities(i)
            if not tm.layer_enabled(layer.name, "intensities"):
                relative = relative.detach()

            weights.append(relative * occupancy)

        return weights

    #: alter Name
    get_intensities = get_atom_weights

    def get_intensities_raw(self):
        return self.tunable_parameters.intensities

    def _layer_index(self, layer):
        return self.layer_names.index(layer) if isinstance(layer, str) else int(layer)

    def get_pos_and_ints(self, pos_idx, filter_by_area=False, occ_cutoff=0.5):
        pos_idx = self._layer_index(pos_idx)

        intensities = self.get_intensities_raw()[pos_idx].detach().clone().cpu().numpy()
        occ = self.tunable_parameters.occupancies[pos_idx].sigmoid().detach().cpu().numpy()
        occ_neg_mask = occ < occ_cutoff
        intensities[occ_neg_mask] = None
        positions = self.get_positions()[pos_idx].cpu().detach().numpy()

        if filter_by_area:
            mask_img = self.train_methods.mask_for(self.train_methods.active_group)
            if mask_img is not None:
                mask_img = mask_img.cpu().numpy()
                inside = ((positions[:, 0] >= 0) & (positions[:, 0] < mask_img.shape[0]) &
                          (positions[:, 1] >= 0) & (positions[:, 1] < mask_img.shape[1]))
                positions = positions[inside]
                intensities = intensities[inside]
                mask = mask_img[positions[:, 0].astype(int), positions[:, 1].astype(int)] > 0
                positions = positions[mask]
                intensities = intensities[mask]

        return (positions[:, 0], positions[:, 1]), intensities

    def _occupied_mask(self, i, occ_cutoff, positions=None):
        occ = self.tunable_parameters.occupancies[i].sigmoid().detach().cpu().numpy()
        occ_mask = occ > occ_cutoff
        if positions is None:
            positions = self.get_positions()[i].cpu().detach().numpy()
        pad = self.config.padding
        limit_mask = ((positions[:, 0] >= pad) & (positions[:, 0] < self.H + pad) &
                      (positions[:, 1] >= pad) & (positions[:, 1] < self.W + pad))
        return occ_mask & limit_mask, positions

    def occupied_free_shifts(self, occ_cutoff=0.5):
        shifts = []
        pos_list = []
        for i in range(self.N_l):
            mask, positions = self._occupied_mask(i, occ_cutoff)
            shi = self.tunable_parameters.position_shifts[i]
            shifts.append(shi.cpu().detach().numpy()[mask])
            pos_list.append(positions[mask])

        return pos_list, shifts

    def occupied_positions_A(self, occ_cutoff=0.5, groups=None, exclude_groups=()):
        """Besetzte Atompositionen in Angstroem (fuer den XYZ-Export).

        ``groups``: nur diese Gruppen exportieren (None = alle).
        ``exclude_groups``: diese Gruppen ueberspringen.
        """

        pos_list = []
        Z_list = []
        for i, layer in enumerate(self.layers):
            if groups is not None and layer.group not in groups:
                continue
            if layer.group in exclude_groups:
                continue

            mask, positions = self._occupied_mask(i, occ_cutoff)

            filtered_positions = positions[mask] * self.config.pixel_size_nm * 10
            z_pos_A = float(layer.z_nm) * 10
            z_poses = np.full(filtered_positions.shape[0], z_pos_A)
            full_positions = np.hstack((filtered_positions, z_poses[:, np.newaxis]))
            pos_list.append(full_positions)
            Z_list.append(self.get_intensities_raw()[i][mask].cpu().detach().numpy())

        if not pos_list:
            return np.zeros((0, 3)), np.zeros((0,))
        return np.concatenate(pos_list), np.concatenate(Z_list)

    # -- Loss / Forward -------------------------------------------------------
    def aux_loss(self):
        tm = self.train_methods

        entropies = []
        for i, layer in enumerate(self.layers):
            if not layer.occ_entropy:
                continue
            p = self.tunable_parameters.occupancies[i].sigmoid()
            entropies.append(-(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).mean())
        entropy = torch.mean(torch.stack(entropies)) if entropies else 0.0

        if tm.position_shifts_regularization_a > 0.0:
            pos_shift_reg_stack = [
                torch.abs(self.tunable_parameters.position_shifts[i]).mean()
                for i, layer in enumerate(self.layers)
                if tm.layer_enabled(layer.name, "free_positions")
            ]
            pos_shift_reg = torch.mean(torch.stack(pos_shift_reg_stack)) if pos_shift_reg_stack else 0.0
        else:
            pos_shift_reg = 0.0

        if tm.strain_energy_a > 0.0:
            energies = [
                self.group_elastic_loss(group.name,
                                        lambda_=tm.strain_energy_lambda,
                                        mu=tm.strain_energy_mu)
                for group in self.groups
                if tm.group_enabled(group.name, "strain_field")
            ]
            strain_energy = torch.mean(torch.stack(energies)) if energies else 0.0
        else:
            strain_energy = 0.0

        return (tm.occ_entropy_a * entropy +
                tm.position_shifts_regularization_a * pos_shift_reg +
                tm.strain_energy_a * strain_energy)

    def active_layer_mask(self):
        """Welche Schichten tragen aktuell zum Potential bei?

        Ist eine Gruppe aktiv, die ``isolate_when_active`` gesetzt hat, werden
        alle uebrigen Schichten auf Null gesetzt (frueher: hart verdrahtet auf
        ``only_train_small_area == 0`` -> nur Graphen).
        """

        active = getattr(self.train_methods, "active_group", None)
        if active is None or active not in self.groups_by_name:
            return [True] * self.N_l
        if not self.groups_by_name[active].isolate_when_active:
            return [True] * self.N_l
        return [layer.group == active for layer in self.layers]

    def layer_potential(self, i, positions_i, weight_i):
        """Projiziertes Potential einer Schicht.

        Ohne Elementliste wie bisher: ein Kernel, die Intensitaet skaliert
        seine Amplitude.  Mit Elementliste wird die Intensitaet als
        kontinuierliche Ordnungszahl gelesen und auf die Nachbarelemente
        aufgeteilt - jedes mit seinem eigenen Kirkland-Kernel.
        """

        layer = self.layers[i]

        if not self.mixes_elements(i):
            return self.projected_slice_potential_kirkland(
                positions_i, weight_i, atom_type=layer.element)

        tp = self.tunable_parameters
        intensities = tp.intensities[i]
        trains_intensities = self.train_methods.layer_enabled(layer.name, "intensities")
        if not trains_intensities:
            intensities = intensities.detach()

        self._warn_intensities_out_of_range(i, intensities.detach())
        mixture = self.element_mixture_weights(i, intensities)

        potential = None
        for symbol, share in zip(self.layer_elements[i], mixture):
            weight = share * weight_i
            # Elemente ohne Beitrag ueberspringen - aber nur, wenn die
            # Intensitaeten fest sind: sonst faehrt man sich den Gradienten weg,
            # mit dem die Mischung ueberhaupt erst in Bewegung kommt
            if not trains_intensities and not bool(torch.any(weight > 0)):
                continue
            part = self.projected_slice_potential_kirkland(
                positions_i, weight, atom_type=symbol, subtract_median=False)
            potential = part if potential is None else potential + part

        if potential is None:
            return torch.zeros((self.H + self.potential_pad * 2,
                                self.W + self.potential_pad * 2),
                               dtype=torch.float32, device=self.device)

        return potential - potential.median()

    def forward(self, _):
        positions = self.get_positions()
        weights = self.get_atom_weights()
        contributes = self.active_layer_mask()

        potential_list = []
        for i, layer in enumerate(self.layers):
            if not contributes[i]:
                # None statt eines Nullbildes: der Multislice ueberspringt die
                # Schicht dann komplett (spart Transmission und Propagation)
                potential_list.append(None)
                continue

            potential_i = self.layer_potential(i, positions[i], weights[i])
            contrast = self.tunable_parameters.contrast_factors[layer.group]
            if not self.train_methods.group_enabled(layer.group, "contrast"):
                contrast = contrast.detach()
            potential_i = potential_i * contrast
            potential_list.append(potential_i)

        return potential_list, self.aux_loss()

    def get_potential_images(self):
        """Potentialbilder in Schicht-Reihenfolge (``config.layers``).

        Nicht beitragende Schichten (isolierte Gruppe) liefern ein Nullbild.
        """

        pot_list, _ = self(None)
        pad = self.potential_pad
        empty = np.zeros((self.H, self.W), dtype=np.float32)
        return [empty.copy() if p is None
                else p.detach().cpu().numpy()[pad:self.H + pad, pad:self.W + pad]
                for p in pot_list]

    def rasterize_atoms_bilinear(self, H, W, positions_xy, weights):
        rho = torch.zeros((H, W), dtype=torch.float32, device=self.device)

        x = positions_xy[:, 0]
        y = positions_xy[:, 1]

        mask = (x >= 0.0) & (x < H - 1) & (y >= 0.0) & (y < W - 1)
        x = x[mask]
        y = y[mask]
        w = weights[mask]

        if x.numel() == 0:
            return rho

        ix = torch.floor(x).long()
        iy = torch.floor(y).long()

        fx = x - ix
        fy = y - iy

        w00 = (1 - fx) * (1 - fy) * w
        w10 = fx * (1 - fy) * w
        w01 = (1 - fx) * fy * w
        w11 = fx * fy * w

        #rho.index_put_((iy, ix), w00, accumulate=True)
        #rho.index_put_((iy + 1, ix), w01, accumulate=True)
        #rho.index_put_((iy, ix + 1), w10, accumulate=True)
        #rho.index_put_((iy + 1, ix + 1), w11, accumulate=True)

        rho.index_put_((ix, iy), w00, accumulate=True)
        rho.index_put_((ix, iy + 1), w01, accumulate=True)
        rho.index_put_((ix + 1, iy), w10, accumulate=True)
        rho.index_put_((ix + 1, iy + 1), w11, accumulate=True)

        return rho




    def projected_slice_potential_lobato(self, positions_xy, occupancies, atom_type):
        pad = max(self.H // 2, self.W // 2)
        Hp = self.H + 2 * pad
        Wp = self.W + 2 * pad

        positions_pad = positions_xy + torch.tensor([pad, pad], device=self.device, dtype=positions_xy.dtype)

        rho = self.rasterize_atoms_bilinear(
            H=Hp,
            W=Wp,
            positions_xy=positions_pad,
            weights=occupancies
        )

        kernel = projected_atomic_potential_kernel_lobato(
            H=Hp,
            W=Wp,
            pixel_size_nm=self.pixel_size_nm,
            atom_type=atom_type,
            device=self.device,
        )

        Vp = torch.fft.ifft2(torch.fft.fft2(rho) * torch.fft.fft2(kernel)).real
        V = Vp[pad:pad + self.H, pad:pad + self.W]


        # globaler Offset ist für TEM-Intensität nicht relevant
        V = V - V.median()
        return V

    def projected_slice_potential_kirkland(self, positions_xy, occupancies, atom_type,
                                           subtract_median=True):
        pad = self.potential_pad#max(self.H // 2, self.W // 2)
        Hp = self.H + 2 * pad
        Wp = self.W + 2 * pad

        positions_pad = positions_xy + torch.tensor([pad, pad], device=self.device, dtype=positions_xy.dtype)

        rho = self.rasterize_atoms_bilinear(
            H=Hp,
            W=Wp,
            positions_xy=positions_pad,
            weights=occupancies
        )


        Vp = self.kirkland_precalc_pot.projected_slice_potential_kirkland(rho, atom_type, self.pixel_size_nm, self.device)
        V = Vp#[pad:pad + self.H, pad:pad + self.W]
        if subtract_median:
            # konstanter Offset - physikalisch irrelevant (globale Phase), haelt
            # aber die Zahlen klein. Bei einer Elementmischung erst am Ende.
            V = V - V.median()

        return V

