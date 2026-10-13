"""Atompositionen aus einer Gitterbeschreibung erzeugen.

Gemeinsame Grundlage fuer die Mains und fuer das ImageJ-Plugin: aus Gittertyp,
Gitterkonstante, Orientierung und Versatz werden die Startpositionen der Atome
je Schicht gebaut - in Pixelkoordinaten des Bildes.

Unterstuetzte Typen:
    "graphene"  Wabengitter, ein Schicht
    "mx2"       2H-Uebergangsmetalldichalkogenid (WS2, MoS2 ...), drei Schichten
    "custom"    eigene Zelle: a, b, gamma und eine Basis aus (Element, fx, fy, z)
    "file"      Struktur aus .cif/.xyz/... laden und nach z in Schichten schneiden
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LATTICE_TYPES = ("graphene", "mx2", "custom", "file")

#: Gitterkonstanten in Angstroem
DEFAULT_LATTICE_CONSTANT = {"graphene": 2.46, "mx2": 3.18}


@dataclass
class LatticeSpec:
    """Beschreibung des Startgitters."""

    type: str = "graphene"

    #: Gitterkonstante a in Angstroem (bei "file" ignoriert)
    a: float = 2.46
    #: zweite Gitterkonstante; None -> gleich a
    b: Optional[float] = None
    #: Winkel zwischen den Gittervektoren in Grad (hexagonal: 120)
    gamma_deg: float = 120.0

    #: fuer "mx2": Formel, z.B. "WS2"
    formula: str = "WS2"
    #: fuer "file": Pfad zur Strukturdatei
    file: Optional[str] = None

    #: Basis fuer "custom": (Element, frac_a, frac_b, z_nm)
    basis: List[Tuple[str, float, float, float]] = field(default_factory=list)

    #: Orientierung und Lage im Bild
    rotation_deg: float = 0.0
    offset_px: Tuple[float, float] = (0.0, 0.0)

    #: z-Abstaende: bei "file"/"mx2" aus der Struktur, sonst aus der Basis
    layer_names: Optional[List[str]] = None

    def lattice_vectors(self) -> np.ndarray:
        """Gittervektoren in Angstroem, Zeilen sind a1 und a2."""

        a = float(self.a)
        b = float(self.b if self.b else self.a)
        gamma = math.radians(self.gamma_deg)
        return np.array([[a, 0.0],
                         [b * math.cos(gamma), b * math.sin(gamma)]])


def _graphene_basis() -> List[Tuple[str, float, float, float]]:
    # hexagonale Zelle mit gamma = 120 Grad, zwei Kohlenstoffe
    return [("C", 0.0, 0.0, 0.0), ("C", 1.0 / 3.0, 2.0 / 3.0, 0.0)]


def _mx2_basis(formula: str):
    """Basis und z-Positionen eines 2H-MX2 aus ase.build.mx2."""

    import ase.build

    atoms = ase.build.mx2(formula=formula, vacuum=1.0)
    cell = atoms.cell.lengths()
    positions = atoms.get_positions()
    z = positions[:, 2] - positions[:, 2].min()

    scaled = atoms.get_scaled_positions()
    basis = [(atoms[i].symbol, float(scaled[i, 0]), float(scaled[i, 1]), float(z[i]) / 10.0)
             for i in range(len(atoms))]
    return basis, float(cell[0])


def resolve_basis(spec: LatticeSpec):
    """Basis (Element, frac_a, frac_b, z_nm) und Gitterkonstante der Zelle.

    Eine explizit angegebene Basis gewinnt immer - so rechnet das ImageJ-Plugin
    fuer sein Overlay garantiert mit derselben Zelle wie die Simulation.
    """

    if spec.basis:
        return [tuple(entry) for entry in spec.basis], spec.a

    if spec.type == "graphene":
        return _graphene_basis(), spec.a or DEFAULT_LATTICE_CONSTANT["graphene"]

    if spec.type == "mx2":
        basis, a = _mx2_basis(spec.formula)
        return basis, (spec.a or a)

    if spec.type == "custom":
        if not spec.basis:
            raise ValueError("custom-Gitter braucht eine Basis")
        return [tuple(entry) for entry in spec.basis], spec.a

    raise ValueError(f"resolve_basis kennt {spec.type!r} nicht")


def layers_from_basis(basis, tolerance_nm: float = 0.01):
    """Basisatome nach z-Position zu Schichten zusammenfassen."""

    order = sorted(range(len(basis)), key=lambda i: basis[i][3])
    layers: List[Dict] = []
    for i in order:
        symbol, fa, fb, z = basis[i]
        if layers and abs(layers[-1]["z_nm"] - z) <= tolerance_nm:
            layers[-1]["atoms"].append((symbol, fa, fb))
        else:
            layers.append({"z_nm": float(z), "atoms": [(symbol, fa, fb)]})
    return layers


def canvas_size_px(shape, max_shift_px: float, max_rotation_deg: float, factor: float = 1.2):
    """Wie gross das Gitter sein muss, damit es bei Verschiebung und Drehung deckt."""

    H, W = shape
    rot = math.radians(abs(max_rotation_deg))
    height = math.cos(rot) * H + math.sin(rot) * W + 2 * factor * max_shift_px
    width = math.cos(rot) * W + math.sin(rot) * H + 2 * factor * max_shift_px
    return int(math.ceil(height)), int(math.ceil(width))


def build_layer_positions(spec: LatticeSpec, shape, pixel_size_nm: float,
                          max_shift_px: float = 0.0, max_rotation_deg: float = 0.0):
    """Startpositionen je Schicht in Pixelkoordinaten.

    Liefert eine Liste von Dicts mit ``z_nm``, ``elements`` (Symbol je Atom) und
    ``positions`` (N x 2, Pixel, Bildkoordinaten [Zeile, Spalte]).
    """

    if spec.type == "file":
        return _positions_from_file(spec, shape, pixel_size_nm, max_shift_px, max_rotation_deg)

    basis, a = resolve_basis(spec)
    layers = layers_from_basis(basis)

    pixel_A = pixel_size_nm * 10.0
    canvas_h, canvas_w = canvas_size_px(shape, max_shift_px, max_rotation_deg)

    vectors = LatticeSpec(type=spec.type, a=a, b=spec.b, gamma_deg=spec.gamma_deg).lattice_vectors()
    vectors_px = vectors / pixel_A

    # so viele Zellen, dass die Canvas sicher gedeckt ist
    reach = int(math.ceil(1.5 * max(canvas_h, canvas_w) / max(np.linalg.norm(vectors_px, axis=1).min(), 1e-6)))
    grid = np.arange(-reach, reach + 1)
    ii, jj = np.meshgrid(grid, grid, indexing="ij")
    cells = np.stack([ii.ravel(), jj.ravel()], axis=-1).astype(np.float64)

    theta = math.radians(spec.rotation_deg)
    rotation = np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])

    centre = np.array([shape[0] / 2.0, shape[1] / 2.0])
    offset = np.asarray(spec.offset_px, dtype=np.float64)
    margin = np.array([(canvas_h - shape[0]) / 2.0, (canvas_w - shape[1]) / 2.0])

    out = []
    for layer in layers:
        positions = []
        elements = []
        for symbol, fa, fb in layer["atoms"]:
            xy = cells @ vectors_px + (fa * vectors_px[0] + fb * vectors_px[1])
            xy = xy @ rotation.T + centre + offset
            positions.append(xy)
            elements.extend([symbol] * len(xy))
        positions = np.concatenate(positions, axis=0)

        inside = ((positions[:, 0] >= -margin[0]) & (positions[:, 0] < shape[0] + margin[0]) &
                  (positions[:, 1] >= -margin[1]) & (positions[:, 1] < shape[1] + margin[1]))
        out.append({"z_nm": layer["z_nm"],
                    "positions": positions[inside],
                    "elements": [e for e, keep in zip(elements, inside) if keep]})
    return out


def _positions_from_file(spec: LatticeSpec, shape, pixel_size_nm, max_shift_px, max_rotation_deg):
    """Struktur aus einer Datei laden und nach z in Schichten schneiden."""

    import ase.io

    atoms = ase.io.read(spec.file)
    pixel_A = pixel_size_nm * 10.0
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    z = positions[:, 2] - positions[:, 2].min()
    xy = positions[:, :2] / pixel_A

    theta = math.radians(spec.rotation_deg)
    rotation = np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])
    centre = np.array([shape[0] / 2.0, shape[1] / 2.0])
    xy = (xy - xy.mean(axis=0)) @ rotation.T + centre + np.asarray(spec.offset_px)

    layers = []
    for value in sorted(set(np.round(z, 2))):
        mask = np.abs(np.round(z, 2) - value) < 1e-9
        layers.append({"z_nm": float(value) / 10.0,
                       "positions": xy[mask],
                       "elements": [symbols[i] for i in np.where(mask)[0]]})
    return layers
