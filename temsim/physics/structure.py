r"""Part I, Section 4: atomic structure.

Theory
======

From atoms to a density map
---------------------------
A layer :math:`\ell` is a set of atoms at positions :math:`\mathbf r_i`
(pixel coordinates, rows first) with weights :math:`w_i`.  The projected potential
of the layer is the superposition of the atomic potentials :math:`v(\mathbf r)`
(Section 3),

.. math::

    V_\ell(\mathbf r) = \sum_i w_i\; v(\mathbf r - \mathbf r_i)
                      = (\rho_\ell * v)(\mathbf r),
    \qquad
    \rho_\ell(\mathbf r) = \sum_i w_i\,\delta(\mathbf r - \mathbf r_i).
    \qquad (4.1)

On the pixel grid the density :math:`\rho_\ell` is built by **bilinear
rasterisation**: an atom at :math:`(y, x)` with :math:`y = j + f_y`,
:math:`x = i + f_x` (integer parts :math:`j, i`, fractions :math:`f`) deposits its
weight on the four neighbouring pixels with the weights
:math:`(1-f_y)(1-f_x)`, :math:`(1-f_y)f_x`, :math:`f_y(1-f_x)`, :math:`f_y f_x`.
This preserves the total weight and the centre of mass exactly and makes the map
differentiable with respect to the positions.  The convolution of Eq. (4.1) is a
product in Fourier space,

.. math::

    \tilde V_\ell(\mathbf k) = \tilde\rho_\ell(\mathbf k)\,\tilde v(\mathbf k),
    \qquad (4.2)

evaluated with real FFTs (:func:`rasterise`; the convolution itself lives in
:mod:`temsim.physics.multislice`).  The grid is *padded* by a margin so that atoms
near the border contribute correctly to the image area; the periodicity of the FFT
then wraps only vacuum around.  An atom whose rasterisation cell does not lie
completely inside the padded grid is dropped.

Atom weights
------------
The weight of an atom combines two continuous numbers, both of which can be
optimised: the **occupancy** :math:`p_i = \sigma(o_i) \in (0,1)` (logistic of the
logit :math:`o_i`; "is there an atom?") and the **effective atomic number**
:math:`Z_i` ("which atom?").  Two interpretations of :math:`Z_i` exist.

*Single-element layer.*  The kernel is that of the layer's element with reference
number :math:`Z_i^0` (its start value), and :math:`Z_i` scales the amplitude:

.. math:: w_i = \frac{Z_i}{Z_i^0}\; p_i. \qquad (4.3)

At the start this is exactly :math:`p_i`; "from 6 to 7" means "scatters like N
instead of C" to first order, without distorting the scale of the whole potential.

*Mixed layer* (list of elements :math:`E_1 < E_2 < \dots` with atomic numbers
:math:`N_1 < N_2 < \dots`).  Here :math:`Z_i` selects the element by piecewise
linear interpolation ("tent functions", partition of unity):

.. math::

    m_k(Z) = \max\Big(0,\ \min\Big(\tfrac{Z - N_{k-1}}{N_k - N_{k-1}},\
              \tfrac{N_{k+1} - Z}{N_{k+1} - N_k}\Big)\Big),
    \qquad \sum_k m_k(Z) = 1,
    \qquad (4.4)

and the layer is the sum of one density per element, :math:`\rho_k = \sum_i
m_k(Z_i)\, p_i\,\delta(\cdot - \mathbf r_i)`, each convolved with its own kernel.
The number of atoms is conserved by the partition of unity, unlike with Eq. (4.3).
:math:`Z` outside :math:`[N_1, N_{\mathrm{last}}]` is clamped (gradient zero there).

Rigid group transformation
--------------------------
Atoms of one group move together.  With the group shift :math:`\mathbf s`,
rotation angle :math:`\theta`, and per-axis scale :math:`(s_y, s_x)`, about the
centre :math:`\mathbf c = (H/2, W/2)` of the image area,

.. math::

    \mathbf r' = \mathbf c + \mathbf s + S\,R(\theta)\,(\mathbf r - \mathbf c),
    \qquad
    R = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix},
    \quad S = \operatorname{diag}(s_y, s_x).
    \qquad (4.5)

The smooth strain field of Section 5 is applied afterwards (:func:`transform`).
Bounds on :math:`\mathbf s`, :math:`\theta`, :math:`S` are optimisation constraints
and belong to Part II.

Building initial lattices
-------------------------
Starting structures come from a lattice recipe (:class:`~temsim.data.schema.LatticeSpec`):
a 2D cell with vectors :math:`\mathbf a_1 = a(1, 0)`,
:math:`\mathbf a_2 = b(\cos\gamma, \sin\gamma)` and a basis of
``(element, f_1, f_2, z)``; atom positions are
:math:`\mathbf R_{mn} + f_1\mathbf a_1 + f_2\mathbf a_2` for all lattice points
:math:`\mathbf R_{mn} = m\mathbf a_1 + n\mathbf a_2` covering the image (plus a
margin that allows for shifts and rotation of the group), rotated about the image
centre and shifted by an offset.  Basis atoms with equal height (within 0.01 nm)
form one layer.  Types: ``graphene`` (two C atoms, hexagonal),
``mx2`` (2H dichalcogenide from ``ase.build.mx2``, three layers), ``custom``
(explicit basis) and ``file`` (any ASE-readable structure, sliced by height).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data.schema import LatticeSpec
from . import strain as strain_module

# ---------------------------------------------------------------------------
# Atoms of one layer
# ---------------------------------------------------------------------------


@dataclass
class AtomLayer:
    """The physical content of one slice: positions and weights as torch tensors.

    ``positions`` are the *effective* positions in the unpadded pixel frame
    (after shifts, group transformation, strain).  ``weights`` already contain
    occupancy (and, for single-element layers, the relative atomic number,
    Eq. 4.3).  For mixed layers ``atomic_number`` holds :math:`Z_i` and
    ``weights`` the occupancy only.
    """

    z_nm: float
    element: str
    positions: torch.Tensor
    weights: torch.Tensor
    elements: Sequence[str] = ()
    atomic_number: Optional[torch.Tensor] = None
    debye_waller_u_A: float = 0.0
    #: factor multiplying the layer's potential (the group's Stobbs factor)
    contrast: object = 1.0
    #: False -> the slice is skipped (e.g. an isolated group)
    active: bool = True

    @property
    def mixed(self) -> bool:
        return len(self.elements) > 1


ATOMIC_NUMBERS: Dict[str, int] = {}


def atomic_number_of(symbol: str) -> int:
    if not ATOMIC_NUMBERS:
        from ase.data import atomic_numbers
        ATOMIC_NUMBERS.update(atomic_numbers)
    if symbol not in ATOMIC_NUMBERS:
        raise KeyError(f"unknown element '{symbol}'")
    return ATOMIC_NUMBERS[symbol]


def sorted_elements(layer_element: str, elements: Sequence[str]) -> List[str]:
    """Elements of a mixed layer sorted by atomic number (``[layer_element]`` if not mixed)."""

    symbols = list(elements or [])
    if not symbols:
        return [layer_element]
    if layer_element not in symbols:
        raise ValueError(f"element {layer_element!r} must be in elements={symbols} "
                         f"(it provides the start value of the atomic numbers)")
    return sorted(set(symbols), key=atomic_number_of)


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

def mixture_shares(atomic_number: torch.Tensor, nodes: Sequence[float]) -> List[torch.Tensor]:
    """Eq. (4.4): shares of the elements with atomic numbers ``nodes`` (ascending)."""

    nodes_t = [float(n) for n in nodes]
    z = torch.clamp(atomic_number, nodes_t[0], nodes_t[-1])
    shares = []
    for k, n in enumerate(nodes_t):
        left = torch.ones_like(z) if k == 0 else (z - nodes_t[k - 1]) / (n - nodes_t[k - 1])
        right = torch.ones_like(z) if k == len(nodes_t) - 1 else (nodes_t[k + 1] - z) / (nodes_t[k + 1] - n)
        shares.append(torch.clamp(torch.minimum(left, right), 0.0, 1.0))
    return shares


def relative_weights(atomic_number: torch.Tensor, reference: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
    """Eq. (4.3)."""

    return atomic_number / reference * occupancy


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def transform(points: torch.Tensor, shape: Tuple[int, int], shift, rotation, scale,
              strain_control: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Eq. (4.5) followed by the strain displacement of Eq. (5.1)."""

    centre = torch.tensor((shape[0] / 2.0, shape[1] / 2.0), dtype=points.dtype, device=points.device)
    rel = points - centre
    cos, sin = torch.cos(rotation), torch.sin(rotation)
    rotated = torch.stack([cos * rel[:, 0] - sin * rel[:, 1],
                           sin * rel[:, 0] + cos * rel[:, 1]], dim=1)
    out = rotated * scale + centre + shift
    if strain_control is not None:
        out = strain_module.displace(strain_control, out, shape)
    return out


def rasterise(positions: torch.Tensor, weights: torch.Tensor, shape: Tuple[int, int], padding: int = 0) -> torch.Tensor:
    """Bilinear deposition of weighted points on the padded grid, shape ``(H + 2p, W + 2p)``."""

    Hp, Wp = shape[0] + 2 * padding, shape[1] + 2 * padding
    rho = torch.zeros((Hp, Wp), dtype=torch.float32, device=positions.device)
    y = positions[:, 0] + padding
    x = positions[:, 1] + padding
    keep = (y >= 0.0) & (y < Hp - 1) & (x >= 0.0) & (x < Wp - 1)
    y, x, w = y[keep], x[keep], weights[keep]
    if y.numel() == 0:
        return rho
    iy, ix = torch.floor(y).long(), torch.floor(x).long()
    fy, fx = y - iy, x - ix
    rho.index_put_((iy, ix), (1 - fy) * (1 - fx) * w, accumulate=True)
    rho.index_put_((iy, ix + 1), (1 - fy) * fx * w, accumulate=True)
    rho.index_put_((iy + 1, ix), fy * (1 - fx) * w, accumulate=True)
    rho.index_put_((iy + 1, ix + 1), fy * fx * w, accumulate=True)
    return rho


def element_densities(layer: AtomLayer, shape: Tuple[int, int], padding: int = 0) -> Dict[str, torch.Tensor]:
    """Density maps per element of one layer (Eq. 4.1/4.4); one map unless the layer is mixed."""

    if not layer.mixed:
        return {layer.element: rasterise(layer.positions, layer.weights, shape, padding)}
    nodes = [atomic_number_of(e) for e in layer.elements]
    shares = mixture_shares(layer.atomic_number, nodes)
    return {e: rasterise(layer.positions, share * layer.weights, shape, padding)
            for e, share in zip(layer.elements, shares)}


# ---------------------------------------------------------------------------
# lattices
# ---------------------------------------------------------------------------

LATTICE_TYPES = ("graphene", "mx2", "custom", "file")


@dataclass
class LatticeLayer:
    """One slice of a built lattice."""

    z_nm: float
    positions: np.ndarray                    # (n, 2) pixels, [row, column]
    elements: List[str] = field(default_factory=list)


def lattice_vectors(a: float, b: Optional[float], gamma_deg: float) -> np.ndarray:
    """Rows are :math:`\\mathbf a_1, \\mathbf a_2` in Angstrom."""

    b = float(b) if b else float(a)
    g = math.radians(gamma_deg)
    return np.array([[float(a), 0.0], [b * math.cos(g), b * math.sin(g)]])


def _mx2_basis(formula: str):
    import ase.build

    atoms = ase.build.mx2(formula=formula, vacuum=1.0)
    positions = atoms.get_positions()
    z = positions[:, 2] - positions[:, 2].min()
    scaled = atoms.get_scaled_positions()
    basis = [(atoms[i].symbol, float(scaled[i, 0]), float(scaled[i, 1]), float(z[i]) / 10.0)
             for i in range(len(atoms))]
    return basis, float(atoms.cell.lengths()[0])


def resolve_basis(spec: LatticeSpec):
    """Basis ``(element, f1, f2, z_nm)`` and lattice constant.  An explicit basis always wins."""

    if spec.basis:
        return [tuple(entry) for entry in spec.basis], spec.a
    if spec.type == "graphene":
        return [("C", 0.0, 0.0, 0.0), ("C", 1.0 / 3.0, 2.0 / 3.0, 0.0)], spec.a or 2.46
    if spec.type == "mx2":
        basis, a = _mx2_basis(spec.formula)
        return basis, (spec.a or a)
    raise ValueError(f"lattice type {spec.type!r} needs a basis")


def layers_from_basis(basis, tolerance_nm: float = 0.01):
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
    """Size the lattice must have to cover the image after the allowed shift and rotation."""

    H, W = shape
    rot = math.radians(abs(max_rotation_deg))
    return (int(math.ceil(math.cos(rot) * H + math.sin(rot) * W + 2 * factor * max_shift_px)),
            int(math.ceil(math.cos(rot) * W + math.sin(rot) * H + 2 * factor * max_shift_px)))


def _rotation_matrix(deg: float) -> np.ndarray:
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def build_lattice(spec: LatticeSpec, shape, pixel_size_nm: float, max_shift_px: float = 0.0,
                  max_rotation_deg: float = 0.0) -> List[LatticeLayer]:
    """Initial atoms per layer in pixel coordinates (see module docstring)."""

    if spec.type == "file":
        return _build_from_file(spec, shape, pixel_size_nm)
    if spec.type not in LATTICE_TYPES:
        raise ValueError(f"unknown lattice type {spec.type!r}, choose from {LATTICE_TYPES}")

    basis, a = resolve_basis(spec)
    pixel_A = pixel_size_nm * 10.0
    canvas_h, canvas_w = canvas_size_px(shape, max_shift_px, max_rotation_deg)
    vectors_px = lattice_vectors(a, spec.b, spec.gamma_deg) / pixel_A

    reach = int(math.ceil(1.5 * max(canvas_h, canvas_w) / max(np.linalg.norm(vectors_px, axis=1).min(), 1e-6)))
    grid = np.arange(-reach, reach + 1)
    ii, jj = np.meshgrid(grid, grid, indexing="ij")
    cells = np.stack([ii.ravel(), jj.ravel()], axis=-1).astype(np.float64)

    rotation = _rotation_matrix(spec.rotation_deg)
    centre = np.array([shape[0] / 2.0, shape[1] / 2.0])
    offset = np.asarray(spec.offset_px, dtype=np.float64)
    margin = np.array([(canvas_h - shape[0]) / 2.0, (canvas_w - shape[1]) / 2.0])

    out = []
    for layer in layers_from_basis(basis):
        positions, elements = [], []
        for symbol, fa, fb in layer["atoms"]:
            xy = cells @ vectors_px + (fa * vectors_px[0] + fb * vectors_px[1])
            positions.append(xy @ rotation.T + centre + offset)
            elements.extend([symbol] * len(xy))
        positions = np.concatenate(positions, axis=0)
        inside = ((positions[:, 0] >= -margin[0]) & (positions[:, 0] < shape[0] + margin[0]) &
                  (positions[:, 1] >= -margin[1]) & (positions[:, 1] < shape[1] + margin[1]))
        out.append(LatticeLayer(z_nm=layer["z_nm"] + spec.z_offset_nm, positions=positions[inside],
                                elements=[e for e, keep in zip(elements, inside) if keep]))
    return out


def _build_from_file(spec: LatticeSpec, shape, pixel_size_nm) -> List[LatticeLayer]:
    import ase.io

    atoms = ase.io.read(spec.file)
    pixel_A = pixel_size_nm * 10.0
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    z = positions[:, 2] - positions[:, 2].min()
    xy = positions[:, :2] / pixel_A
    centre = np.array([shape[0] / 2.0, shape[1] / 2.0])
    xy = (xy - xy.mean(axis=0)) @ _rotation_matrix(spec.rotation_deg).T + centre + np.asarray(spec.offset_px)
    layers = []
    for value in sorted(set(np.round(z, 2))):
        mask = np.abs(np.round(z, 2) - value) < 1e-9
        layers.append(LatticeLayer(z_nm=float(value) / 10.0, positions=xy[mask],
                                   elements=[symbols[i] for i in np.where(mask)[0]]))
    return layers
