r"""Part I, Section 5: the displacement (strain) field.

Theory
======

A real 2D crystal is never a perfect lattice: it is rippled, stretched by the
substrate and bent around defects.  Instead of one free shift per atom (thousands
of parameters, hopelessly ill-conditioned) the *smooth* part of the deformation is
described by a displacement field :math:`\mathbf u(\mathbf r)`, and an atom at
:math:`\mathbf r` is moved to

.. math:: \mathbf r' = \mathbf r + \mathbf u(\mathbf r)
          \qquad (5.1)

(the per-atom shifts of Part II then describe only what the field cannot).

Parametrisation
---------------
The field is sampled on a coarse control grid of :math:`H_c \times W_c` nodes,
which spans the *image area* :math:`[0, H-1]\times[0, W-1]` (pixel coordinates,
rows first).  Node :math:`(i,j)` sits at

.. math:: \mathbf r_{ij} = \Big(\tfrac{i}{H_c-1}(H-1),\; \tfrac{j}{W_c-1}(W-1)\Big),
          \qquad (5.2)

and carries the two components of :math:`\mathbf u` (row direction, column
direction).  Between the nodes the field is interpolated with a bicubic
convolution kernel (Keys with :math:`a=-1/2`, i.e. Catmull-Rom), which is
:math:`C^1`-continuous, passes through the node values and reproduces polynomials
up to second degree.  With the :math:`t = (n-1)\,r/(N-1)` coordinate in node units,
:math:`i = \lfloor t\rfloor`, :math:`f = t - i`, one axis reads

.. math::

    u(t) = \sum_{m=-1}^{2} u_{i+m}\, h_m(f), \quad
    h_{-1} = \tfrac12(-f^3 + 2f^2 - f),\;
    h_0 = \tfrac12(3f^3 - 5f^2 + 2),\;
    h_1 = \tfrac12(-3f^3 + 4f^2 + f),\;
    h_2 = \tfrac12(f^3 - f^2),

and the 2D field is the tensor product.  The stencil of the outermost cells needs
nodes beyond the grid; they are the *linear extrapolation* of the two outermost
nodes (:math:`u_{-1} = 2u_0 - u_1`), so that a uniform strain is reproduced
exactly up to the image border.  Outside the image the coordinate is clamped to
the border (constant displacement, no strain there).  The interpolation is plain
tensor algebra and hence differentiable to any order, which Eq. (5.4) needs
(derivative of the strain with respect to the control nodes).

The control tensor has shape ``(1, 2, Hc, Wc)``: channel 0 holds the row
component, channel 1 the column component.

Strain, rotation, energy
------------------------
With the displacement gradient :math:`J_{ij} = \partial u_i / \partial r_j`
(computed by automatic differentiation of the interpolation) the small-strain
tensor and the local rotation are

.. math::

    \varepsilon = \tfrac12 (J + J^{\mathsf T}), \qquad
    \omega = \tfrac12 (J - J^{\mathsf T})
    \qquad (5.3)

and the linear isotropic elastic energy density (Lamé parameters
:math:`\lambda, \mu`)

.. math::

    w = \tfrac12 \lambda\, (\operatorname{tr}\varepsilon)^2
        + \mu\, \varepsilon : \varepsilon.
    \qquad (5.4)

Part II uses :math:`\langle w\rangle` over the atoms as a regulariser: it penalises
stretching and shearing but not rigid rotation, which belongs to the group
transformation.  The relation to the trace, :math:`\operatorname{tr}\varepsilon =
\nabla\cdot\mathbf u`, is the local relative area change.

Refining the grid
-----------------
:func:`resample_control` re-samples an existing field on a finer or coarser node
grid.  Because the interpolation passes through the nodes, refining is lossless up
to the interpolation error between old nodes; coarsening loses structure the new
grid cannot represent.

Reading old states
------------------
Earlier program versions normalised the point coordinates by the *padded* image
size and sampled the control grid with transposed axes.  A saved strain state of
that kind is converted by :func:`from_legacy_control`, which evaluates the old
field at the nodes of the present convention (exact at the nodes; between them the
two interpolants differ by the interpolation error only).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _extended(control: torch.Tensor) -> torch.Tensor:
    """Control tensor ``(C, Hc, Wc)`` with one linearly extrapolated node on every side."""

    c = control[0]
    c = torch.cat([2 * c[:, :1] - c[:, 1:2], c, 2 * c[:, -1:] - c[:, -2:-1]], dim=1)
    return torch.cat([2 * c[:, :, :1] - c[:, :, 1:2], c, 2 * c[:, :, -1:] - c[:, :, -2:-1]], dim=2)


def _catmull_rom(f: torch.Tensor) -> torch.Tensor:
    f2, f3 = f * f, f * f * f
    return torch.stack([0.5 * (-f3 + 2 * f2 - f), 0.5 * (3 * f3 - 5 * f2 + 2),
                        0.5 * (-3 * f3 + 4 * f2 + f), 0.5 * (f3 - f2)], dim=-1)


def _axis(fraction: torch.Tensor, n: int):
    """Indices ``(N, 4)`` into the extended grid and weights ``(N, 4)`` for node-fractions in [0, 1]."""

    t = torch.clamp(fraction, 0.0, 1.0) * (n - 1)
    i = torch.clamp(torch.floor(t.detach()), 0, n - 2)
    f = t - i
    index = i.long().unsqueeze(-1) + torch.arange(4, device=t.device)       # extended index = node + 1 + m
    return index, _catmull_rom(f)


def sample_fraction(control: torch.Tensor, row_fraction: torch.Tensor, col_fraction: torch.Tensor) -> torch.Tensor:
    """Field at positions given as fractions of the node domain, ``(N,)`` each -> ``(N, 2)``."""

    ext = _extended(control)
    iy, wy = _axis(row_fraction.to(control.dtype), control.shape[-2])
    ix, wx = _axis(col_fraction.to(control.dtype), control.shape[-1])
    patch = ext[:, iy[:, :, None], ix[:, None, :]]                          # (2, N, 4, 4)
    return torch.einsum("cnab,na,nb->nc", patch, wy, wx)


def displacement(control: torch.Tensor, points: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """:math:`\\mathbf u` at ``points`` ``(N, 2)``; returns ``(N, 2)`` (row, column component)."""

    H, W = shape
    return sample_fraction(control, points[:, 0] / max(H - 1, 1), points[:, 1] / max(W - 1, 1))


def displace(control: torch.Tensor, points: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Eq. (5.1)."""

    return points + displacement(control, points, shape)


def jacobian(control: torch.Tensor, points: torch.Tensor, shape: Tuple[int, int],
             create_graph: bool = True) -> torch.Tensor:
    """:math:`J[n, i, j] = \\partial u_i / \\partial r_j` at ``points``, shape ``(N, 2, 2)``."""

    pts = points.detach().clone().requires_grad_(True)
    u = displacement(control, pts, shape)
    rows = []
    for i in range(2):
        g, = torch.autograd.grad(u[:, i].sum(), pts, create_graph=create_graph, retain_graph=True)
        rows.append(g)
    return torch.stack(rows, dim=1)


def strain_tensor(J: torch.Tensor) -> torch.Tensor:
    """Eq. (5.3), symmetric part."""

    return 0.5 * (J + J.transpose(-1, -2))


def rotation_tensor(J: torch.Tensor) -> torch.Tensor:
    """Eq. (5.3), antisymmetric part."""

    return 0.5 * (J - J.transpose(-1, -2))


def elastic_energy_density(J: torch.Tensor, lame_lambda: float = 1.0, lame_mu: float = 1.0) -> torch.Tensor:
    """Eq. (5.4), one value per point."""

    eps = strain_tensor(J)
    trace = eps[:, 0, 0] + eps[:, 1, 1]
    return 0.5 * lame_lambda * trace ** 2 + lame_mu * (eps ** 2).sum(dim=(-2, -1))


def resample_control(control: torch.Tensor, new_shape: Sequence[int]) -> torch.Tensor:
    """Control tensor ``(1, 2, Hc, Wc)`` on another node grid; the field itself is carried along."""

    control = torch.as_tensor(control)
    if control.dim() != 4 or control.shape[1] != 2 or control.shape[0] != 1:
        raise ValueError(f"control grid must have shape (1, 2, Hc, Wc), got {tuple(control.shape)}")
    Hc, Wc = int(new_shape[0]), int(new_shape[1])
    if min(Hc, Wc) < 2:
        raise ValueError("the control grid needs at least 2x2 nodes")
    rows = torch.linspace(0.0, 1.0, Hc, dtype=control.dtype, device=control.device)
    cols = torch.linspace(0.0, 1.0, Wc, dtype=control.dtype, device=control.device)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    with torch.no_grad():
        values = sample_fraction(control.detach(), yy.reshape(-1), xx.reshape(-1))      # (Hc*Wc, 2)
    return values.T.reshape(1, 2, Hc, Wc).contiguous()


def resample_deviation(old: torch.Tensor, new: torch.Tensor, shape: Tuple[int, int], probe: int = 96) -> float:
    """Largest change of the field [px] on a ``probe`` x ``probe`` raster caused by a resampling."""

    rows = torch.linspace(0, shape[0] - 1, probe, dtype=old.dtype)
    cols = torch.linspace(0, shape[1] - 1, probe, dtype=old.dtype)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    pts = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
    with torch.no_grad():
        return float((displacement(new, pts, shape) - displacement(old, pts, shape)).abs().max())


def maps(control: torch.Tensor, shape: Tuple[int, int], lame_lambda: Optional[float] = None,
         lame_mu: Optional[float] = None, resolution: Optional[Tuple[int, int]] = None) -> Dict[str, "numpy.ndarray"]:
    """Displacement and strain maps over the image area as numpy arrays.

    Keys: ``u_row, u_col, eps_rr, eps_cc, eps_rc, rot, div`` and, if the Lamé
    parameters are given, ``energy``.  With ``resolution`` the field is sampled on
    a coarser raster spread over the whole area; otherwise at every pixel (the
    pixel with index k sits at coordinate k).
    """

    H, W = shape
    h, w = resolution or (H, W)
    rows = torch.linspace(0, H - 1, h, dtype=control.dtype, device=control.device)
    cols = torch.linspace(0, W - 1, w, dtype=control.dtype, device=control.device)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    pts = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
    u = displacement(control, pts, shape)
    J = jacobian(control, pts, shape, create_graph=False)
    eps = strain_tensor(J)
    omega = rotation_tensor(J)

    def grab(t):
        return t.detach().reshape(h, w).cpu().numpy()

    out = {"u_row": grab(u[:, 0]), "u_col": grab(u[:, 1]),
           "eps_rr": grab(eps[:, 0, 0]), "eps_cc": grab(eps[:, 1, 1]), "eps_rc": grab(eps[:, 0, 1]),
           "rot": grab(omega[:, 1, 0]), "div": grab(eps[:, 0, 0] + eps[:, 1, 1])}
    if lame_lambda is not None and lame_mu is not None:
        out["energy"] = grab(elastic_energy_density(J, lame_lambda, lame_mu))
    return out


def from_legacy_control(control: torch.Tensor, shape: Tuple[int, int], padding: int,
                        new_shape: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Convert a control tensor of the old convention (see module docstring) to the present one.

    The old field was ``u_old(p) = sample(control, x = 2 p0 / (W_pad - 1) - 1, y = 2 p1 / (H_pad - 1) - 1)``
    for a point ``p = (p0, p1)`` in the unpadded frame, with the control axes
    ``(Hc, Wc)`` indexed by ``(y, x)`` and the output channels ``(u_x, u_y)``
    added to ``(p0, p1)``.  The result has the node grid ``new_shape`` (default: the
    old one) and is exact at its nodes.
    """

    H, W = shape
    Hp, Wp = H + 2 * padding, W + 2 * padding
    control = torch.as_tensor(control)
    new_shape = tuple(new_shape) if new_shape is not None else tuple(control.shape[-2:])
    Hc, Wc = new_shape
    rows = torch.linspace(0, H - 1, Hc, dtype=control.dtype)
    cols = torch.linspace(0, W - 1, Wc, dtype=control.dtype)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    grid = torch.stack([2.0 * yy / max(Wp - 1, 1) - 1.0, 2.0 * xx / max(Hp - 1, 1) - 1.0], dim=-1).unsqueeze(0)
    with torch.no_grad():
        return F.grid_sample(control, grid, mode="bicubic", padding_mode="border", align_corners=True).contiguous()
