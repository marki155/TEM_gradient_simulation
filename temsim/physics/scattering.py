"""Part I, Section 3: projected atomic potentials.

Theory
======

From scattering amplitude to potential
--------------------------------------
Tabulated elastic electron scattering amplitudes :math:`f_e(q)` (Angstrom, with
:math:`q = |\\mathbf q|` in 1/Angstrom, :math:`q = 2\\sin\\theta/\\lambda`) are
related to the atomic potential by Eq. (1.1),
:math:`\\tilde V(q) = (h^2 / 2\\pi m_0 e) f_e(q)`.  The multislice method needs the
potential *projected* along the beam, :math:`V_{\\mathrm{proj}}(r) = \\int V\\,dz`.
By the projection-slice theorem its two-dimensional Fourier transform is the central
section :math:`\\tilde V(q_x, q_y, 0)`, so

.. math::

    V_{\\mathrm{proj}}(r) = \\frac{h^2}{2\\pi m_0 e}
        \\int_0^\\infty f_e(q)\\, J_0(2\\pi q r)\\, 2\\pi q\\, dq
    \\qquad (3.1)

(:data:`~temsim.physics.constants.POTENTIAL_PREFACTOR_V_A2` = :math:`h^2/2\\pi m_0 e`).
The tests evaluate this integral numerically and compare with the closed forms below.

Parametrisations
----------------
Two analytic fits of :math:`f_e` are available, selected by name.

**Kirkland** (E. J. Kirkland, *Advanced Computing in Electron Microscopy*, 2nd ed.,
2010) fits three Lorentzians and three Gaussians,

.. math::

    f_e(q) = \\sum_{i=1}^{3} \\frac{a_i}{q^2 + b_i}
           + \\sum_{i=1}^{3} c_i\\, e^{-d_i q^2}.

Inserting this into Eq. (3.1) and using
:math:`\\int_0^\\infty J_0(2\\pi q r)\\, 2\\pi q\\, dq/(q^2+b) = 2\\pi K_0(2\\pi\\sqrt b\\, r)`
and the Gaussian Hankel transform gives

.. math::

    V_{\\mathrm{proj}}(r) = \\sum_{i=1}^{3}\\Big[\\, 2\\,\\alpha_i\\, K_0(\\beta_i r)
        + \\sqrt{\\pi/\\delta_i}\\;\\gamma_i\\, e^{-\\delta_i r^2}\\Big],
    \\qquad (3.2)

with the rescaled parameters
:math:`\\alpha_i = \\pi a_i C`, :math:`\\beta_i = 2\\pi\\sqrt{b_i}`,
:math:`\\gamma_i = \\pi^{3/2} c_i\\, C / d_i^{3/2}`, :math:`\\delta_i = \\pi^2/d_i`,
where :math:`C = h^2/(2\\pi m_0 e)`.

**Lobato & Van Dyck** (Acta Cryst. A 70, 636, 2014) use five terms
:math:`f_e(q) = \\sum_{i=1}^{5} A_i (2 + B_i q^2) / (1 + B_i q^2)^2`, which behave
correctly at large :math:`q` (proper Mott–Bethe asymptotics) and are the better
choice for heavy atoms and high scattering angles.  The corresponding projected
potential is

.. math::

    V_{\\mathrm{proj}}(r) = 2\\sum_{i=1}^{5}\\Big[
        \\frac{2\\alpha_i}{\\beta_i} K_0(\\beta_i r) + \\alpha_i\\, r\\, K_1(\\beta_i r)\\Big],
    \\qquad (3.3)

with :math:`\\alpha_i = \\pi^2 A_i C / B_i^{3/2}` and :math:`\\beta_i = 2\\pi/\\sqrt{B_i}`.

The parameter tables (103 elements each) are those published with the two papers;
they are shipped with the package in ``temsim/physics/data/`` (the files originate
from the abTEM distribution, which compiled them from the original publications).

Numerical kernel on a pixel grid
--------------------------------
Eqs. (3.2) and (3.3) are *integrable* logarithmic singularities at :math:`r = 0`
(:math:`K_0(\\beta r) \\sim -\\ln r`).  Sampling at pixel centres would give a
pixel-size dependent, wrong peak value.  Two measures keep the sampled kernel a
faithful *pixel average*:

1. **Soft core.**  Every evaluation uses :math:`r_{\\mathrm{eff}} = \\sqrt{r^2 + a^2}`
   with :math:`a = \\epsilon\\,\\Delta` (:math:`\\Delta` pixel size, default
   :math:`\\epsilon = 10^{-3}`), which removes the singularity without noticeably
   changing the potential.
2. **Supersampling.**  Pixels whose centre lies within ``supersample_radius_px``
   pixels of the atom are replaced by the mean of an
   :math:`n\\times n` sub-grid (default :math:`n = 8`).

The kernel is returned in V nm with the atom at index ``[0, 0]`` (FFT convention),
so that convolution with a density map is a product in Fourier space (Section 4).

Thermal motion
--------------
The tables describe a *static* atom.  Isotropic thermal displacements with root
mean square amplitude :math:`u` (per direction) smear the potential with a
Gaussian, :math:`V \\to V * G_u`, which is a multiplication in Fourier space,

.. math::

    \\tilde V(k) \\to \\tilde V(k)\\; e^{-2\\pi^2 u^2 k^2}
    \\qquad (3.4)

(equivalent to :math:`e^{-B s^2}` with :math:`B = 8\\pi^2 u^2`, :math:`s = k/2`).
The area integral is conserved; only the peak is lowered.  Eq. (3.4) is baked into
the stored spectrum so that it costs nothing at run time (:class:`KernelBank`).
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .constants import ANGSTROM, POTENTIAL_PREFACTOR_V_A2

PARAMETRIZATIONS = ("kirkland", "lobato")
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

#: default numerical settings of the pixel kernel (see module docstring)
SUPERSAMPLE_RADIUS_PX = 2.0
SUPERSAMPLE_FACTOR = 8
SOFTCORE_FRACTION = 1e-3


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _table(parametrization: str) -> Dict[str, np.ndarray]:
    if parametrization not in PARAMETRIZATIONS:
        raise ValueError(f"unknown parametrisation '{parametrization}', choose from {PARAMETRIZATIONS}")
    with open(os.path.join(_DATA_DIR, parametrization + ".json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {symbol: np.asarray(p, dtype=np.float64) for symbol, p in raw.items()}


def available_elements(parametrization: str = "kirkland") -> List[str]:
    return list(_table(parametrization))


def raw_parameters(element: str, parametrization: str = "kirkland") -> np.ndarray:
    """Published table entry: Kirkland ``(4, 3)`` = rows a, b, c, d; Lobato ``(2, 5)`` = rows A, B."""

    table = _table(parametrization)
    if element not in table:
        raise KeyError(f"element '{element}' is not tabulated for '{parametrization}'")
    return table[element]


def scaled_parameters(element: str, parametrization: str = "kirkland") -> np.ndarray:
    """Rescaled parameters of Eq. (3.2) (rows alpha, beta, gamma, delta) or Eq. (3.3) (alpha, beta)."""

    p = raw_parameters(element, parametrization)
    C = POTENTIAL_PREFACTOR_V_A2
    if parametrization == "kirkland":
        a, b, c, d = p
        return np.vstack([math.pi * a * C,
                          2.0 * math.pi * np.sqrt(b),
                          math.pi ** 1.5 * c * C / d ** 1.5,
                          math.pi ** 2 / d])
    A, B = p
    return np.vstack([math.pi ** 2 * A * C / B ** 1.5, 2.0 * math.pi / np.sqrt(B)])


# ---------------------------------------------------------------------------
# radial functions
# ---------------------------------------------------------------------------

def projected_potential_VA(r_A: torch.Tensor, element: str, parametrization: str = "kirkland") -> torch.Tensor:
    """Eqs. (3.2)/(3.3): projected potential in V Angstrom at radius ``r_A`` (Angstrom)."""

    r = torch.as_tensor(r_A, dtype=torch.float64)
    p = scaled_parameters(element, parametrization)
    out = torch.zeros_like(r)
    if parametrization == "kirkland":
        for alpha, beta, gamma, delta in p.T:
            out = out + 2.0 * alpha * torch.special.modified_bessel_k0(beta * r)
            out = out + math.sqrt(math.pi / delta) * gamma * torch.exp(-delta * r * r)
    else:
        for alpha, beta in p.T:
            out = out + 2.0 * (2.0 * alpha / beta * torch.special.modified_bessel_k0(beta * r)
                               + alpha * r * torch.special.modified_bessel_k1(beta * r))
    return out


def scattering_amplitude_A(q_per_A: np.ndarray, element: str, parametrization: str = "kirkland") -> np.ndarray:
    """:math:`f_e(q)` in Angstrom, ``q`` in 1/Angstrom (used by the STEM cross section and the tests)."""

    q2 = np.asarray(q_per_A, dtype=np.float64) ** 2
    p = raw_parameters(element, parametrization)
    if parametrization == "kirkland":
        a, b, c, d = p
        return (sum(a[i] / (q2 + b[i]) for i in range(3))
                + sum(c[i] * np.exp(-d[i] * q2) for i in range(3)))
    A, B = p
    return sum(A[i] * (2.0 + B[i] * q2) / (1.0 + B[i] * q2) ** 2 for i in range(5))


# ---------------------------------------------------------------------------
# pixel kernel
# ---------------------------------------------------------------------------

def _signed_index(n: int) -> np.ndarray:
    """Pixel index in FFT order -> signed offset from the atom, ``0, 1, ..., -2, -1``."""

    return np.fft.fftfreq(n, d=1.0 / n).round().astype(np.int64)


def real_space_kernel(shape: Tuple[int, int], pixel_size_nm: float, element: str,
                      parametrization: str = "kirkland",
                      supersample_radius_px: float = SUPERSAMPLE_RADIUS_PX,
                      supersample_factor: int = SUPERSAMPLE_FACTOR,
                      softcore_fraction: float = SOFTCORE_FRACTION) -> torch.Tensor:
    """Pixel-averaged projected potential of one atom in V nm, atom at ``[0, 0]`` (float64)."""

    H, W = shape
    pixel_A = float(pixel_size_nm) / ANGSTROM
    soft = max(pixel_A * softcore_fraction, 1e-8)

    my = _signed_index(H)[:, None] * pixel_A
    mx = _signed_index(W)[None, :] * pixel_A
    r_eff = np.sqrt(my ** 2 + mx ** 2 + soft ** 2)
    kernel = projected_potential_VA(torch.from_numpy(r_eff), element, parametrization)

    n = int(supersample_factor)
    if supersample_radius_px and supersample_radius_px > 0 and n > 1:
        reach = int(math.ceil(supersample_radius_px)) + 1
        iy = np.arange(max(-reach, -(H // 2)), min(reach, (H - 1) // 2) + 1)
        ix = np.arange(max(-reach, -(W // 2)), min(reach, (W - 1) // 2) + 1)
        cy, cx = np.meshgrid(iy * pixel_A, ix * pixel_A, indexing="ij")
        inside = np.hypot(cy, cx) <= supersample_radius_px * pixel_A
        offsets = ((np.arange(n) + 0.5) / n - 0.5) * pixel_A
        sy = cy[..., None, None] + offsets[:, None]
        sx = cx[..., None, None] + offsets[None, :]
        r_sub = np.sqrt(sy ** 2 + sx ** 2 + soft ** 2)
        v_sub = projected_potential_VA(torch.from_numpy(r_sub), element, parametrization)
        mean = v_sub.mean(dim=(-1, -2))
        for a, i in enumerate(iy):
            for b, j in enumerate(ix):
                if inside[a, b]:
                    kernel[i % H, j % W] = mean[a, b]
    return kernel * ANGSTROM          # V Angstrom -> V nm


def debye_waller_factor(shape: Tuple[int, int], pixel_size_nm: float, u_nm: float) -> torch.Tensor:
    """Eq. (3.4) on the ``rfft2`` grid, shape ``(H, W//2 + 1)``."""

    H, W = shape
    ky = torch.fft.fftfreq(H, d=pixel_size_nm, dtype=torch.float64)
    kx = torch.fft.rfftfreq(W, d=pixel_size_nm, dtype=torch.float64)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    return torch.exp(-2.0 * math.pi ** 2 * u_nm ** 2 * k2)


class KernelBank:
    """Fourier spectra of the atomic kernels of all (element, thermal amplitude) pairs in use.

    ``spectrum(element, u_nm)`` returns ``rfft2`` of the kernel with Eq. (3.4) applied.
    A potential map is then ``irfft2(rfft2(density) * spectrum)`` (Section 4, Eq. 4.2).
    Spectra are cached; the same element may appear with different ``u`` in different
    layers (the previous implementation required one ``u`` per element).
    """

    def __init__(self, shape: Tuple[int, int], pixel_size_nm: float, parametrization: str = "kirkland",
                 device="cpu", supersample_radius_px: float = SUPERSAMPLE_RADIUS_PX,
                 supersample_factor: int = SUPERSAMPLE_FACTOR,
                 softcore_fraction: float = SOFTCORE_FRACTION):
        self.shape = (int(shape[0]), int(shape[1]))
        self.pixel_size_nm = float(pixel_size_nm)
        self.parametrization = parametrization
        self.device = device
        self._numerics = dict(supersample_radius_px=supersample_radius_px,
                              supersample_factor=supersample_factor,
                              softcore_fraction=softcore_fraction)
        self._real: Dict[str, torch.Tensor] = {}
        self._spectra: Dict[Tuple[str, float], torch.Tensor] = {}

    def kernel(self, element: str) -> torch.Tensor:
        """Real-space kernel (float32, atom at [0, 0])."""

        if element not in self._real:
            k = real_space_kernel(self.shape, self.pixel_size_nm, element, self.parametrization,
                                  **self._numerics)
            self._real[element] = k.to(torch.float32).to(self.device)
        return self._real[element]

    def spectrum(self, element: str, u_nm: float = 0.0) -> torch.Tensor:
        key = (element, round(float(u_nm), 9))
        if key not in self._spectra:
            spec = torch.fft.rfft2(self.kernel(element))
            if u_nm > 0:
                spec = spec * debye_waller_factor(self.shape, self.pixel_size_nm, u_nm).to(
                    dtype=torch.float32, device=self.device)
            self._spectra[key] = spec
        return self._spectra[key]
