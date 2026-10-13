r"""Part I, Section 7: the objective lens (wave optics).

Theory
======

The exit wave :math:`\psi_{\mathrm{exit}}(\mathbf r)` leaves the specimen and
passes the objective lens.  In the weak-lens, linear-in-the-aberrations picture
used throughout, the lens acts in Fourier space (spatial frequency
:math:`\mathbf k`, with the scattering angle :math:`\alpha = \lambda |\mathbf k|`)
by a multiplication with a **transfer function** :math:`T(\mathbf k)`,

.. math::

    \tilde\psi_{\mathrm{img}}(\mathbf k) = T(\mathbf k)\,A(\mathbf k)\,
    \tilde\psi_{\mathrm{exit}}(\mathbf k).
    \qquad (7.1)

Wave aberration
---------------
Departures of the lens from the ideal are collected in the aberration function,
a series in the angle :math:`\alpha` and the azimuth :math:`\varphi`
(Krivanek notation, coefficients :math:`C_{nm}` of order :math:`n` and
multiplicity :math:`m`),

.. math::

    \chi(\alpha, \varphi) = \frac{2\pi}{\lambda}
    \sum_{n,m} \frac{C_{nm}}{n+1}\,\alpha^{\,n+1}\cos\!\big(m(\varphi - \varphi_{nm})\big).
    \qquad (7.2)

The azimuth is measured from the *row* axis towards the *column* axis,
:math:`\varphi = \operatorname{atan2}(k_{\mathrm{col}}, k_{\mathrm{row}})`.  The
orders in use are listed in :data:`ABERRATION_TERMS`: defocus :math:`C_{10}`,
two-fold astigmatism :math:`C_{12}`, coma :math:`C_{21}`, three-fold
astigmatism :math:`C_{23}`, spherical aberration :math:`C_{30}`, and higher
terms.  Units: :math:`C_{10}, C_{12}, C_{21}, C_{23}, C_{32}, \dots` in nm;
:math:`C_{30}, C_{34}, C_{45}` in micrometres (they are of the order of
millimetres).

*Cartesian form.*  With :math:`a_{nm} = C_{nm}\cos(m\varphi_{nm})` and
:math:`b_{nm} = C_{nm}\sin(m\varphi_{nm})`,

.. math::

    C_{nm}\cos\!\big(m(\varphi-\varphi_{nm})\big) = a_{nm}\cos m\varphi + b_{nm}\sin m\varphi.
    \qquad (7.3)

Both forms are the same function.  The Cartesian one is preferable for
optimisation: the gradient with respect to the angle :math:`\varphi_{nm}` scales
with :math:`C_{nm}` and vanishes for small coefficients, whereas
:math:`(a, b)` are well-conditioned everywhere.  :func:`polar_to_cartesian` and
:func:`cartesian_to_polar` convert.

Transfer function
-----------------
.. math::

    T(\mathbf k) = \big(\sqrt{1-Q^2} - \mathrm i Q\big)\,\mathrm e^{-\mathrm i\chi(\mathbf k)}\,
    E(\mathbf k),
    \qquad (7.4)

where :math:`Q\in[0,1]` is the *amplitude contrast* (the part of the contrast not
caused by the phase of the specimen, e.g. from inelastic losses) and :math:`E`
the product of the coherence envelopes below.

Aperture
--------
The objective aperture :math:`A(\mathbf k) = 1` for :math:`\alpha \le
\alpha_{\max}` and 0 otherwise cuts off high angles.

Envelopes
---------
Partial coherence damps high spatial frequencies.  Each source gives a Gaussian
factor:

* *Chromatic* (energy spread :math:`\Delta E` of the electrons and chromatic
  aberration :math:`C_c`) gives a spread of defocus
  :math:`\Delta f = C_c\,\Delta E / E_0` (:math:`E_0 = eU`):

  .. math::
      E_c(k) = \exp\!\big[-(\pi\lambda\,\Delta f\,k^2)^2\big].
      \qquad (7.5)

  Here :math:`\Delta E` is a parameter of this model (half the :math:`1/e`
  width of the focal-spread distribution, see :class:`Envelope`); it is fitted
  together with :math:`C_c` and need not coincide with the nominal source value.

* *Spatial* (finite source or illumination semi-angle :math:`\beta` in rad)
  damps through the gradient of the aberration function,

  .. math::
      E_s(k) = \exp\!\Big[-\Big(\frac{\pi\beta}{\lambda}\Big)^2
               \big(C_s\lambda^3k^3 + \Delta z\,\lambda k\big)^2\Big]
      \qquad (7.6)

  with defocus :math:`\Delta z` and spherical aberration :math:`C_s` (both in nm
  and taken from Eq. 7.2; higher aberrations are neglected in the envelope).

* *Instability* (vibration, drift: defocus jitter :math:`\sigma_z`):
  :math:`E_i(k) = \exp[-(\pi\lambda\sigma_z k^2)^2]`.

* *Detector* (optional MTF): :math:`E_d(k) = \exp[-k^2/(2\sigma_k^2)]`.

The envelope is the product of the enabled factors, clipped to :math:`[0, 1]`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Aberration table
# ---------------------------------------------------------------------------

#: key -> (power of alpha, prefactor 1/(n+1), unit scale to nm, multiplicity m).
#: ``m = 0`` is rotationally symmetric.
ABERRATION_TERMS: Dict[str, Tuple[int, float, float, int]] = {
    "C10": (2, 0.5, 1.0, 0),
    "C12": (2, 0.5, 1.0, 2),
    "C21": (3, 1 / 3, 1.0, 1),
    "C23": (3, 1 / 3, 1.0, 3),
    "C30": (4, 0.25, 1e3, 0),
    "C32": (4, 0.25, 1.0, 2),
    "C34": (4, 0.25, 1e3, 4),
    "C41": (5, 0.2, 1.0, 1),
    "C43": (5, 0.2, 1.0, 3),
    "C45": (5, 0.2, 1e3, 5),
    "C50": (6, 1 / 6, 1.0, 0),
    "C52": (6, 1 / 6, 1.0, 2),
    "C54": (6, 1 / 6, 1.0, 4),
    "C56": (6, 1 / 6, 1.0, 6),
}

#: Typical TEM aberrations (nm; C30, C34, C45 in um), a start for the fit.
DEFAULT_ABERRATIONS = {
    "C10": -13.07, "C12": 0.35, "phi12": 0.0, "C23": 0.17, "phi23": 0.0,
    "C21": 3.71, "phi21": 0.0, "C30": 4.16, "C34": -0.2, "phi34": 0.0,
    "C45": 0.25, "phi45": 0.0,
}

#: Residual aberrations of a corrected STEM probe (start values, see Part II).
DEFAULT_PROBE_ABERRATIONS = {
    "C10": 1.0, "C12": 1.0, "phi12": 0.0, "C21": 5.0, "phi21": 0.0,
    "C23": 3.0, "phi23": 0.0, "C30": 0.5,
}


def angle_key(magnitude: str) -> str:
    """``C12`` -> ``phi12``."""

    return "phi" + magnitude[1:]


def cartesian_keys(magnitude: str) -> Tuple[str, str]:
    """``C12`` -> ``('a12', 'b12')``."""

    return "a" + magnitude[1:], "b" + magnitude[1:]


def is_angle_key(key: str) -> bool:
    return key.startswith("phi")


def magnitude_key(key: str) -> str:
    """Map ``phi12``, ``a12``, ``b12`` to ``C12``; other keys unchanged."""

    if is_angle_key(key):
        return "C" + key[3:]
    if key[0] in ("a", "b") and key[1:].isdigit():
        return "C" + key[1:]
    return key


def polar_to_cartesian(coeffs: Mapping[str, float]) -> Dict[str, float]:
    """``{C12, phi12}`` -> ``{a12, b12}`` (Eq. 7.3); symmetric terms are kept."""

    out: Dict[str, float] = {}
    for key, value in coeffs.items():
        if is_angle_key(key):
            continue
        term = ABERRATION_TERMS.get(key)
        if term is None or term[3] == 0:
            out[key] = value
            continue
        m = term[3]
        phi = float(coeffs.get(angle_key(key), 0.0))
        a_key, b_key = cartesian_keys(key)
        out[a_key] = float(value) * math.cos(m * phi)
        out[b_key] = float(value) * math.sin(m * phi)
    return out


def cartesian_to_polar(coeffs: Mapping[str, float]) -> Dict[str, float]:
    """``{a12, b12}`` -> ``{C12, phi12}``."""

    out: Dict[str, float] = {k: v for k, v in coeffs.items()
                             if not (k[0] in ("a", "b") and k[1:].isdigit())}
    for magnitude, (_p, _f, _s, m) in ABERRATION_TERMS.items():
        if m == 0:
            continue
        a_key, b_key = cartesian_keys(magnitude)
        if a_key in coeffs and b_key in coeffs:
            a, b = float(coeffs[a_key]), float(coeffs[b_key])
            out[magnitude] = math.hypot(a, b)
            out[angle_key(magnitude)] = math.atan2(b, a) / m
    return out


def convert(coeffs: Mapping[str, float], parametrization: str) -> Dict[str, float]:
    if parametrization == "cartesian":
        return polar_to_cartesian(coeffs)
    if parametrization == "polar":
        return cartesian_to_polar(coeffs)
    raise ValueError(f"unknown parametrization {parametrization!r}")


# ---------------------------------------------------------------------------
# Fourier grid
# ---------------------------------------------------------------------------

class FourierGrid:
    """Spatial frequencies of an ``(H, W)`` raster in FFT order, with derived maps.

    ``k_row``, ``k_col`` in 1/nm; ``alpha = lambda k`` in rad; ``phi`` per
    convention of Eq. (7.2).  Maps that do not depend on the coefficients
    (powers of alpha, cos/sin of the azimuth) are computed once.
    """

    def __init__(self, shape, pixel_size_nm: float, wavelength_nm: float, device="cpu",
                 k_row: Optional[torch.Tensor] = None, k_col: Optional[torch.Tensor] = None):
        self.shape = (int(shape[0]), int(shape[1]))
        self.pixel_size_nm = float(pixel_size_nm)
        self.wavelength_nm = float(wavelength_nm)
        self.device = device
        if k_row is None or k_col is None:
            fr = torch.fft.fftfreq(self.shape[0], d=pixel_size_nm, device=device)
            fc = torch.fft.fftfreq(self.shape[1], d=pixel_size_nm, device=device)
            k_row, k_col = torch.meshgrid(fr, fc, indexing="ij")
        self.k_row, self.k_col = k_row.to(device), k_col.to(device)
        self.k2 = self.k_row ** 2 + self.k_col ** 2
        self.k = torch.sqrt(self.k2 + 1e-30)
        self.alpha = wavelength_nm * self.k
        self.phi = torch.atan2(self.k_col, self.k_row)
        self._basis: Dict[str, torch.Tensor] = {}
        self._powers: Dict[int, torch.Tensor] = {}
        self.cache: Dict[str, torch.Tensor] = {}

    def _power(self, n: int) -> torch.Tensor:
        if n not in self._powers:
            self._powers[n] = self.alpha ** n
        return self._powers[n]

    def basis(self, key: str) -> torch.Tensor:
        """Coefficient-independent map of one term; ``key`` is ``C10``-like or ``a12``/``b12``."""

        if key not in self._basis:
            if key in ABERRATION_TERMS and ABERRATION_TERMS[key][3] == 0:
                power, pref, scale, _m = ABERRATION_TERMS[key]
                self._basis[key] = pref * scale * self._power(power)
            else:
                magnitude = magnitude_key(key)
                power, pref, scale, m = ABERRATION_TERMS[magnitude]
                trig = torch.cos(m * self.phi) if key[0] == "a" else torch.sin(m * self.phi)
                self._basis[key] = pref * scale * self._power(power) * trig
        return self._basis[key]


def frequency_grid(shape, pixel_size_nm: float, wavelength_nm: float, device="cpu") -> FourierGrid:
    return FourierGrid(shape, pixel_size_nm, wavelength_nm, device)


# ---------------------------------------------------------------------------
# aberration function, transfer function
# ---------------------------------------------------------------------------

def phase_error(grid: FourierGrid, coeffs: Mapping[str, object]) -> torch.Tensor:
    """:math:`\\chi(\\mathbf k)` of Eq. (7.2) in rad; coefficients are floats or tensors.

    Both the polar (``C12``, ``phi12``) and the Cartesian (``a12``, ``b12``)
    spelling are understood.  Terms without coefficients are skipped.
    """

    chi = torch.zeros_like(grid.alpha, dtype=torch.float32)
    for magnitude, (_power, _pref, _scale, m) in ABERRATION_TERMS.items():
        if m == 0:
            if magnitude in coeffs:
                chi = chi + grid.basis(magnitude) * coeffs[magnitude]
            continue
        a_key, b_key = cartesian_keys(magnitude)
        if a_key in coeffs and b_key in coeffs:
            a, b = coeffs[a_key], coeffs[b_key]
        elif magnitude in coeffs and angle_key(magnitude) in coeffs:
            amplitude = coeffs[magnitude]
            argument = m * torch.as_tensor(coeffs[angle_key(magnitude)], dtype=torch.float32, device=grid.device)
            amplitude = torch.as_tensor(amplitude, dtype=torch.float32, device=grid.device)
            a, b = amplitude * torch.cos(argument), amplitude * torch.sin(argument)
        else:
            continue
        chi = chi + grid.basis(a_key) * a + grid.basis(b_key) * b
    return 2.0 * math.pi * chi / grid.wavelength_nm


@dataclass
class Envelope:
    """Switches and fixed parameters of the envelopes (Eqs. 7.5/7.6).

    ``delta_E_eV`` and ``Cc_um`` are *not* stored here: they are fitted and passed
    to :func:`envelope` as (possibly differentiable) numbers.
    """

    #: illumination semi-angle beta [rad]; 0 switches the spatial envelope off
    source_size: float = 0.0
    use_spatial: bool = True
    use_chromatic: bool = True
    use_instability: bool = True
    instab_defocus_nm: float = 0.0
    use_detector: bool = False
    detector_sigma_k: Optional[float] = None

    @classmethod
    def from_dict(cls, params: Optional[Mapping[str, object]]) -> "Envelope":
        params = params or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in params.items() if k in known})


def envelope(grid: FourierGrid, settings: Envelope, voltage_kV: float, delta_E_eV, Cc_um,
             defocus_nm=None, cs_nm=None) -> torch.Tensor:
    """Product of the enabled envelopes, in [0, 1]."""

    lam, k, k2 = grid.wavelength_nm, grid.k, grid.k2
    energy_eV = voltage_kV * 1e3
    out = torch.ones_like(k)

    if settings.use_spatial and settings.source_size > 0:
        if "k3" not in grid.cache:
            grid.cache["k3"] = k ** 3
        gradient = torch.zeros_like(k)
        if defocus_nm is not None:
            gradient = gradient + defocus_nm * lam * k
        if cs_nm is not None:
            gradient = gradient + cs_nm * lam ** 3 * grid.cache["k3"]
        out = out * torch.exp(-((math.pi * settings.source_size / lam) ** 2) * gradient ** 2)

    if settings.use_chromatic:
        focal_spread_nm = Cc_um * 1e3 * delta_E_eV / energy_eV
        out = out * torch.exp(-(math.pi * lam * focal_spread_nm * k2) ** 2)

    if settings.use_instability and settings.instab_defocus_nm > 0:
        if "instability" not in grid.cache:
            grid.cache["instability"] = torch.exp(-(math.pi * lam * settings.instab_defocus_nm * k2) ** 2)
        out = out * grid.cache["instability"]

    if settings.use_detector and settings.detector_sigma_k is not None:
        out = out * torch.exp(-k2 / (2.0 * float(settings.detector_sigma_k) ** 2))

    return torch.clamp(out, 0.0, 1.0)


def transfer_function(grid: FourierGrid, coeffs: Mapping[str, object], amp_contrast,
                      env: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Eq. (7.4), complex64, FFT order.  ``amp_contrast`` is clipped to [0, 1]."""

    q = torch.clamp(torch.as_tensor(amp_contrast, dtype=torch.float32, device=grid.device), 0.0, 1.0)
    chi = phase_error(grid, coeffs)
    t = (torch.sqrt(1.0 - q ** 2) - 1j * q) * torch.exp(-1j * chi)
    if env is not None:
        t = t * env.to(t.dtype)
    return t


def aperture(grid: FourierGrid, alpha_max_mrad: float) -> torch.Tensor:
    return (grid.alpha <= alpha_max_mrad * 1e-3).to(torch.float32)


def scherzer_defocus_nm(cs_um: float, wavelength_nm: float) -> float:
    r""":math:`\Delta z_{\mathrm{Sch}} = -1.2\sqrt{C_s\lambda}` (nm)."""

    return -1.2 * math.sqrt(cs_um * 1e3 * wavelength_nm)
