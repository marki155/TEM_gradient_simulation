r"""Part I, Section 8: image formation (TEM and STEM-HAADF).

Theory
======

Conventional TEM (coherent)
---------------------------
The exit wave :math:`\psi_{\mathrm{exit}}` of Section 6 is transferred by the lens,
Eq. (7.1), on the *padded* grid (the padding absorbs wrap-around of the Fourier
method) and cropped to the field of view afterwards,

.. math::

    \psi_{\mathrm{img}} = \mathcal F^{-1}\big[T\,A\,\mathcal F\,\psi_{\mathrm{exit}}\big],
    \qquad
    I_{\mathrm{TEM}} = \begin{cases}|\psi_{\mathrm{img}}| & \text{signal = amplitude}\\
    |\psi_{\mathrm{img}}|^2 & \text{signal = intensity.}\end{cases}
    \qquad (8.1)

A detector records the *intensity* :math:`|\psi|^2`.  All earlier versions of this
project compared the *amplitude* :math:`|\psi|` with the measured image; this is
kept as the default for old files (``tem_signal = "amplitude"``) because the
fitted contrast parameters (Stobbs factors, normalisation) were calibrated with
it, and the physically correct intensity is available as an option.

STEM-HAADF (incoherent)
-----------------------
A focused probe :math:`P(\mathbf r - \mathbf r_0)` is scanned over the specimen.  The high-angle annular
dark-field detector collects electrons scattered between the angles
:math:`\theta_{\mathrm{in}}` and :math:`\theta_{\mathrm{out}}`.  This signal is
dominated by thermal diffuse (incoherent) scattering from the individual atoms,
so the image is an *incoherent* convolution of the probe intensity with an
object function :math:`O`,

.. math::

    I_{\mathrm{HAADF}}(\mathbf r_0) = \big(|P|^2 * O\big)(\mathbf r_0),
    \qquad
    O(\mathbf r) = \sum_{e} \sigma_e\,\rho_e(\mathbf r).
    \qquad (8.2)

*Probe.*  The probe in Fourier space is the aperture times the aberration phase,
:math:`\tilde P(\mathbf k) = A_{\alpha_c}(\mathbf k)\,\mathrm e^{-\mathrm i\chi(\mathbf k)}`
with the convergence semi-angle :math:`\alpha_c`; its intensity is blurred by
the finite size of the electron source (Gaussian of standard deviation
:math:`s`, so :math:`\exp(-2\pi^2 s^2 k^2)` in Fourier space) and normalised to
unit area,

.. math::

    |P|^2 = \Big|\mathcal F^{-1}\tilde P\Big|^2 * G_s,\qquad \sum |P|^2 = 1.
    \qquad (8.3)

The aberration phase :math:`\chi` is the same function as in TEM, Eq. (7.2), but
with its own (corrected-probe) coefficients.

*Object.*  :math:`\rho_e` is the atom density of element :math:`e` and
:math:`\sigma_e` the cross-section for scattering into the detector ring.  It is
*not* the projected potential: the integral of the projected potential decreases
from B over C and N to O while the HAADF signal grows roughly as
:math:`Z^{1.7}`.  Using the elastic scattering factor :math:`f_e(q)`
(Section 3) and the Einstein model for thermal diffuse scattering with the
one-dimensional r.m.s. displacement :math:`u`, the differential cross-section is
:math:`|f_e|^2\,[1-\exp(-4\pi^2u^2q^2)]`, hence

.. math::

    \sigma_e = 2\pi\lambda^2\int_{q_{\mathrm{in}}}^{q_{\mathrm{out}}}
    |f_e(q)|^2\,\big[1-\mathrm e^{-4\pi^2u^2q^2}\big]\,q\,\mathrm dq,
    \qquad q = \frac{2\sin(\theta/2)}{\lambda}\approx\frac{\theta}{\lambda}.
    \qquad (8.4)

For :math:`u=0` the thermal term vanishes; then the elastic cross-section into the
ring is used so that the model does not collapse.  (:math:`q` in 1/Å, :math:`f_e` in Å, :math:`\lambda` in Å,
:math:`\sigma` in Å².)
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from . import electron
from .optics import FourierGrid, aperture, phase_error
from .scattering import scattering_amplitude_A

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# TEM
# ---------------------------------------------------------------------------

def crop(image: torch.Tensor, shape, padding: int) -> torch.Tensor:
    return image[..., padding:shape[0] + padding, padding:shape[1] + padding]


def tem_image(wave: torch.Tensor, transfer: torch.Tensor, objective_aperture: torch.Tensor,
              shape, padding: int, signal: str = "amplitude") -> torch.Tensor:
    """Eq. (8.1); ``wave`` is on the padded grid, the result on ``shape``."""

    spectrum = torch.fft.fft2(wave) * transfer * objective_aperture
    image = torch.fft.ifft2(spectrum)
    if signal == "amplitude":
        out = image.abs()
    elif signal == "intensity":
        out = image.real ** 2 + image.imag ** 2
    else:
        raise ValueError(f"unknown TEM signal {signal!r}")
    return crop(out, shape, padding)


# ---------------------------------------------------------------------------
# STEM
# ---------------------------------------------------------------------------

def haadf_cross_section_A2(element: str, inner_mrad: float, outer_mrad: float, voltage_kV: float,
                           u_rms_A: float = 0.0, parametrization: str = "kirkland",
                           samples: int = 4000) -> float:
    """Eq. (8.4) in square Angstrom."""

    lam_A = electron.wavelength_nm(voltage_kV) * 10.0
    q = np.linspace(inner_mrad * 1e-3 / lam_A, outer_mrad * 1e-3 / lam_A, samples)
    f = scattering_amplitude_A(q, element, parametrization)
    if u_rms_A > 0:
        weight = 1.0 - np.exp(-4 * math.pi ** 2 * u_rms_A ** 2 * q ** 2)
    else:
        weight = np.ones_like(q)
    return float(2 * math.pi * lam_A ** 2 * _trapezoid(f ** 2 * weight * q, q))


class CrossSections:
    """Cache of Eq. (8.4) per (element, thermal displacement)."""

    def __init__(self, inner_mrad: float, outer_mrad: float, voltage_kV: float,
                 parametrization: str = "kirkland"):
        self.args = (inner_mrad, outer_mrad, voltage_kV)
        self.parametrization = parametrization
        self._values: Dict[Tuple[str, float], float] = {}

    def __call__(self, element: str, u_rms_A: float) -> float:
        key = (element, round(float(u_rms_A), 4))
        if key not in self._values:
            self._values[key] = haadf_cross_section_A2(element, *self.args, u_rms_A=key[1],
                                                       parametrization=self.parametrization)
        return self._values[key]


def probe_intensity(chi: torch.Tensor, probe_aperture: torch.Tensor, source_size_A: float,
                    pixel_size_A: float) -> torch.Tensor:
    """Eq. (8.3), FFT order (probe centre at pixel [0, 0])."""

    probe = torch.fft.ifft2(probe_aperture * torch.exp(-1j * chi))
    intensity = probe.real ** 2 + probe.imag ** 2
    if source_size_A and source_size_A > 0:
        fr = torch.fft.fftfreq(intensity.shape[0], d=pixel_size_A, device=intensity.device)
        fc = torch.fft.rfftfreq(intensity.shape[1], d=pixel_size_A, device=intensity.device)
        k2 = fr[:, None] ** 2 + fc[None, :] ** 2
        blur = torch.exp(-2 * math.pi ** 2 * source_size_A ** 2 * k2)
        intensity = torch.fft.irfft2(torch.fft.rfft2(intensity) * blur, s=intensity.shape)
    return intensity / intensity.sum()


def object_map(densities: Sequence[Optional[Mapping[str, torch.Tensor]]], u_rms_A: Sequence[float],
               cross_sections: CrossSections, shape_padded, device="cpu") -> torch.Tensor:
    """Eq. (8.2), :math:`O = \\sum_e \\sigma_e \\rho_e` over all layers (``None`` layers are skipped)."""

    total = None
    for layer_densities, u in zip(densities, u_rms_A):
        if layer_densities is None:
            continue
        for element, density in layer_densities.items():
            part = density * cross_sections(element, u)
            total = part if total is None else total + part
    if total is None:
        return torch.zeros(tuple(shape_padded), dtype=torch.float32, device=device)
    return total


def stem_image(obj: torch.Tensor, probe: torch.Tensor, shape, padding: int) -> torch.Tensor:
    """Eq. (8.2): circular convolution on the padded grid, cropped."""

    image = torch.fft.irfft2(torch.fft.rfft2(obj) * torch.fft.rfft2(probe), s=obj.shape[-2:])
    return crop(image, shape, padding)
