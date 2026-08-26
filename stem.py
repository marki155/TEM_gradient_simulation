"""STEM-Abbildung (HAADF) in der inkohaerenten Naeherung.

    I(r) = |P(r)|^2  *  O(r)

``P`` ist die Sonde, gebaut aus derselben Aberrationsfunktion wie die CTF -
Defokus, Astigmatismus & Co. sind also zwischen TEM- und STEM-Zweig geteilt.

``O`` ist **nicht** das projizierte Potential, sondern die Dichte des
Streuquerschnitts in den Detektorring.  Das ist wichtig: das Flaechenintegral
des projizierten Potentials faellt von B ueber C und N nach O, waehrend das
HAADF-Signal mit ungefaehr Z^1.7 steigt.  Wuerde man V als Objekt nehmen,
haette der Z-Kontrast das falsche Vorzeichen.

Der Querschnitt kommt aus derselben Kirkland-Parametrisierung wie das
Potential (abTEM), im Einstein-Modell fuer die thermisch diffuse Streuung:

    dsigma/dOmega = |f_e(q)|^2 * [1 - exp(-4 pi^2 u^2 q^2)]
    sigma        = 2 pi lambda^2 * int_{q_in}^{q_out} |f_e|^2 [...] q dq

mit q = 1/d = 2 sin(theta)/lambda in 1/A und f_e in A.  Damit ergibt sich
Z^1.8 von selbst statt per Dekret (gemessen: 1.80 bei 40-200 mrad, 1.89 bei
100-300 mrad).
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

from abtem.parametrizations import KirklandParametrization

from physics_helper import relativistic_wavelength_nm

_PARAMETRIZATION = KirklandParametrization()
_SCATTERING_PARAMETERS: Dict[str, np.ndarray] = {}


def scattering_parameters(element: str) -> np.ndarray:
    """Kirkland-Parameter (a, b, c, d) des Elektronen-Streufaktors."""

    if element not in _SCATTERING_PARAMETERS:
        _SCATTERING_PARAMETERS[element] = np.asarray(
            _PARAMETRIZATION.scaled_parameters(element, "scattering_factor"), dtype=np.float64)
    return _SCATTERING_PARAMETERS[element]


def electron_scattering_factor(element: str, q):
    """f_e(q) in Angstroem, q = 1/d = 2 sin(theta)/lambda in 1/A.

    Geprueft: f_e(0) trifft die Literaturwerte (C 2.511 A, W 12.482 A) und der
    Grenzfall grosser q die Mott-Bethe-Beziehung 0.023934*Z/s^2 auf 1-3 %.
    """

    a, b, c, d = scattering_parameters(element)
    q = np.asarray(q, dtype=np.float64)
    return sum(a[i] / (q ** 2 + b[i]) + c[i] * np.exp(-d[i] * q ** 2) for i in range(3))


def haadf_cross_section(element: str, inner_mrad: float, outer_mrad: float,
                        voltage_kV: float, u_rms_A: float = 0.0, samples: int = 4000) -> float:
    """Streuquerschnitt eines Atoms in den Detektorring, in Angstroem^2.

    ``u_rms_A`` ist die eindimensionale thermische Auslenkung. Ohne sie (0)
    waere die thermisch diffuse Streuung null - dann wird der rein elastische
    Querschnitt in den Ring genommen, damit das Modell nicht zusammenbricht.
    """

    lam_A = relativistic_wavelength_nm(voltage_kV) * 10.0
    q = np.linspace(inner_mrad * 1e-3 / lam_A, outer_mrad * 1e-3 / lam_A, samples)
    f = electron_scattering_factor(element, q)

    if u_rms_A > 0:
        weight = 1.0 - np.exp(-4 * math.pi ** 2 * u_rms_A ** 2 * q ** 2)
    else:
        weight = np.ones_like(q)

    return float(2 * math.pi * lam_A ** 2 * np.trapz(f ** 2 * weight * q, q))


def probe_intensity_from_aberrations(chi, aperture, source_size_A=0.0, pixel_size_A=1.0):
    """|P(r)|^2 aus der Aberrationsphase, flaechennormiert auf 1.

    ``source_size_A`` ist die Gauss-Breite (Sigma) der endlichen Quelle und
    verschmiert die Sondenintensitaet - der uebliche Weg, partielle raeumliche
    Kohaerenz im STEM zu beruecksichtigen.
    """

    wave = aperture * torch.exp(-1j * chi)
    probe = torch.fft.ifft2(torch.fft.ifftshift(wave))
    intensity = (probe.real ** 2 + probe.imag ** 2)

    if source_size_A and source_size_A > 0:
        H, W = intensity.shape
        fy = torch.fft.fftfreq(H, d=pixel_size_A, device=intensity.device)
        fx = torch.fft.rfftfreq(W, d=pixel_size_A, device=intensity.device)
        k2 = fy[:, None] ** 2 + fx[None, :] ** 2
        blur = torch.exp(-2 * math.pi ** 2 * source_size_A ** 2 * k2)
        intensity = torch.fft.irfft2(torch.fft.rfft2(intensity) * blur, s=intensity.shape)

    return intensity / intensity.sum()


class DifferentiableHAADF(nn.Module):
    """HAADF-Bild aus den Elementdichten der Schichten.

    Erwartet als Eingabe ``(dichten, aux_loss)`` - genau das, was
    ``DifferentiablePotential`` im Modus ``element_density`` liefert.
    """

    def __init__(self, config, train_method, probe_layer, device=None):
        super().__init__()

        self.config = config
        self.train_method = train_method
        self.probe_layer = probe_layer
        self.device = device

        self.H, self.W = config.shape
        self.pad = config.padding
        self.pixel_size_A = config.pixel_size_nm * 10.0
        self.voltage_kV = config.voltage_kV

        self.inner_mrad, self.outer_mrad = config.stem_detector_mrad
        self.convergence_mrad = config.stem_convergence_mrad
        self.source_size_A = config.stem_source_size_A

        # Querschnitte je (Element, thermische Auslenkung) - konstante Zahlen
        self.cross_sections: Dict[str, float] = {}
        for layer in config.layers:
            for element in (layer.elements or [layer.element]):
                key = self._key(element, layer.debye_waller_u)
                if key not in self.cross_sections:
                    self.cross_sections[key] = haadf_cross_section(
                        element, self.inner_mrad, self.outer_mrad,
                        self.voltage_kV, layer.debye_waller_u)

        # Sondenapertur auf demselben Gitter wie die CTF
        lam_nm = relativistic_wavelength_nm(config.voltage_kV)
        k = torch.sqrt(probe_layer.kx ** 2 + probe_layer.ky ** 2)   # 1/nm
        alpha = lam_nm * k
        self.register_buffer("probe_aperture",
                             (alpha <= self.convergence_mrad * 1e-3).to(torch.float32))

    @staticmethod
    def _key(element: str, u_rms_A: float) -> str:
        return f"{element}@{float(u_rms_A):.4f}"

    # -- Sonde ---------------------------------------------------------------
    def probe_intensity(self):
        chi = self.probe_layer.aberration_phase()
        return probe_intensity_from_aberrations(
            chi, self.probe_aperture, self.source_size_A, self.pixel_size_A)

    def probe_profile(self):
        """Sondenintensitaet zentriert, fuer die Anzeige."""

        return torch.fft.fftshift(self.probe_intensity())

    # -- Objekt --------------------------------------------------------------
    def object_map(self, densities):
        """Querschnittsdichte aus den Elementdichten je Schicht."""

        total = None
        for i, layer_densities in enumerate(densities):
            if layer_densities is None:
                continue
            layer = self.config.layers[i]
            for element, density in layer_densities.items():
                sigma = self.cross_sections[self._key(element, layer.debye_waller_u)]
                weighted = density * sigma
                total = weighted if total is None else total + weighted

        if total is None:
            return torch.zeros((self.H + 2 * self.pad, self.W + 2 * self.pad),
                               dtype=torch.float32, device=self.probe_aperture.device)
        return total

    def object_images(self, densities):
        """Objektkarten je Schicht (fuer die Anzeige), ohne Padding."""

        pad = self.pad
        empty = np.zeros((self.H, self.W), dtype=np.float32)
        images = []
        for i, layer_densities in enumerate(densities):
            if layer_densities is None:
                images.append(empty.copy())
                continue
            layer = self.config.layers[i]
            total = None
            for element, density in layer_densities.items():
                sigma = self.cross_sections[self._key(element, layer.debye_waller_u)]
                weighted = density * sigma
                total = weighted if total is None else total + weighted
            images.append(total.detach().cpu().numpy()[pad:self.H + pad, pad:self.W + pad])
        return images

    # -- Abbildung -----------------------------------------------------------
    def forward(self, inputs):
        densities, aux_loss = inputs

        obj = self.object_map(densities)
        probe = self.probe_intensity()

        image = torch.fft.irfft2(torch.fft.rfft2(obj) * torch.fft.rfft2(probe),
                                 s=obj.shape[-2:])

        pad = self.pad
        return image[pad:self.H + pad, pad:self.W + pad], aux_loss

    # -- fuer die GUI --------------------------------------------------------
    @property
    def tunable_parameters(self):
        return self.probe_layer.tunable_parameters

    def to_diffraction_patterns(self, max_angle_mrad=None, gpts=None):
        """Sonde im k-Raum - die GUI zeigt davon die Phase, wie sonst die CTF."""

        chi = self.probe_layer.aberration_phase()
        wave = self.probe_aperture * torch.exp(-1j * chi)
        return wave, self.probe_aperture, self.probe_aperture, self.probe_profile()
