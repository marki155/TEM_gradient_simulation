r"""Part I, Section 6: the multislice method.

Theory
======

Wave equation in the specimen
-----------------------------
A fast electron with wave function :math:`\Psi(\mathbf r, z)` travelling along
:math:`z` obeys, after separating the fast phase :math:`\mathrm e^{2\pi \mathrm i z/\lambda}`
and neglecting backscattering (the paraxial / high-energy approximation),

.. math::

    \frac{\partial\psi}{\partial z} = \frac{\mathrm i\lambda}{4\pi}\nabla_\perp^2\psi
        + \mathrm i\sigma\,V(\mathbf r, z)\,\psi,
    \qquad (6.1)

with the interaction parameter :math:`\sigma` of Eq. (2.3) and the potential
:math:`V` in volts.  The first term is diffraction (free propagation), the second
scattering by the specimen.

Splitting the operators
-----------------------
Cutting the specimen into slices of thickness :math:`\Delta z_j` and treating the
two terms of Eq. (6.1) alternately (a first-order operator splitting; Cowley and
Moodie) gives for slice :math:`j`

.. math::

    \psi_{j}^{+} = t_j\,\psi_j, \qquad
    t_j(\mathbf r) = \exp\!\big(\mathrm i\sigma V^{\mathrm{proj}}_j(\mathbf r)\big)
    \qquad (6.2)

(the *phase grating*, :math:`V^{\mathrm{proj}}_j = \int V\,dz` over the slice)
followed by free propagation over the distance :math:`\Delta z` to the next slice,
which is a multiplication in Fourier space,

.. math::

    \tilde\psi_{j+1}(\mathbf k) = P(\mathbf k;\Delta z)\,\tilde\psi_j^{+}(\mathbf k),
    \qquad
    P(\mathbf k;\Delta z) = \exp\!\big(-\mathrm i\pi\lambda\,\Delta z\,|\mathbf k|^2\big).
    \qquad (6.3)

Eq. (6.3) is the paraxial (Fresnel) propagator.  Propagators compose,
:math:`P(\Delta z_1)P(\Delta z_2) = P(\Delta z_1+\Delta z_2)`; therefore empty
slices need not be visited and the remaining path to the end of the stack is a
single propagation (:func:`exit_wave`).

Slice order
-----------
Each layer carries its own height :math:`z_\ell`.  The beam passes the layers in
ascending :math:`z` for ``beam_direction = "+z"`` and in descending :math:`z` for
``"-z"``; the propagation distance between two layers is the difference of their
own heights (never the distance on the other side).  The exit plane is at the
height of the last layer in beam direction.

Sampling and aliasing (band limit)
----------------------------------
On a grid of spacing :math:`\Delta x` the highest representable frequency is
:math:`k_N = 1/(2\Delta x)`.  The transmission function :math:`\exp(\mathrm i\sigma V)`
contains harmonics of all orders of the atomic potential; the product with the
wave folds frequencies above :math:`k_N` back into the band (aliasing).  The
standard remedy is to restrict the wave to :math:`|\mathbf k| \le \tfrac23 k_N`
after every transmission (Kirkland, ch. 6): then the products of two band-limited
waves fit into the grid again.  This is done by setting the propagator to zero
outside the disk,

.. math::

    P_{\mathrm{bl}}(\mathbf k) = P(\mathbf k)\;\Theta\big(f\,k_N - |\mathbf k|\big),
    \qquad f = \tfrac23.
    \qquad (6.4)

``band_limit = 0`` switches the limit off (the behaviour of all earlier versions,
and the setting used when legacy files are loaded).  Honest remark: for the
sparse, weakly scattering specimens of this project the numerical gain could not
be demonstrated against a much finer reference (differences stay at the
:math:`10^{-3}` level with or without the limit); the limit is kept as the
textbook default because it guarantees an alias-free wave for strongly scattering
or coarsely sampled specimens, and costs nothing inside the retained band.

Projected potential of a slice
------------------------------
The potential of a slice is the convolution of the atom density with the atomic
kernel, Eq. (4.2).  Elements of mixed layers add; a constant offset is removed
(the median, a global phase that does not influence intensities but keeps the
numbers small), and the result is scaled by the group's contrast factor
:math:`c_g`:

.. math::

    V^{\mathrm{proj}}_\ell = c_{g(\ell)}\Big[\sum_e \rho_{\ell,e} * v_e
    - \operatorname{median}\Big(\sum_e \rho_{\ell,e} * v_e\Big)\Big].
    \qquad (6.5)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch

from . import electron
from .scattering import KernelBank
from .structure import AtomLayer, element_densities


# ---------------------------------------------------------------------------
# potential of one slice
# ---------------------------------------------------------------------------

def slice_potential(layer: AtomLayer, bank: KernelBank, shape, padding: int,
                    subtract_median: bool = True) -> torch.Tensor:
    """Eq. (6.5) on the padded grid, in V nm (float32)."""

    densities = element_densities(layer, shape, padding)
    u_nm = layer.debye_waller_u_A * 0.1
    spectrum = None
    for symbol, rho in densities.items():
        part = torch.fft.rfft2(rho) * bank.spectrum(symbol, u_nm)
        spectrum = part if spectrum is None else spectrum + part
    potential = torch.fft.irfft2(spectrum, s=rho.shape)
    if subtract_median:
        potential = potential - potential.median()
    return potential * layer.contrast


# ---------------------------------------------------------------------------
# propagation
# ---------------------------------------------------------------------------

class Multislice:
    """Phase grating + Fresnel propagation on a fixed padded grid.

    ``shape`` is the *padded* grid.  Propagators are cached per distance.
    """

    def __init__(self, shape, pixel_size_nm: float, voltage_kV: float, band_limit: float = 2.0 / 3.0,
                 device="cpu"):
        self.shape = (int(shape[0]), int(shape[1]))
        self.pixel_size_nm = float(pixel_size_nm)
        self.voltage_kV = float(voltage_kV)
        self.band_limit = float(band_limit)
        self.device = device
        self.wavelength_nm = electron.wavelength_nm(voltage_kV)
        self.sigma = electron.interaction_parameter(voltage_kV)
        fr = torch.fft.fftfreq(self.shape[0], d=self.pixel_size_nm, device=device)
        fc = torch.fft.fftfreq(self.shape[1], d=self.pixel_size_nm, device=device)
        kr, kc = torch.meshgrid(fr, fc, indexing="ij")
        self._k2 = kr ** 2 + kc ** 2
        if self.band_limit > 0:
            k_limit = self.band_limit / (2.0 * self.pixel_size_nm)
            self._mask = (self._k2 <= k_limit ** 2).to(torch.float32)
        else:
            self._mask = None
        self._propagators: Dict[float, torch.Tensor] = {}

    def propagator(self, dz_nm: float) -> torch.Tensor:
        """Eq. (6.3)/(6.4)."""

        key = round(float(dz_nm), 9)
        if key not in self._propagators:
            p = torch.exp(-1j * math.pi * self.wavelength_nm * key * self._k2)
            if self._mask is not None:
                p = p * self._mask
            self._propagators[key] = p
        return self._propagators[key]

    def propagate(self, wave: torch.Tensor, dz_nm: float) -> torch.Tensor:
        return torch.fft.ifft2(torch.fft.fft2(wave) * self.propagator(dz_nm))

    def transmission(self, potential: torch.Tensor) -> torch.Tensor:
        """Eq. (6.2)."""

        return torch.exp(1j * self.sigma * potential)

    def plan(self, potentials: Sequence[Optional[torch.Tensor]], z_nm: Sequence[float],
             reverse: bool = False):
        """Order of the contributing layers, distances between them and to the exit plane.

        Returns ``(order, distances, exit_distance)``: ``order`` lists the indices of the
        layers with a potential in beam order, ``distances[k]`` is the free flight between
        ``order[k]`` and ``order[k+1]``, ``exit_distance`` the flight from the last
        contributing layer to the exit plane (the last layer in beam direction).
        """

        if len(potentials) != len(z_nm):
            raise ValueError(f"{len(potentials)} potentials but {len(z_nm)} heights")
        full_order = sorted(range(len(z_nm)), key=lambda i: z_nm[i])
        if reverse:
            full_order = full_order[::-1]
        order = [i for i in full_order if potentials[i] is not None]
        distances = [abs(z_nm[b] - z_nm[a]) for a, b in zip(order[:-1], order[1:])]
        exit_distance = abs(z_nm[full_order[-1]] - z_nm[order[-1]]) if order else 0.0
        return order, distances, exit_distance

    def exit_wave(self, potentials: Sequence[Optional[torch.Tensor]], z_nm: Sequence[float],
                  reverse: bool = False, incident: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Wave behind the stack for a plane incident wave (or ``incident``).

        ``potentials[i]`` is the slice at height ``z_nm[i]`` (``None``: the layer
        does not contribute and is skipped, but the exit plane stays where it is).
        """

        order, distances, exit_distance = self.plan(potentials, z_nm, reverse)
        wave = torch.ones(self.shape, dtype=torch.complex64, device=self.device) if incident is None else incident
        for k, idx in enumerate(order):
            wave = wave * self.transmission(potentials[idx])
            if k < len(order) - 1:
                wave = self.propagate(wave, distances[k])
        if order and exit_distance > 0:
            wave = self.propagate(wave, exit_distance)
        return wave
