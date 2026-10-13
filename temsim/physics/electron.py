"""Part I, Section 2: the fast electron.

Theory
======

An electron accelerated through the voltage :math:`U` has the kinetic energy
:math:`E = eU`.  At 80 kV this is already 16 % of the rest energy, so the
relativistic relations are used throughout.

Momentum and wavelength
-----------------------
.. math::

    p = \\frac{1}{c}\\sqrt{E^2 + 2 E m_0 c^2}, \\qquad
    \\lambda = \\frac{h}{p}
    \\qquad (2.1)

(:func:`wavelength_nm`; 4.18 pm at 80 kV, 2.51 pm at 200 kV.)

Relativistic factor
-------------------
.. math::

    \\gamma = 1 + \\frac{E}{m_0 c^2}
    \\qquad (2.2)

Interaction parameter
---------------------
Inside the specimen the electron wave :math:`\\psi` obeys the Schroedinger equation
with the relativistically corrected mass :math:`\\gamma m_0` and the potential
:math:`V(\\mathbf r)`.  Within the projection (phase-grating) approximation of
Section 4 the wave acquires, per slice, the phase

.. math::

    \\varphi(x,y) = \\sigma\\, V_{\\mathrm{proj}}(x,y), \\qquad
    \\sigma = \\frac{2\\pi\\,\\gamma\\, m_0\\, e\\, \\lambda}{h^2}
    \\qquad (2.3)

with the projected potential :math:`V_{\\mathrm{proj}} = \\int V\\,dz`.  In the units
of :mod:`temsim.physics.constants` :math:`\\sigma` is given in rad/(V nm)
(:func:`interaction_parameter`); at 80 kV it is 0.0101 rad/(V nm), at 200 kV
0.0073 rad/(V nm).
"""

from __future__ import annotations

import math

from .constants import (ELECTRON_MASS, ELEMENTARY_CHARGE, PLANCK, REST_ENERGY,
                        SPEED_OF_LIGHT)


def kinetic_energy_J(voltage_kV: float) -> float:
    return ELEMENTARY_CHARGE * voltage_kV * 1e3


def lorentz_factor(voltage_kV: float) -> float:
    """Eq. (2.2)."""

    return 1.0 + kinetic_energy_J(voltage_kV) / REST_ENERGY


def wavelength_nm(voltage_kV: float) -> float:
    """Eq. (2.1), in nm."""

    energy = kinetic_energy_J(voltage_kV)
    momentum = math.sqrt(energy ** 2 + 2.0 * energy * REST_ENERGY) / SPEED_OF_LIGHT
    return PLANCK / momentum * 1e9


def interaction_parameter(voltage_kV: float) -> float:
    """Eq. (2.3), in rad / (V nm)."""

    lam_m = wavelength_nm(voltage_kV) * 1e-9
    sigma_si = (2.0 * math.pi * lorentz_factor(voltage_kV) * ELECTRON_MASS
                * ELEMENTARY_CHARGE * lam_m / PLANCK ** 2)          # rad / (V m)
    return sigma_si * 1e-9
