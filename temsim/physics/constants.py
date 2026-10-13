"""Part I, Section 1: physical constants and units.

Theory
======

All of Part I is built from five fundamental constants.  They are defined **once**,
here, with the values of CODATA 2018 (the SI redefinition of 2019 fixes ``h``, ``e``
and ``c`` exactly; ``m_e`` and ``eps_0`` are measured):

=============  ===========================  ==================
symbol         meaning                      value
=============  ===========================  ==================
:math:`h`      Planck constant              6.62607015e-34 J s
:math:`e`      elementary charge            1.602176634e-19 C
:math:`m_0`    electron rest mass           9.1093837015e-31 kg
:math:`c`      speed of light               299792458 m/s
:math:`a_0`    Bohr radius                  5.29177210903e-11 m
=============  ===========================  ==================

(The previous implementation carried three slightly different electron masses and
two wavelength formulas.  The differences are of order :math:`10^{-8}` and have no
practical effect, but a single source removes the question.)

Units
-----
The simulation works in **nanometres** for lengths and **volts** for potentials:

* lengths and pixel sizes: nm,
* spatial frequencies: 1/nm,
* projected potentials: V nm (potential integrated along the beam),
* interaction parameter: rad / (V nm),
* angles: rad inside the code (mrad at the user interface),
* aberration coefficients: nm (as in the notation of Krivanek, see
  :mod:`temsim.physics.optics`).

Atomic form-factor tables are published in Angstrom; :data:`ANGSTROM` converts.

Projected potential prefactor
-----------------------------
The Fourier transform of the electrostatic potential of an atom is proportional to
its electron scattering amplitude :math:`f_e(q)` (Mott–Bethe, nonrelativistic
electron mass; the relativistic mass change enters the *interaction parameter*, not
the potential):

.. math::

    \\tilde V(q) = \\frac{h^2}{2\\pi m_0 e}\\, f_e(q)
    \\qquad (1.1)

With :math:`f_e` in Angstrom the prefactor evaluates to 47.878 V Å\\ :sup:`2`
(:data:`POTENTIAL_PREFACTOR_V_A2`).  This is the reciprocal of the constant
:math:`\\kappa = 2\\varepsilon_0 / (a_0 e)` that appears in the Bohr-radius form of
the literature (and in abTEM); the equality is checked in the tests.
"""

from __future__ import annotations

import math

PLANCK = 6.62607015e-34            # J s
ELEMENTARY_CHARGE = 1.602176634e-19  # C
ELECTRON_MASS = 9.1093837015e-31   # kg
SPEED_OF_LIGHT = 299792458.0       # m/s
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
BOHR_RADIUS = 5.29177210903e-11    # m

#: 1 Angstrom in nm.
ANGSTROM = 0.1

#: Electron rest energy m_0 c^2 in J.
REST_ENERGY = ELECTRON_MASS * SPEED_OF_LIGHT ** 2

#: h^2 / (2 pi m_0 e) in V Angstrom^2, Eq. (1.1).
POTENTIAL_PREFACTOR_V_A2 = (PLANCK ** 2 / (2.0 * math.pi * ELECTRON_MASS * ELEMENTARY_CHARGE)) * 1e20
