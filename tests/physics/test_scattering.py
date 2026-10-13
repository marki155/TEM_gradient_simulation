"""Part I, Sections 1-3: constants, electron, atomic potentials."""

import math

import numpy as np
import pytest
import torch
from scipy import integrate, special

from temsim.physics import constants, electron, scattering


def test_prefactor_equals_bohr_radius_form():
    kappa = 2 * constants.VACUUM_PERMITTIVITY / (constants.BOHR_RADIUS * constants.ELEMENTARY_CHARGE) * 1e-20
    assert 1.0 / kappa == pytest.approx(constants.POTENTIAL_PREFACTOR_V_A2, rel=2e-8)


def test_wavelength_reference_values():
    assert electron.wavelength_nm(80) == pytest.approx(4.1757e-3, rel=1e-4)
    assert electron.wavelength_nm(200) == pytest.approx(2.5079e-3, rel=1e-4)
    assert electron.wavelength_nm(300) == pytest.approx(1.9687e-3, rel=1e-4)


def test_wavelength_nonrelativistic_limit():
    # lambda [nm] = 1.2264 / sqrt(U [V]) for U -> small
    lam = electron.wavelength_nm(0.1)
    assert lam == pytest.approx(1.22643 / math.sqrt(100.0), rel=2e-4)


def test_interaction_parameter_reference():
    # Kirkland: sigma(100 kV) = 0.009244 rad/(V nm) ... 0.00924 per V nm
    assert electron.interaction_parameter(100) == pytest.approx(0.009244, rel=1e-3)


@pytest.mark.parametrize("parametrization,element", [("kirkland", "C"), ("kirkland", "W"),
                                                      ("lobato", "C"), ("lobato", "S")])
def test_projected_potential_is_hankel_transform_of_scattering_amplitude(parametrization, element):
    """Eq. (3.1) evaluated by quadrature equals the closed forms (3.2)/(3.3)."""

    def closed(r):
        return float(scattering.projected_potential_VA(torch.tensor([r]), element, parametrization))

    q = np.linspace(0.0, 600.0, 1_200_001)
    amplitude = scattering.scattering_amplitude_A(q, element, parametrization)

    def numeric(r):
        integrand = amplitude * special.j0(2 * math.pi * q * r) * 2 * math.pi * q
        return constants.POTENTIAL_PREFACTOR_V_A2 * integrate.simpson(integrand, x=q)

    for r in (0.25, 0.5, 1.0, 2.0):
        assert closed(r) == pytest.approx(numeric(r), rel=2e-3)


@pytest.mark.parametrize("parametrization", ["kirkland", "lobato"])
def test_matches_abtem(parametrization):
    abtem = pytest.importorskip("abtem.parametrizations")
    cls = {"kirkland": abtem.KirklandParametrization, "lobato": abtem.LobatoParametrization}[parametrization]
    par = cls()
    r = np.linspace(0.05, 4.0, 60)
    for element in ("C", "S", "W", "Mo"):
        ref = par.projected_potential(element)(r)
        own = scattering.projected_potential_VA(torch.from_numpy(r), element, parametrization).numpy()
        np.testing.assert_allclose(own, ref, rtol=2e-5)


def test_all_elements_tabulated():
    assert len(scattering.available_elements("kirkland")) == 103
    assert len(scattering.available_elements("lobato")) == 103


def test_kernel_matches_legacy():
    """Vectorised supersampled kernel == legacy loop implementation."""

    legacy = pytest.importorskip("kirkland_supersampled")
    for shape, px in (((64, 64), 0.02), ((65, 48), 0.0125)):
        for element in ("C", "W"):
            old = legacy.projected_atomic_potential_kernel_kirkland(
                shape[0], shape[1], element, px, "cpu", supersample_radius_px=2.0,
                supersample_factor=8, softcore_fraction=0.001).numpy()
            new = scattering.real_space_kernel(shape, px, element).numpy()
            np.testing.assert_allclose(new, old, rtol=2e-5, atol=1e-6 * np.abs(old).max())


def test_kernel_area_integral_is_scattering_amplitude_at_zero():
    """Sum over pixels * pixel area = V_tilde(k=0) = C f_e(0) (the integral is conserved)."""

    px = 0.01
    shape = (512, 512)
    kernel = scattering.real_space_kernel(shape, px, "C")
    area_integral = float(kernel.sum()) * px ** 2                     # V nm * nm^2
    expected = constants.POTENTIAL_PREFACTOR_V_A2 * float(scattering.scattering_amplitude_A(np.array(0.0), "C")) * 1e-3
    assert area_integral == pytest.approx(expected, rel=1e-2)       # truncated tail of the 1/q^2 term


def test_debye_waller_conserves_area_and_lowers_peak():
    bank = scattering.KernelBank((128, 128), 0.02, "kirkland")
    plain = bank.spectrum("C", 0.0)
    smeared = bank.spectrum("C", 0.01)
    assert torch.allclose(plain[0, 0], smeared[0, 0])
    assert torch.fft.irfft2(smeared, s=(128, 128))[0, 0] < torch.fft.irfft2(plain, s=(128, 128))[0, 0]


def test_bank_allows_same_element_with_two_amplitudes():
    bank = scattering.KernelBank((64, 64), 0.02)
    assert bank.spectrum("C", 0.005) is not bank.spectrum("C", 0.01)
