"""Part I, Section 7: objective lens."""

import math

import pytest
import torch

from temsim.physics import electron, optics
from temsim.physics.optics import Envelope, FourierGrid

LAM = electron.wavelength_nm(80.0)
SHAPE, PX = (40, 56), 0.02


def legacy_grid(ctf):
    fx = torch.fft.fftshift(torch.linspace(-.5 / PX, .5 / PX, SHAPE[0]))
    fy = torch.fft.fftshift(torch.linspace(-.5 / PX, .5 / PX, SHAPE[1]))
    return torch.meshgrid(fx, fy, indexing="ij")


COEFFS = {"C10": -13.0, "C12": 2.0, "phi12": 0.4, "C21": 30.0, "phi21": 1.1, "C23": 20.0, "phi23": -0.3,
          "C30": 1.2, "C34": 0.3, "phi34": 0.2, "C45": 0.1, "phi45": 0.7, "C32": 5.0, "phi32": 0.9}


def test_polar_cartesian_roundtrip():
    cart = optics.polar_to_cartesian(COEFFS)
    assert "phi12" not in cart and "a12" in cart and cart["C10"] == COEFFS["C10"]
    back = optics.cartesian_to_polar(cart)
    for key, value in COEFFS.items():
        m = optics.ABERRATION_TERMS[optics.magnitude_key(key)][3]
        if key.startswith("phi"):                      # angle defined modulo 2 pi / m
            diff = (back[key] - value) * m
            assert math.isclose(math.sin(diff), 0, abs_tol=1e-9) and math.cos(diff) > 0
        else:
            assert back[key] == pytest.approx(value)


def test_phase_matches_legacy_polar_and_cartesian(legacy):
    mod = legacy("ctf")
    kx, ky = legacy_grid(mod)
    grid = FourierGrid(SHAPE, PX, LAM, k_row=kx, k_col=ky)
    alpha = LAM * torch.sqrt(kx ** 2 + ky ** 2)
    phi = torch.atan2(ky, kx)
    for coeffs in (COEFFS, optics.polar_to_cartesian(COEFFS)):
        tensors = {k: torch.tensor(float(v)) for k, v in coeffs.items()}
        old = 2 * math.pi * mod.zernike_phase(alpha, phi, tensors) / LAM
        new = optics.phase_error(grid, tensors)
        torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-4)


def test_polar_equals_cartesian():
    grid = FourierGrid(SHAPE, PX, LAM)
    a = optics.phase_error(grid, {k: torch.tensor(float(v)) for k, v in COEFFS.items()})
    b = optics.phase_error(grid, {k: torch.tensor(float(v)) for k, v in optics.polar_to_cartesian(COEFFS).items()})
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-4)


def test_defocus_and_cs_closed_form():
    grid = FourierGrid((64, 64), PX, LAM)
    chi = optics.phase_error(grid, {"C10": -10.0, "C30": 1.0})
    k2 = grid.k2
    expected = math.pi * LAM * (-10.0) * k2 + 0.5 * math.pi * 1.0e3 * LAM ** 3 * k2 ** 2
    torch.testing.assert_close(chi, expected, rtol=1e-4, atol=1e-4)


def test_envelope_matches_legacy(legacy):
    mod = legacy("ctf")
    kx, ky = legacy_grid(mod)
    grid = FourierGrid(SHAPE, PX, LAM, k_row=kx, k_col=ky)
    params = {"source_size": 0.2e-3, "use_spatial": True, "use_chromatic": True, "use_instability": True,
              "instab_defocus_nm": 1.5, "use_detector": True, "detector_sigma_k": 20.0}
    old = mod.build_ctf_envelope_on_k(kx, ky, 80.0, LAM, params, deltaE=torch.tensor(0.35),
                                      Cc_um=torch.tensor(1500.0), defocus_nm=torch.tensor(-13.0),
                                      cs_nm=torch.tensor(1200.0))
    new = optics.envelope(grid, Envelope.from_dict(params), 80.0, torch.tensor(0.35), torch.tensor(1500.0),
                          defocus_nm=torch.tensor(-13.0), cs_nm=torch.tensor(1200.0))
    torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)


def test_transfer_function_modulus_and_amplitude_contrast():
    grid = FourierGrid((32, 32), PX, LAM)
    coeffs = {"C10": -5.0, "C30": 1.0}
    t = optics.transfer_function(grid, coeffs, 0.0)
    torch.testing.assert_close(t.abs(), torch.ones(32, 32), rtol=1e-5, atol=1e-5)
    t = optics.transfer_function(grid, coeffs, 0.3)
    torch.testing.assert_close(t.abs(), torch.ones(32, 32), rtol=1e-5, atol=1e-5)
    assert float(t[0, 0].imag) == pytest.approx(-0.3, abs=1e-6)       # k = 0: chi = 0
    t = optics.transfer_function(grid, coeffs, 2.0)                   # clipped to 1
    assert float(t[0, 0].imag) == pytest.approx(-1.0, abs=1e-6)


def test_gradients_flow_to_coefficients():
    grid = FourierGrid((32, 32), PX, LAM)
    c = {"a12": torch.tensor(1.0, requires_grad=True), "b12": torch.tensor(0.5, requires_grad=True),
         "C10": torch.tensor(-3.0, requires_grad=True)}
    de = torch.tensor(0.35, requires_grad=True)
    env = optics.envelope(grid, Envelope(source_size=1e-4, instab_defocus_nm=0.5), 80.0, de,
                          torch.tensor(1500.0), defocus_nm=c["C10"], cs_nm=1200.0)
    t = optics.transfer_function(grid, c, 0.1, env)
    t.real.sum().backward()
    assert all(v.grad is not None and float(v.grad.abs()) > 0 for v in (*c.values(), de))


def test_aperture():
    grid = FourierGrid((64, 64), PX, LAM)
    ap = optics.aperture(grid, 10.0)
    assert ap[0, 0] == 1 and ap.sum() < 64 * 64
    assert float(grid.alpha[ap > 0].max()) <= 10e-3 + 1e-9
    assert float(grid.alpha[ap == 0].min()) > 10e-3
