"""Part I, Section 8: image formation."""

import math

import numpy as np
import pytest
import torch

from temsim.physics import electron, imaging, optics, scattering, structure
from temsim.physics.optics import FourierGrid

LAM = electron.wavelength_nm(80.0)


@pytest.mark.parametrize("element,u", [("C", 0.0), ("C", 0.06), ("W", 0.08), ("Mo", 0.05), ("S", 0.07)])
def test_cross_section_matches_legacy(legacy, element, u):
    old = legacy("stem").haadf_cross_section(element, 60.0, 200.0, 80.0, u)
    new = imaging.haadf_cross_section_A2(element, 60.0, 200.0, 80.0, u)
    assert new == pytest.approx(old, rel=2e-5)


def test_cross_section_grows_with_z_roughly_z17():
    z = {"C": 6, "Si": 14, "Mo": 42, "W": 74}
    s = {e: imaging.haadf_cross_section_A2(e, 60.0, 200.0, 80.0, 0.06) for e in z}
    slope = np.polyfit(np.log(list(z.values())), np.log(list(s.values())), 1)[0]
    assert 1.4 < slope < 2.1


def test_cross_section_lobato_close_to_kirkland():
    a = imaging.haadf_cross_section_A2("Mo", 60.0, 200.0, 80.0, 0.06, "kirkland")
    b = imaging.haadf_cross_section_A2("Mo", 60.0, 200.0, 80.0, 0.06, "lobato")
    assert b == pytest.approx(a, rel=0.1)


def test_probe_intensity_matches_legacy(legacy):
    stem = legacy("stem")
    shape, px = (48, 48), 0.02
    grid = FourierGrid(shape, px, LAM)
    chi = optics.phase_error(grid, {"C10": 2.0, "C12": 1.0, "phi12": 0.3, "C30": 0.5})
    ap = optics.aperture(grid, 25.0)
    for s in (0.0, 0.5):
        old = stem.probe_intensity_from_aberrations(chi, ap, s, px * 10)
        new = imaging.probe_intensity(chi, ap, s, px * 10)
        torch.testing.assert_close(new, old, rtol=1e-4, atol=1e-8)
        assert float(new.sum()) == pytest.approx(1.0, rel=1e-5)


def test_probe_is_centred_and_aberration_free_is_symmetric():
    grid = FourierGrid((64, 64), 0.02, LAM)
    p = imaging.probe_intensity(optics.phase_error(grid, {}), optics.aperture(grid, 25.0), 0.0, 0.2)
    assert int(torch.argmax(p)) == 0                                   # maximum at pixel [0, 0]
    torch.testing.assert_close(p[1, 0], p[-1, 0], rtol=1e-4, atol=1e-9)


def test_stem_image_single_atom_is_probe_times_cross_section():
    shape, pad = (32, 32), 8
    padded = (48, 48)
    grid = FourierGrid(padded, 0.02, LAM)
    probe = imaging.probe_intensity(optics.phase_error(grid, {}), optics.aperture(grid, 25.0), 0.5, 0.2)
    layer = structure.AtomLayer(z_nm=0.0, element="W", positions=torch.tensor([[16.0, 16.0]]),
                                weights=torch.tensor([1.0]), debye_waller_u_A=0.06)
    cs = imaging.CrossSections(60.0, 200.0, 80.0)
    dens = structure.element_densities(layer, shape, pad)
    obj = imaging.object_map([dens, None], [0.06, 0.06], cs, padded)
    img = imaging.stem_image(obj, probe, shape, pad)
    assert img.shape == shape
    full = imaging.stem_image(obj, probe, padded, 0)                                # no cropping loss
    assert float(full.sum()) == pytest.approx(cs("W", 0.06), rel=1e-4)               # probe is normalised
    assert float(img.sum()) < float(full.sum())
    assert tuple(np.unravel_index(int(img.argmax()), shape)) == (16, 16)


def test_tem_image_identity_and_signal_options():
    shape, pad = (24, 24), 8
    padded = (40, 40)
    wave = torch.exp(1j * 0.3 * torch.rand(padded))
    ones = torch.ones(padded, dtype=torch.complex64)
    a = imaging.tem_image(wave, ones, torch.ones(padded), shape, pad, "amplitude")
    i = imaging.tem_image(wave, ones, torch.ones(padded), shape, pad, "intensity")
    torch.testing.assert_close(a, torch.ones(shape), atol=1e-5, rtol=0)         # pure phase object
    torch.testing.assert_close(i, a ** 2, atol=1e-5, rtol=0)
    with pytest.raises(ValueError):
        imaging.tem_image(wave, ones, torch.ones(padded), shape, pad, "bogus")


def test_tem_weak_phase_contrast_follows_ctf():
    """Linear imaging of a weak phase object: image - 1 = sigma V * sin(chi)-kernel (Scherzer's picture)."""

    n = 64
    px = 0.02
    grid = FourierGrid((n, n), px, LAM)
    x = torch.arange(n, dtype=torch.float32)
    v = 0.02 * torch.cos(2 * math.pi * 4 * x / n)[None, :].expand(n, n) / electron.interaction_parameter(80.0)
    wave = torch.exp(1j * electron.interaction_parameter(80.0) * v)
    chi = optics.phase_error(grid, {"C10": -20.0})
    t = optics.transfer_function(grid, {"C10": -20.0}, 0.0)
    img = imaging.tem_image(wave, t, torch.ones(n, n), (n, n), 0, "intensity")
    k = 4 / (n * px)
    chi_k = float(2 * math.pi / LAM * 0.5 * (-20.0) * (LAM * k) ** 2)
    expected = 1 + 2 * 0.02 * math.sin(chi_k) * torch.cos(2 * math.pi * 4 * x / n)[None, :]
    torch.testing.assert_close(img, expected.expand(n, n), atol=1e-3, rtol=0)
