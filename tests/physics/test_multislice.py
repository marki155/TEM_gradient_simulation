"""Part I, Section 6: multislice."""

import math

import numpy as np
import pytest
import torch

from temsim.data.schema import Layer
from temsim.physics import electron, multislice, scattering, structure
from temsim.physics.multislice import Multislice

PX = 0.01


def random_potentials(n, shape, seed=0, scale=20.0):
    g = torch.Generator().manual_seed(seed)
    return [scale * torch.rand(shape, generator=g) for _ in range(n)]


def test_matches_legacy_phase_object(legacy):
    mod = legacy("multislice")
    shape = (48, 40)
    z = [0.3, 0.0, 0.9, 0.55]
    pots = random_potentials(4, shape)
    for reverse in (False, True):
        for drop in ((), (1,), (0, 2)):
            slices = [None if i in drop else p for i, p in enumerate(pots)]
            old = mod.PhysicalPhaseObjectLayer(*shape, PX, 80.0, "cpu").forward(slices, z, reverse=reverse)
            new = Multislice(shape, PX, 80.0, band_limit=0.0).exit_wave(slices, z, reverse=reverse)
            torch.testing.assert_close(new, old, rtol=1e-4, atol=1e-5)


def test_all_none_gives_plane_wave():
    ms = Multislice((16, 16), PX, 80.0)
    out = ms.exit_wave([None, None], [0.0, 1.0])
    assert torch.all(out == 1)


def test_propagation_is_unitary_and_composes():
    ms = Multislice((64, 64), PX, 200.0, band_limit=0.0)
    wave = torch.randn(64, 64, dtype=torch.complex64)
    a = ms.propagate(ms.propagate(wave, 1.3), 2.1)
    b = ms.propagate(wave, 3.4)
    torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
    assert float((b.abs() ** 2).sum()) == pytest.approx(float((wave.abs() ** 2).sum()), rel=1e-4)


def test_talbot_self_imaging():
    """A periodic wave reappears after the Talbot distance 2 d^2 / lambda."""

    d_px = 8
    shape = (64, 64)
    ms = Multislice(shape, PX, 80.0)
    x = torch.arange(64.0)
    wave = (1.0 + 0.3 * torch.cos(2 * math.pi * x / d_px))[None, :].expand(64, 64).to(torch.complex64)
    z_talbot = 2 * (d_px * PX) ** 2 / ms.wavelength_nm
    out = ms.propagate(wave, z_talbot)
    torch.testing.assert_close(out, wave, atol=2e-4, rtol=0)
    half = ms.propagate(wave, z_talbot / 2)                         # contrast reverses (phase pi)
    expected = (1.0 - 0.3 * torch.cos(2 * math.pi * x / d_px))[None, :].expand(64, 64)
    torch.testing.assert_close(half.real, expected, atol=2e-4, rtol=0)


def test_weak_phase_object():
    ms = Multislice((32, 32), PX, 80.0)
    v = 0.01 * torch.rand(32, 32)
    out = ms.exit_wave([v], [0.0])
    torch.testing.assert_close(out.imag, ms.sigma * v, atol=1e-5, rtol=1e-3)


def test_band_limit_zeroes_outside_disk_and_stays_close_to_unlimited():
    """The 2/3 limit empties the corners of Fourier space; inside, the result hardly changes.

    A measurable accuracy gain against a 16x finer reference was *not* found for
    single atoms (differences are at the 1e-3 level either way), so this test
    documents the actual behaviour: no content below the cut-off is lost and no
    energy is created.
    """

    n, px = 128, 0.03
    bank = scattering.KernelBank((n, n), px)
    pos = torch.tensor([[n / 2 + 0.3, n / 2 - 0.2], [n / 3, n / 4]])
    layer = structure.AtomLayer(z_nm=0.0, element="W", positions=pos, weights=torch.ones(2))
    v = multislice.slice_potential(layer, bank, (n, n), 0, subtract_median=False)
    z = [0.0, 0.3, 0.6]
    waves = {}
    for band in (0.0, 2.0 / 3.0):
        waves[band] = Multislice((n, n), px, 80.0, band_limit=band).exit_wave([v, v, v, None], z + [0.9])
    fr = torch.fft.fftfreq(n, d=px)
    k = torch.sqrt(fr[:, None] ** 2 + fr[None, :] ** 2)
    k_limit = (2.0 / 3.0) / (2 * px)
    spec = torch.fft.fft2(waves[2.0 / 3.0])
    assert spec.abs()[k > k_limit * 1.001].max() < 1e-4 * spec.abs().max()
    inner = k < 0.5 * k_limit
    diff = (torch.fft.fft2(waves[0.0]) - spec).abs()[inner].max() / n ** 2
    assert diff < 5e-3
    assert waves[2.0 / 3.0].abs().pow(2).mean() <= waves[0.0].abs().pow(2).mean() * (1 + 1e-5)


def test_slice_potential_single_atom_on_pixel_equals_kernel():
    shape, pad = (32, 32), 8
    bank = scattering.KernelBank((48, 48), 0.02)
    layer = structure.AtomLayer(z_nm=0.0, element="C", positions=torch.tensor([[10.0, 12.0]]),
                                weights=torch.tensor([1.0]))
    v = multislice.slice_potential(layer, bank, shape, pad, subtract_median=False)
    expected = torch.roll(bank.kernel("C"), shifts=(10 + pad, 12 + pad), dims=(0, 1))
    torch.testing.assert_close(v, expected, rtol=1e-4, atol=1e-5 * float(expected.max()))


def test_slice_potential_contrast_and_gradient():
    bank = scattering.KernelBank((48, 48), 0.02)
    pos = torch.tensor([[10.3, 12.6]], requires_grad=True)
    w = torch.tensor([1.0], requires_grad=True)
    layer = structure.AtomLayer(z_nm=0.0, element="C", positions=pos, weights=w, contrast=2.0)
    base = structure.AtomLayer(z_nm=0.0, element="C", positions=pos.detach(), weights=w.detach())
    v = multislice.slice_potential(layer, bank, (32, 32), 8)
    v1 = multislice.slice_potential(base, bank, (32, 32), 8)
    torch.testing.assert_close(v, 2 * v1)
    v.pow(2).sum().backward()
    assert pos.grad is not None and w.grad is not None and float(pos.grad.abs().sum()) > 0
