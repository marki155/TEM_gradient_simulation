"""Part II, Section 2: data term against the legacy losses and against analytic values."""

import numpy as np
import pytest
import torch

from temsim.data.schema import LossWeights, TrainSettings
from temsim.learning import objective as obj

H, W = 220, 260


def images(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(H, W, generator=g)
    y = 0.6 * x + 0.5 * torch.randn(H, W, generator=g) + 0.2
    mask = torch.zeros(H, W)
    mask[20:190, 30:200] = 1.0
    mask[40:60, 40:60] = 0.0
    return x, y, mask


def four(t):
    return t[None, None]


@pytest.mark.parametrize("use_mask", [False, True])
def test_terms_match_legacy(legacy, use_mask):
    lf = legacy("loss_functions")
    x, y, mask = images()
    m = mask if use_mask else None
    m4 = None if m is None else four(m)
    pairs = {
        "mse": lf.MaskedMSELoss(),
        "l1": lf.MaskedL1Loss(),
        "ncc": lf.NCCLoss(),
        "local_ncc_100": lf.LocalNCCLoss(win_size=100),
        "var": lf.GlobalDifferenceVarianceLoss(reduction="mean"),
        "local_var_100": lf.LocalDifferenceVarianceLoss(win_size=100, reduction="mean"),
        "ssim": lf.GlobalSSIMLoss(data_range=1.0),
    }
    for name, old in pairs.items():
        new = obj.TERMS[name](x, y, m)
        ref = old(four(x), four(y), m4) if m4 is not None else old(four(x), four(y))
        torch.testing.assert_close(new, ref.reshape(()), rtol=1e-5, atol=1e-6, msg=name)


def test_analytic_values():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert float(obj.mse(x, x)) == 0.0
    assert float(obj.mse(x, x + 2)) == pytest.approx(4.0)
    assert float(obj.l1(x, x - 3)) == pytest.approx(3.0)
    assert float(obj.ncc_loss(x, 5 * x + 7)) == pytest.approx(0.0, abs=1e-6)
    assert float(obj.ncc_loss(x, -x)) == pytest.approx(2.0, abs=1e-6)
    assert float(obj.variance_loss(x, x + 10)) == pytest.approx(0.0, abs=1e-6)
    assert float(obj.ssim_loss(x, x)) == pytest.approx(0.0, abs=1e-6)


def test_empty_mask_gives_zero_not_nan():
    x, y, _ = images()
    zero = torch.zeros(H, W)
    for name, fn in obj.TERMS.items():
        value = float(fn(x, y, zero))
        assert np.isfinite(value), name


def test_standardise_matches_legacy_formula():
    x, y, mask = images()
    ref = mask > 0
    n = torch.tensor([0.3, -0.2])
    sx, sy = obj.standardise(x, y, ref, n)
    s = x[ref]
    mean = s.mean() + n[0] * s.std()
    std = s.std() * torch.clamp(1 + n[1], min=0.1)
    torch.testing.assert_close(sx, (x - mean) / std)
    t = y[ref]
    torch.testing.assert_close(sy, (y - t.mean()) / t.std())
    # reference statistics: zero mean, unit spread on the reference for n = 0
    sx0, _ = obj.standardise(x, y, ref, torch.zeros(2))
    assert float(sx0[ref].mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(sx0[ref].std()) == pytest.approx(1.0, abs=1e-5)


def test_normalisation_is_relative_for_tiny_images():
    x, y, _ = images()
    tiny = 1e-6 * x
    a, _ = obj.standardise(tiny, y, None, torch.tensor([0.1, 0.0]))
    b, _ = obj.standardise(x, y, None, torch.tensor([0.1, 0.0]))
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def test_gradient_matches_finite_differences():
    x, y, mask = images()
    x = x.double().requires_grad_(True)
    y = y.double()
    w = LossWeights(mse=1.0, ncc=0.5, ssim=0.3, l1=0.2, var=0.4, local_ncc_100=0.3, local_var_100=0.2)
    n = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    def f():
        value, _ = obj.data_loss(*obj.standardise(x, y, mask > 0, n), mask.double(), w)
        return value

    f().backward()
    assert torch.isfinite(x.grad).all()
    for idx in [(30, 50), (100, 120), (200, 10)]:
        step = 1e-6
        with torch.no_grad():
            x[idx] += step
            up = f()
            x[idx] -= 2 * step
            down = f()
            x[idx] += step
        assert float(x.grad[idx]) == pytest.approx(float((up - down) / (2 * step)), rel=1e-4, abs=1e-9)
    assert n.grad is not None


def test_objective_selects_target_and_mask():
    x, y, mask = images()
    other = torch.randn(H, W)
    settings = TrainSettings(loss=LossWeights(mse=1.0))
    settings.masks = {"A": mask.numpy(), "B": 1.0 - mask.numpy()}
    settings.targets = {"B": other.numpy()}
    o = obj.Objective(y.numpy(), settings)
    zero = torch.zeros(2)

    settings.active_group = None
    t, m, r = o.regions()
    assert m is None and torch.equal(t, y)
    assert bool(r.all()) and r.shape == (H, W)  # union of both masks covers everything

    settings.active_group = "A"
    t, m, r = o.regions()
    assert torch.equal(t, y) and torch.equal(m, mask > 0) and torch.equal(r, mask > 0)

    settings.active_group = "B"
    t, m, r = o.regions()
    assert torch.allclose(t, other) and m is None and torch.equal(r, (1.0 - mask) > 0)

    loss, parts = o(x, zero)
    assert "mse" in parts and float(loss) > 0
