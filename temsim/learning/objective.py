r"""Part II, Section 2: the data term of the objective.

Theory
======

Why not simply a mean squared error?
------------------------------------
Simulated and measured images do not share units or offset.  A TEM image from a
detector is in counts, a simulated one in units of the incident intensity; a
HAADF image is in cross-section units of order :math:`10^{-6}`.  Both are
therefore *standardised* before they are compared, using the statistics of a
*reference region* :math:`\mathcal R` (Section 2.3): for an image :math:`I`

.. math::

    \hat I = \frac{I - \mu_{\mathcal R}(I)}{\sigma_{\mathcal R}(I)},
    \qquad (2.1)

with :math:`\mu` the mean and :math:`\sigma` the sample standard deviation over
:math:`\mathcal R`.  For the *simulation* two trainable numbers
:math:`(n_0, n_1)` correct this standardisation *relative to the spread* of the
image,

.. math::

    \mu_{\rm sim} = \mu_{\mathcal R}(I_{\rm sim}) + n_0\,\sigma_{\mathcal R}(I_{\rm sim}),
    \qquad
    \sigma_{\rm sim} = \sigma_{\mathcal R}(I_{\rm sim})\,\max(1+n_1,\;0.1).
    \qquad (2.2)

Relative corrections matter: the HAADF image has a spread of :math:`\sim 10^{-6}`,
and an *additive* offset of :math:`10^{-4}` would have turned the denominator
negative.  For :math:`n_0 = n_1 = 0` Eq. (2.2) is the plain standardisation.  The
standard deviations are clamped from below by :math:`10^{-8}`, not shifted by it
(a shift of that size would already be a 1 % error at a spread of
:math:`7\cdot 10^{-7}`).

Terms
-----
Let :math:`M` be the *loss mask* (1 where a pixel counts), :math:`N = \sum M`, and
:math:`x = \hat I_{\rm sim}`, :math:`y = \hat I_{\rm exp}`, :math:`d = x - y`.
Sums run over all pixels, weighted by :math:`M`.

.. math::

    \mathcal L_{\rm mse} = \frac{1}{N}\sum M\,d^2,
    \qquad
    \mathcal L_{1} = \frac{1}{N}\sum M\,|d|,
    \qquad (2.3)

.. math::

    \mathcal L_{\rm ncc} = 1 - \frac{\sum M\,(x-\bar x)(y-\bar y)}
    {\sqrt{\sum M(x-\bar x)^2\,\sum M(y-\bar y)^2}},
    \qquad (2.4)

where the means are taken over :math:`M`.  The denominator is clamped to
:math:`\ge 10^{-8}`.  :math:`\mathcal L_{\rm ncc}` ignores gain and offset of the
simulation altogether; the standardisation matters for the other terms and for
the *relative* weight of terms.

.. math::

    \mathcal L_{\rm var} = \frac{1}{N}\sum M\,(d - \bar d)^2
    \qquad (2.5)

is the variance of the difference: it ignores an offset but not a gain error.

.. math::

    \mathcal L_{\rm ssim} = 1 - \frac{(2\bar x\bar y + c_1)(2\sigma_{xy} + c_2)}
    {(\bar x^2 + \bar y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)},
    \quad c_1 = (0.01\,L)^2,\; c_2 = (0.03\,L)^2,\; L = 1,
    \qquad (2.6)

evaluated once for the whole mask (the *global* SSIM; the data range
:math:`L = 1` is a convention, the images being standardised).

Local variants cut the image into non-overlapping windows of :math:`w\times w`
pixels (the image is cropped to a multiple of :math:`w`), evaluate (2.4) or (2.5)
in every window using only masked pixels, and average over the windows that
contain at least one masked pixel.  They reward agreement of the *local* contrast
and are insensitive to slow drifts of brightness across the field of view.  The
window is 100 px.

The data term is the weighted sum of the terms whose weight is positive,

.. math::

    \mathcal L_{\rm data} = a_{\rm mse}\mathcal L_{\rm mse} + a_{\rm ncc}\mathcal L_{\rm ncc}
    + a_{\rm lncc}\mathcal L_{\rm lncc} + a_{\rm ssim}\mathcal L_{\rm ssim}
    + a_{1}\mathcal L_1 + a_{\rm var}\mathcal L_{\rm var}
    + a_{\rm lvar}\mathcal L_{\rm lvar}.
    \qquad (2.7)

Target, loss mask and reference region
--------------------------------------
Which pixels count depends on the *active group* :math:`g`
(:attr:`~temsim.data.schema.TrainSettings.active_group`):

.. list-table::
   :header-rows: 1

   * - situation
     - target
     - loss mask :math:`M`
     - reference :math:`\mathcal R`
   * - no active group
     - global target
     - all pixels
     - union of all masks (all pixels without masks)
   * - group with its own target
     - group target
     - all pixels
     - mask of the group
   * - group without target
     - global target
     - mask of the group
     - mask of the group

The reference region is the part of the image where simulation and measurement
are known to describe the same object; the standardisation (2.1) uses it so that
a bright support or an unexplained area does not bias the statistics.

Implementation
==============

The functions of this module are plain tensor functions on ``(H, W)`` images, so
that they can be tested against analytic values.  A mask of ``None`` is the same
as a mask of ones.  :class:`Objective` binds them to the
:class:`~temsim.data.schema.TrainSettings` and the measured image.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ..data.schema import TrainSettings

EPS = 1e-8
LOCAL_WINDOW = 100


# ---------------------------------------------------------------------------
# standardisation
# ---------------------------------------------------------------------------

def standardise(simulation: torch.Tensor, target: torch.Tensor, reference: Optional[torch.Tensor],
                normalisation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eqs. (2.1) and (2.2); ``reference`` is a boolean mask or ``None`` (all pixels)."""

    if reference is None:
        sim_ref, tgt_ref = simulation.flatten(), target.flatten()
    else:
        sim_ref, tgt_ref = simulation[reference], target[reference]
    sim_mean_raw, sim_std_raw = sim_ref.mean(), sim_ref.std()
    sim_mean = sim_mean_raw + normalisation[0] * sim_std_raw
    sim_std = sim_std_raw * torch.clamp(1.0 + normalisation[1], min=0.1)
    tgt_mean, tgt_std = tgt_ref.mean(), tgt_ref.std()
    return ((simulation - sim_mean) / torch.clamp(sim_std, min=EPS),
            (target - tgt_mean) / torch.clamp(tgt_std, min=EPS))


# ---------------------------------------------------------------------------
# terms
# ---------------------------------------------------------------------------

def _mask(mask: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(like) if mask is None else mask.to(like.dtype)


def mse(x, y, mask=None):
    m = _mask(mask, x)
    return (((x - y) ** 2) * m).sum() / m.sum().clamp_min(EPS)


def l1(x, y, mask=None):
    m = _mask(mask, x)
    return ((x - y).abs() * m).sum() / m.sum().clamp_min(EPS)


def _centred_moments(x, y, m, dims):
    n = m.sum(dim=dims, keepdim=True)
    safe = n.clamp_min(1.0)
    xc = (x - (x * m).sum(dim=dims, keepdim=True) / safe) * m
    yc = (y - (y * m).sum(dim=dims, keepdim=True) / safe) * m
    return n, xc, yc


def ncc_loss(x, y, mask=None):
    """Eq. (2.4)."""

    m = _mask(mask, x)
    n, xc, yc = _centred_moments(x, y, m, (-2, -1))
    ncc = (xc * yc).sum() / torch.sqrt((xc ** 2).sum() * (yc ** 2).sum()).clamp_min(EPS)
    return 1.0 - torch.where(n.squeeze() > 0, ncc, torch.zeros_like(ncc))


def variance_loss(x, y, mask=None):
    """Eq. (2.5)."""

    m = _mask(mask, x)
    d = x - y
    n = m.sum()
    mean = (d * m).sum() / n.clamp_min(1.0)
    var = (((d - mean) ** 2) * m).sum() / n.clamp_min(1.0)
    return torch.where(n > 0, var, torch.zeros_like(var))


def ssim_loss(x, y, mask=None, data_range: float = 1.0, k1: float = 0.01, k2: float = 0.03):
    """Eq. (2.6)."""

    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    m = _mask(mask, x)
    n = m.sum()
    safe = n.clamp_min(1.0)
    mu_x, mu_y = (x * m).sum() / safe, (y * m).sum() / safe
    xc, yc = (x - mu_x) * m, (y - mu_y) * m
    var_x, var_y, cov = (xc ** 2).sum() / safe, (yc ** 2).sum() / safe, (xc * yc).sum() / safe
    has = n > 0
    zero = torch.zeros_like(mu_x)
    mu_x, mu_y = torch.where(has, mu_x, zero), torch.where(has, mu_y, zero)
    var_x, var_y, cov = torch.where(has, var_x, zero), torch.where(has, var_y, zero), torch.where(has, cov, zero)
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2) + EPS)
    return 1.0 - ssim


def _patches(t: torch.Tensor, ws: int) -> torch.Tensor:
    """Crop to a multiple of ``ws`` and cut into windows: ``(nH, nW, ws*ws)``."""

    h, w = (t.shape[-2] // ws) * ws, (t.shape[-1] // ws) * ws
    t = t[:h, :w].reshape(h // ws, ws, w // ws, ws).permute(0, 2, 1, 3)
    return t.reshape(h // ws, w // ws, ws * ws)


def _check_window(x, ws):
    if x.shape[-2] < ws or x.shape[-1] < ws:
        raise ValueError(f"image {tuple(x.shape)} is smaller than the window of {ws} px")


def local_ncc_loss(x, y, mask=None, window: int = LOCAL_WINDOW):
    """Window-wise Eq. (2.4), averaged over windows with at least one masked pixel."""

    _check_window(x, window)
    m = _mask(mask, x)
    px, py, pm = _patches(x, window), _patches(y, window), _patches(m, window)
    n, xc, yc = _centred_moments(px, py, pm, (-1,))
    den = torch.sqrt((xc ** 2).sum(-1) * (yc ** 2).sum(-1)).clamp_min(EPS)
    valid = (n.squeeze(-1) > 0)
    ncc = torch.where(valid, (xc * yc).sum(-1) / den, torch.zeros_like(den))
    vf = valid.to(ncc.dtype)
    return 1.0 - (ncc * vf).sum() / vf.sum().clamp_min(1.0)


def local_variance_loss(x, y, mask=None, window: int = LOCAL_WINDOW):
    """Window-wise Eq. (2.5), averaged over windows with at least one masked pixel."""

    _check_window(x, window)
    m = _mask(mask, x)
    pd, pm = _patches(x - y, window), _patches(m, window)
    n = pm.sum(-1)
    safe = n.clamp_min(1.0)
    mean = (pd * pm).sum(-1, keepdim=True) / safe.unsqueeze(-1)
    var = (((pd - mean) ** 2) * pm).sum(-1) / safe
    vf = (n > 0).to(var.dtype)
    return (var * vf).sum() / vf.sum().clamp_min(1.0)


TERMS = {
    "mse": mse, "ncc": ncc_loss, "local_ncc_100": local_ncc_loss, "ssim": ssim_loss,
    "l1": l1, "var": variance_loss, "local_var_100": local_variance_loss,
}


def data_loss(x, y, mask, weights) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Eq. (2.7).  ``weights`` is a :class:`~temsim.data.schema.LossWeights`."""

    total = torch.zeros((), dtype=x.dtype, device=x.device)
    parts: Dict[str, float] = {}
    for name, fn in TERMS.items():
        a = float(getattr(weights, name))
        if a > 0:
            value = fn(x, y, mask)
            total = total + a * value
            parts[name] = float(value.detach())
    return total, parts


# ---------------------------------------------------------------------------
# binding to settings
# ---------------------------------------------------------------------------

class Objective:
    """The data term for a measured image and a :class:`TrainSettings`.

    Masks and targets are read from the settings at every call (the GUI may
    replace them), converted to tensors once and cached.
    """

    def __init__(self, target: np.ndarray, settings: TrainSettings, device="cpu"):
        self.settings = settings
        self.device = device
        self._target = self._to_tensor(target)
        self._cache: Dict[int, Tuple[np.ndarray, torch.Tensor]] = {}

    def _to_tensor(self, a) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)

    def _tensor(self, a: np.ndarray, boolean: bool = False) -> torch.Tensor:
        entry = self._cache.get(id(a))
        if entry is None or entry[0] is not a:
            t = self._to_tensor(a)
            entry = (a, (t > 0) if boolean else t)
            self._cache[id(a)] = entry
        return entry[1]

    def set_target(self, target: np.ndarray) -> None:
        self._target = self._to_tensor(target)

    @property
    def target(self) -> torch.Tensor:
        return self._target

    def union_mask(self) -> Optional[torch.Tensor]:
        masks = [self._tensor(m, True) for m in self.settings.masks.values() if m is not None]
        if not masks:
            return None
        out = masks[0]
        for m in masks[1:]:
            out = out | m
        return out

    def regions(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """``(target, loss mask, reference mask)`` for the active group (table in the module docstring)."""

        s = self.settings
        active = s.active_group
        group_mask = s.mask_for(active)
        group_mask = None if group_mask is None else self._tensor(group_mask, True)
        reference = group_mask if active is not None else self.union_mask()
        group_target = s.target_for(active)
        if group_target is not None:
            return self._tensor(group_target), None, reference
        if active is not None:
            return self._target, group_mask, reference
        return self._target, None, reference

    def __call__(self, simulation: torch.Tensor, normalisation: torch.Tensor):
        """``(loss, terms)`` for a simulated image."""

        target, mask, reference = self.regions()
        x, y = standardise(simulation, target, reference, normalisation)
        loss, parts = data_loss(x, y, mask, self.settings.loss)
        return loss, parts
