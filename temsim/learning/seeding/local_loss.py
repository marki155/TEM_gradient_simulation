r"""Part II, Section 6.4: the seeding loss from running sums.

Seeding evaluates thousands of trials; each should cost a window, not an image.
The loss used for the decision is the negative normalised cross-correlation of
the simulated image :math:`s` with the target :math:`t` (Section 2, Eq. 2.1,
:math:`\mathcal L_{\rm ncc}` without the constant):

.. math::

    \mathcal L = -\frac{1}{|K|}\sum_{p\in K} x_p\,y_p,\qquad
    x = \frac{s - a}{\sigma_s},\quad y = \frac{t - \bar t}{\sigma_t},
    \qquad (6.2)

where the statistics come from the reference region :math:`R` (all pixels that
carry any mask) with :math:`n = |R|` and the loss region :math:`K\subseteq R`:

.. math::

    \bar s = \frac{S_R}{n},\quad
    \operatorname{std}\,s = \sqrt{\frac{S^{2}_R - n\bar s^2}{n-1}},\quad
    a = \bar s + \nu_0\operatorname{std}\,s,\quad
    \sigma_s = \operatorname{std}\,s\cdot\max(1+\nu_1, 0.1),
    \qquad (6.3)

and :math:`\bar t,\sigma_t` are computed in the same way from the target.  With
the four sums of the simulated image

.. math::

    S_R=\sum_R s,\quad S^2_R=\sum_R s^2,\quad S_K=\sum_K s,\quad C_K=\sum_K s\,t
    \qquad (6.4)

the loss is a closed expression, :math:`T_K=\sum_K t`,

.. math::

    \mathcal L = -\frac{C_K - a\,T_K - \bar t\,S_K + |K|\,a\,\bar t}{|K|\,\sigma_s\sigma_t}.
    \qquad (6.5)

A change of the image by :math:`d = s^{\rm new}-s^{\rm old}` in a window :math:`\Omega`
changes the sums by

.. math::

    \Delta S_R = \sum_{R\cap\Omega} d,\quad
    \Delta S^2_R = \sum_{R\cap\Omega} d\,(s^{\rm new}+s^{\rm old}),\quad
    \Delta S_K = \sum_{K\cap\Omega} d,\quad
    \Delta C_K = \sum_{K\cap\Omega} t\,d,
    \qquad (6.6)

so a trial costs the window of Part I, Section 10 and a handful of sums.  The
sums are exact; the only approximation is that of the window.  They are rebuilt
from the full image when the image is refreshed.

Insertion gain
--------------
In STEM the image is linear in the atoms, Eq. (10.1): an atom of amplitude
:math:`A` at :math:`\mathbf p` adds :math:`d(\mathbf x) = A\,k(\mathbf x-\mathbf p)` with the probe
intensity :math:`k`.  Eq. (6.6) then reads

.. math::

    \Delta S_R(\mathbf p) = A\,(\chi_R \star k)(\mathbf p),\quad
    \Delta S^2_R(\mathbf p) = 2A\,(\chi_R s \star k)(\mathbf p) + A^2(\chi_R\star k^2)(\mathbf p),
    \qquad (6.7)

and analogously for :math:`S_K` and :math:`C_K` with :math:`\chi_K` and :math:`\chi_K t`
(:math:`\chi` the indicator of a region, :math:`\star` cross-correlation).  The
loss change of a new atom at *every* pixel therefore costs four FFT correlations,
without a single simulation.  This "landscape" drives the gradient seeding of
Section 6.3 and is exact up to the truncation of the probe.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ...physics.window import ImageChange, LocalImage
from ..objective import standardise

EPS = 1e-8


@dataclass
class Trial:
    """A proposed change with the sums and the loss it would give."""

    change: ImageChange
    sums: np.ndarray
    loss: float
    n_after: int


def negative_correlation(simulation: torch.Tensor, target: torch.Tensor, reference: Optional[torch.Tensor],
                         keep: Optional[torch.Tensor], normalisation: torch.Tensor) -> float:
    """Eq. (6.2) evaluated on the full image (the reference the running sums must reproduce)."""

    x, y = standardise(simulation, target, reference, normalisation)
    if keep is not None:
        x, y = x[keep], y[keep]
    return -float((x * y).mean())


class LocalLoss:
    """Eq. (6.5) with running sums for a :class:`LocalImage`."""

    def __init__(self, local: LocalImage, target: torch.Tensor, reference: Optional[torch.Tensor],
                 keep: Optional[torch.Tensor], normalisation, n_atoms: int):
        self.local = local
        self.device = local.device
        shape = tuple(local.sim.shape)

        def mask(m):
            if m is None:
                return torch.ones(shape, dtype=torch.float64, device=self.device)
            return m.to(self.device).to(torch.float64)

        self.t = target.to(self.device).to(torch.float64)
        self.ref = mask(reference)
        self.keep = mask(keep)
        values = torch.as_tensor(normalisation).detach().cpu().tolist()
        self.norm0, self.norm1 = float(values[0]), float(values[1])
        self.n_ref = float(self.ref.sum())
        self.n_keep = float(self.keep.sum())
        self.t_ref = float((self.ref * self.t).sum())
        self.tt_ref = float((self.ref * self.t * self.t).sum())
        self.t_keep = float((self.keep * self.t).sum())
        self.reset(n_atoms)

    # ------------------------------------------------------------------ sums
    def reset(self, n_atoms: int) -> float:
        """Rebuild the four sums of Eq. (6.4) from ``local.image``."""

        s = self.local.image.to(torch.float64)
        self.s_ref = float((self.ref * s).sum())
        self.ss_ref = float((self.ref * s * s).sum())
        self.s_keep = float((self.keep * s).sum())
        self.st_keep = float((self.keep * s * self.t).sum())
        self.n_atoms = int(n_atoms)
        self.loss = math.inf if self.n_atoms == 0 else self._loss(self.s_ref, self.ss_ref, self.s_keep, self.st_keep)
        return self.loss

    def _loss(self, s_ref, ss_ref, s_keep, st_keep):
        n = self.n_ref
        if n < 2 or self.n_keep < 1:
            return math.inf
        mean_s = s_ref / n
        std_s = math.sqrt(max((ss_ref - n * mean_s * mean_s) / (n - 1.0), 0.0))
        mean_t = self.t_ref / n
        std_t = math.sqrt(max((self.tt_ref - n * mean_t * mean_t) / (n - 1.0), 0.0))
        a = mean_s + self.norm0 * std_s
        sigma_s = max(std_s * max(1.0 + self.norm1, 0.1), EPS)
        sigma_t = max(std_t, EPS)
        cross = st_keep - a * self.t_keep - mean_t * s_keep + self.n_keep * a * mean_t
        loss = -cross / (self.n_keep * sigma_s * sigma_t)
        return loss if math.isfinite(loss) else math.inf

    # ------------------------------------------------------------------ trial
    def propose(self, layer: int, changes, n_after: int) -> Optional[Trial]:
        change = self.local.propose(layer, changes)
        if change is None:
            return None
        rows, cols = change.rows, change.cols
        new, old = change.new.to(torch.float64), change.old.to(torch.float64)
        d = new - old
        ref, keep, t = self.ref[rows, cols], self.keep[rows, cols], self.t[rows, cols]
        sums = torch.stack([(ref * d).sum(), (ref * d * (new + old)).sum(),
                            (keep * d).sum(), (keep * t * d).sum()]).cpu().numpy()
        if n_after == 0:
            loss = math.inf
        else:
            loss = self._loss(self.s_ref + sums[0], self.ss_ref + sums[1], self.s_keep + sums[2],
                              self.st_keep + sums[3])
        return Trial(change, sums, loss, int(n_after))

    def commit(self, trial: Trial) -> None:
        self.local.commit(trial.change)
        self.s_ref += float(trial.sums[0])
        self.ss_ref += float(trial.sums[1])
        self.s_keep += float(trial.sums[2])
        self.st_keep += float(trial.sums[3])
        self.loss = trial.loss
        self.n_atoms = trial.n_after

    @contextlib.contextmanager
    def hypothetical(self, trial: Trial):
        """Accept ``trial`` only temporarily (STEM; the TEM state has more than the image)."""

        local = self.local
        if not local.stem:
            raise RuntimeError("a hypothetical state is only supported in STEM mode")
        rows, cols = trial.change.rows, trial.change.cols
        saved = (local.image[rows, cols].clone(), self.s_ref, self.ss_ref, self.s_keep, self.st_keep,
                 self.loss, self.n_atoms, local.accepted_since_refresh)
        self.commit(trial)
        try:
            yield
        finally:
            local.image[rows, cols] = saved[0]
            (self.s_ref, self.ss_ref, self.s_keep, self.st_keep, self.loss, self.n_atoms,
             local.accepted_since_refresh) = saved[1:]

    # ------------------------------------------------------------------ landscape
    @torch.no_grad()
    def insertion_gain(self, amplitude: float) -> Optional[np.ndarray]:
        """Eqs. (6.7): loss change of a new atom of amplitude ``amplitude`` at every pixel (``(H, W)``).

        ``inf`` where the loss is undefined; ``None`` if not applicable (TEM, no atoms).
        """

        local = self.local
        if not local.stem or self.n_atoms == 0:
            return None
        n, n_keep = self.n_ref, self.n_keep
        if n < 2 or n_keep < 1:
            return None
        pad, (H, W) = local.pad, tuple(local.sim.shape)
        size = (local.Nh, local.Nw)
        dtype = torch.float64
        kernel = local.probe_kernel().to(dtype)
        kernel_ft = torch.fft.rfft2(kernel).conj()
        kernel2_ft = torch.fft.rfft2(kernel * kernel).conj()
        s = local.image.to(dtype)

        def correlate(image, k_ft):
            padded = torch.zeros(size, dtype=dtype, device=self.device)
            padded[pad:pad + H, pad:pad + W] = image
            out = torch.fft.irfft2(torch.fft.rfft2(padded) * k_ft, s=size)
            return out[pad:pad + H, pad:pad + W].cpu().numpy()

        a_amp = float(amplitude)
        d_s_ref = a_amp * correlate(self.ref, kernel_ft)
        d_ss_ref = 2.0 * a_amp * correlate(self.ref * s, kernel_ft) + a_amp ** 2 * correlate(self.ref, kernel2_ft)
        d_s_keep = a_amp * correlate(self.keep, kernel_ft)
        d_st_keep = a_amp * correlate(self.keep * self.t, kernel_ft)

        mean_t = self.t_ref / n
        std_t = math.sqrt(max((self.tt_ref - n * mean_t * mean_t) / (n - 1.0), 0.0))
        mean_s = (self.s_ref + d_s_ref) / n
        var_s = (self.ss_ref + d_ss_ref - n * mean_s * mean_s) / (n - 1.0)
        std_s = np.sqrt(np.maximum(var_s, 0.0))
        a = mean_s + self.norm0 * std_s
        sigma_s = np.maximum(std_s * max(1.0 + self.norm1, 0.1), EPS)
        cross = (self.st_keep + d_st_keep) - a * self.t_keep - mean_t * (self.s_keep + d_s_keep) \
            + n_keep * a * mean_t
        loss = -cross / (n_keep * sigma_s * max(std_t, EPS))
        gain = (loss - self.loss).astype(np.float32)
        return np.where(np.isfinite(gain), gain, np.inf)
