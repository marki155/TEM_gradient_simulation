r"""Part II, Section 5: optimiser.

Theory
======

The parameters are updated with Adam [Kingma & Ba 2015].  With the gradient
:math:`g_t = \partial\mathcal L/\partial\theta` it keeps the running moments

.. math::

    m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\quad
    v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,\quad
    \theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
    \qquad (5.1)

with bias-corrected :math:`\hat m, \hat v`.  The step is of order :math:`\eta`
whatever the scale of the gradient.  This matters here because the unknowns have
wildly different units (pixels, radians, nanometres, logits); each *kind* of
parameter therefore gets its own learning rate :math:`\eta_k`, one Adam
parameter group per enabled key of :meth:`TrainSettings.params`.

The set of enabled keys and the tensors behind them change while the program
runs (the user switches parameters on and off, seeding appends atoms).  The
optimiser is therefore rebuilt whenever its *signature* (keys and tensor
identities) changes; the Adam moments of tensors that survive are carried over,
so that switching on a further parameter does not reset the others.  Learning
rates are re-read at every step.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from ..data.schema import TrainSettings
from .parameters import ParameterSet

Signature = Tuple


def signature(ps: ParameterSet, settings: TrainSettings) -> Signature:
    """Keys of the enabled parameters with identity and shape of their tensors."""

    out = []
    for key, param in settings.params().items():
        if not param.enabled:
            continue
        tensors = ps.tensors(key, settings)
        if tensors:
            out.append((key, tuple((id(t), tuple(t.shape)) for t in tensors)))
    return tuple(out)


class AdamGroups:
    """Adam with one parameter group per enabled key."""

    def __init__(self, ps: ParameterSet, settings: TrainSettings):
        self.ps, self.settings = ps, settings
        self.optimiser = None
        self._signature: Signature = ()
        self.rebuild()

    @property
    def empty(self) -> bool:
        return self.optimiser is None

    def stale(self) -> bool:
        return signature(self.ps, self.settings) != self._signature

    def rebuild(self) -> None:
        """Rebuild for the current switches, keeping the moments of unchanged tensors."""

        old = self.optimiser
        groups: List[dict] = []
        for key, param in self.settings.params().items():
            if param.enabled:
                tensors = self.ps.tensors(key, self.settings)
                if tensors:
                    groups.append({"params": tensors, "lr": float(param.lr), "name": key})
        self._signature = signature(self.ps, self.settings)
        if not groups:
            self.optimiser = None
            return
        new = torch.optim.Adam(groups)
        if old is not None:
            for group in new.param_groups:
                for t in group["params"]:
                    state = old.state.get(t)
                    if state and state["exp_avg"].shape == t.shape:
                        new.state[t] = state
        self.optimiser = new

    def sync_lrs(self) -> None:
        if self.optimiser is None:
            return
        params = self.settings.params()
        for group in self.optimiser.param_groups:
            group["lr"] = float(params[group["name"]].lr)

    def zero_grad(self) -> None:
        if self.optimiser is not None:
            self.optimiser.zero_grad(set_to_none=True)

    def step(self) -> None:
        if self.optimiser is not None:
            self.optimiser.step()
