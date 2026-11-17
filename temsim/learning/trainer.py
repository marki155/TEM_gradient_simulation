"""Part II, Section 5 (continued): the training loop.

One step evaluates the loss of Eq. (3.1), back-propagates it and lets the
optimiser of Eq. (5.1) move the enabled parameters.  The trainer is
synchronous and owns nothing but the optimiser and the history; who calls
:meth:`Trainer.step`, when, and from which thread is decided by the service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from .model import Model
from .optimiser import AdamGroups

#: per-atom tensors that stay put for atoms outside the loss window
FROZEN_FIELDS = ("shifts", "intensity", "logit")


@dataclass
class StepResult:
    step: int
    loss: float
    data: Dict[str, float] = field(default_factory=dict)
    prior: Dict[str, float] = field(default_factory=dict)
    #: standardised images of this step's forward pass (only if asked for, see :meth:`Trainer.step`)
    kept: Optional[Dict[str, object]] = None


class Trainer:
    def __init__(self, model: Model):
        self.model = model
        self.optimiser: Optional[AdamGroups] = None
        self.history: List[float] = []
        self.steps = 0

    def reset_history(self) -> None:
        self.history.clear()

    def prepare(self) -> bool:
        """Apply the switches and (re)build the optimiser if needed.  False: nothing is trainable."""

        m = self.model
        m.params.apply_flags(m.settings)
        if self.optimiser is None:
            self.optimiser = AdamGroups(m.params, m.settings)
        elif self.optimiser.stale():
            self.optimiser.rebuild()
        self.optimiser.sync_lrs()
        return not self.optimiser.empty

    def step(self, keep: bool = False) -> StepResult:
        """One optimiser step.  ``keep``: return the images of the forward pass in ``StepResult.kept``."""

        trainable = self.prepare()
        kept: Optional[Dict[str, object]] = {} if keep else None
        loss, data, prior = self.model.loss(kept)
        if trainable:
            frozen = self._outside_window()
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            self._restore(frozen)
        value = float(loss.detach())
        self.steps += 1
        self.history.append(value)
        return StepResult(self.steps, value, data, prior, kept)

    @torch.no_grad()
    def _outside_window(self) -> list:
        """Per layer: the atoms outside the loss window and their values, so that a step leaves them alone."""

        window = self.model.objective.window
        if window is None:
            return []
        r0, r1, c0, c1 = window
        params, frozen = self.model.params, []
        for i, lp in enumerate(params.layers):
            if len(lp) == 0:
                continue
            pos = params.positions(i)
            out = (pos[:, 0] < r0) | (pos[:, 0] >= r1) | (pos[:, 1] < c0) | (pos[:, 1] >= c1)
            if bool(out.any()):
                frozen.append((lp, out, {name: getattr(lp, name)[out].clone() for name in FROZEN_FIELDS}))
        return frozen

    @staticmethod
    @torch.no_grad()
    def _restore(frozen: list) -> None:
        for lp, out, values in frozen:
            for name, value in values.items():
                getattr(lp, name)[out] = value

    def run(self, n: int, callback: Optional[Callable[[StepResult], None]] = None,
            should_stop: Optional[Callable[[], bool]] = None) -> List[StepResult]:
        results = []
        for _ in range(n):
            if should_stop is not None and should_stop():
                break
            result = self.step()
            results.append(result)
            if callback is not None:
                callback(result)
        return results
