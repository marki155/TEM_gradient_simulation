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


@dataclass
class StepResult:
    step: int
    loss: float
    data: Dict[str, float] = field(default_factory=dict)
    prior: Dict[str, float] = field(default_factory=dict)


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

    def step(self) -> StepResult:
        trainable = self.prepare()
        loss, data, prior = self.model.loss()
        if trainable:
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
        value = float(loss.detach())
        self.steps += 1
        self.history.append(value)
        return StepResult(self.steps, value, data, prior)

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
