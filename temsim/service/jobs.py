r"""Part IV, Section 3: jobs and their pacing.

Theory
======

Optimisation and seeding are long; a user wants to watch and to intervene.  Both
are therefore cut into *units of work* that are small compared with the time a
person needs to notice a delay:

* a training job advances one optimiser step (Part II, Eq. 5.1) per unit,
* a seeding job advances a *chunk* of trials (Part II, Section 6) per unit,
  followed by a tuning phase of a few optimiser steps every ``tune_every`` trials.

Between two units the server (Section 4) handles all waiting commands.  A command
therefore never sees a half-finished step, and a *pause* is simply "do not
advance".

Drawing a frame costs a full simulation and is much slower than a step.  Two
mechanisms keep that away from the optimisation.

* A training job can *keep* the images of its own forward pass
  (:meth:`TrainingJob.advance` with ``capture``); a frame made from them costs no simulation,
  and turning it into arrays and messages is left to another thread
  (:mod:`temsim.service.sender`).
* Frames that need a simulation of their own (seeding, the optional extras of the
  interface) are paced by the :class:`Clock`: after a frame that took
  :math:`c` seconds the next one is not before :math:`\max(T_{\rm min},\,\kappa c)`
  seconds, so that at most :math:`1/(1+\kappa)` of the time goes into drawing
  (``kappa = 3`` keeps it below 25 %).

The cheap *light* message (step, loss) is sent every unit.

Implementation
==============
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

from .session import Session, SessionError


class Clock:
    """Due times for frames and autosaves (Eq. above)."""

    def __init__(self, frame_interval: float = 1.5, kappa: float = 3.0, save_interval: float = 120.0,
                 now=time.monotonic):
        self.frame_interval, self.kappa, self.save_interval = frame_interval, kappa, save_interval
        self.now = now
        self.next_frame = 0.0
        #: a frame is wanted at once (the state was changed by hand), whatever kind of job runs
        self.urgent = True
        self.next_save = self.now() + save_interval if save_interval else math.inf

    def frame_due(self) -> bool:
        return self.now() >= self.next_frame

    def frame_done(self, cost: float) -> None:
        self.next_frame = self.now() + max(self.frame_interval, self.kappa * cost)
        self.urgent = False

    def frame_now(self) -> None:
        self.next_frame = 0.0
        self.urgent = True

    def save_due(self) -> bool:
        return self.now() >= self.next_save

    def save_done(self) -> None:
        self.next_save = self.now() + self.save_interval if self.save_interval else math.inf


class Job:
    """One unit of work per :meth:`advance`; ``done`` ends the job."""

    kind = "job"

    def __init__(self, session: Session):
        self.session = session
        self.done = False
        self.captured = None

    def advance(self, capture: bool = False) -> Dict[str, Any]:
        """One unit.  ``capture``: keep what the unit computed in ``self.captured`` (jobs that can)."""

        raise NotImplementedError

    def status(self) -> Dict[str, Any]:
        return {"kind": self.kind, "done": self.done}


class TrainingJob(Job):
    kind = "training"

    def __init__(self, session: Session, steps: Optional[int] = None):
        super().__init__(session)
        if session.trainer is None:
            raise SessionError("nothing is open")
        self.limit = None if steps is None else int(steps)
        self.count = 0
        self.last: Dict[str, Any] = {}
        #: atoms removed by the last automatic merge
        self.merged = 0

    def advance(self, capture: bool = False) -> Dict[str, Any]:
        every = int(self.session.require_model().settings.regularisation.merge_every)
        if every > 0 and self.count > 0 and self.count % every == 0:
            self.merged = self.session.edit("merge_close").get("merged", 0)
        result = self.session.trainer.step(keep=capture)
        if capture:
            self.captured = self.session.capture(result.kept, result)
        self.count += 1
        if self.limit is not None and self.count >= self.limit:
            self.done = True
        self.last = {"kind": self.kind, "step": result.step, "loss": result.loss, "data": result.data,
                     "prior": result.prior, "done": self.done}
        return self.last

    def status(self) -> Dict[str, Any]:
        return {"kind": self.kind, "done": self.done, "count": self.count, "limit": self.limit, **self.last}


class SeedingJob(Job):
    kind = "seeding"
    #: seconds after which a chunk ends early, so that commands (stop, pause) are not kept waiting
    BUDGET = 0.4

    def __init__(self, session: Session, layer=None, trials: Optional[int] = None, chunk: int = 20,
                 tune_every: int = 0, tune_steps: int = 0):
        super().__init__(session)
        self.seeder = session.make_seeder(layer)
        self.limit = None if trials is None else int(trials)
        self.chunk = max(1, int(chunk))
        self.tune_every, self.tune_steps = int(tune_every), int(tune_steps)
        self.count = 0
        self._since_tune = 0

    def advance(self, capture: bool = False) -> Dict[str, Any]:
        seeder = self.seeder
        if self.session.seeder is not seeder:
            raise SessionError("the seeding configuration changed; start seeding again")
        n = self.chunk if self.limit is None else min(self.chunk, self.limit - self.count)
        deadline = time.monotonic() + self.BUDGET
        for _ in range(n):
            if time.monotonic() > deadline:
                break
            seeder.trial()
            self.count += 1
            self._since_tune += 1
            if seeder.stats.full:
                self.done = True
                break
            if self.tune_every and self.tune_steps and self._since_tune >= self.tune_every:
                seeder.tune(self.tune_steps)
                self._since_tune = 0
        if self.limit is not None and self.count >= self.limit:
            self.done = True
        return {**self.session.seeding_status(), "kind": self.kind, "done": self.done, "count": self.count,
                "limit": self.limit}

    def status(self) -> Dict[str, Any]:
        return {**self.session.seeding_status(), "kind": self.kind, "done": self.done, "count": self.count,
                "limit": self.limit}
