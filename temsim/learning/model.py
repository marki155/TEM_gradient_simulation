"""Part II, Section 4: the model, i.e. parameters, simulator and objective in one object.

The model is the unit that a reconstruction session owns.  It performs no
optimisation by itself; it only answers "what is the image and what is the
loss for the current parameters" (Eq. 3.1) and exposes the pieces that the
optimiser (Section 5) and the seeding (Section 6) need.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ..data.schema import Experiment, ModelState, TrainSettings
from ..physics.simulation import Simulator
from .objective import Objective, standardise
from .parameters import ParameterSet
from .priors import prior_loss


class Model:
    """Parameters, simulator, objective and settings of one reconstruction."""

    def __init__(self, experiment: Experiment, target: np.ndarray, settings: Optional[TrainSettings] = None,
                 state: Optional[ModelState] = None, device="cpu"):
        self.experiment = experiment
        self.device = device
        self.settings = settings if settings is not None else TrainSettings()
        self.settings.ensure_for(experiment)
        self.params = ParameterSet(experiment, state, device)
        self.simulator = Simulator(experiment.imaging, device)
        self.objective = Objective(target, self.settings, device)

    # ------------------------------------------------------------------ forward
    def layers(self):
        return self.params.effective_layers(self.settings.active_group)

    def image(self) -> torch.Tensor:
        """Simulated image for the current parameters."""

        return self.simulator.image(self.layers(), self.params.optics_state())

    def loss(self) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        """``(total, data terms, prior terms)``, Eq. (3.1)."""

        data, data_parts = self.objective(self.image(), self.params.normalisation)
        prior, prior_parts = prior_loss(self.params, self.settings)
        return data + prior, data_parts, prior_parts

    @torch.no_grad()
    def evaluate(self) -> Dict[str, object]:
        """Image, target, difference and loss values without gradient."""

        image = self.image()
        target, _, reference = self.objective.regions()
        data, data_parts = self.objective(image, self.params.normalisation)
        prior, prior_parts = prior_loss(self.params, self.settings)
        x, y = standardise(image, target, reference, self.params.normalisation)
        return {"image": image.cpu().numpy(), "target": target.cpu().numpy(),
                "standardised": x.cpu().numpy(), "standardised_target": y.cpu().numpy(),
                "diff": (x - y).cpu().numpy(),
                "loss": float(data + prior), "data": data_parts, "prior": prior_parts}

    # ------------------------------------------------------------------ export
    def state(self) -> ModelState:
        return self.params.to_state()
