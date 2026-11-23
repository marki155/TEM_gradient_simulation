r"""Part II, Section 8.3: training the detector on reconstructions.

Theory
======

A *sample* is one reconstruction seen as labelled data: the input
:math:`(y, y_I)` (target and second target, each standardised to mean 0 and standard
deviation 1 inside the valid region), the heat map :math:`h` of its atoms (Eq. 8.1)
and the valid region :math:`V` (the loss mask: outside it the fit says nothing).
Several samples, from several runs, can be trained together; one run is the special
case of one sample.

**Hold-out.**  With one image there is no second image to test on.  The bottom
``holdout`` fraction of the rows is therefore never trained on; the loss there (the
*validation loss*) and the detection score (F1 of the peaks of Section 8.1 against
the atoms, tolerance :math:`\sigma`-independent) show whether the network has learnt
what an atom looks like or only memorised the picture.

**Loss.**  Atoms cover a small part of the image; the squared error is weighted by
:math:`1 + w_{\rm fg}` where the label exceeds a tenth of its maximum,

.. math::

   \mathcal L = \frac{\sum_V w\,(\hat h - h)^2}{\sum_V w}.                       \tag{8.2}

**Augmentation.**  The lattice has no preferred orientation in the image: every crop
is turned by a random multiple of 90 degrees and possibly mirrored, the inputs are
scaled by a random factor near one and receive a little noise.

Implementation
==============
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import labels
from .unet import UNet, predict

FOREGROUND_WEIGHT = 5.0
MODEL_FILE, META_FILE = "detector.pt", "meta.json"


def standardise(image: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    values = image[valid] if valid is not None and valid.any() else image
    return ((image - float(values.mean())) / (float(values.std()) or 1.0)).astype(np.float32)


def inputs(target: np.ndarray, second: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """The network input ``(2, H, W)``."""

    return np.stack([standardise(target, valid), standardise(second, valid)])


@dataclasses.dataclass
class Sample:
    image: np.ndarray                  # (C, H, W) standardised
    label: np.ndarray                  # (H, W) heat map, Eq. (8.1)
    valid: np.ndarray                  # (H, W) bool
    positions: np.ndarray              # (n, 2) the atoms of the label
    split_row: int                     # rows >= split_row are held out
    name: str = ""
    tolerance_px: Optional[float] = None      # validation of this sample (None: the trainer's)
    min_distance_px: Optional[float] = None

    @property
    def shape(self):
        return self.label.shape


def validation_scales(positions: np.ndarray, pixel_A: float):
    """``(tolerance_px, min_distance_px)`` for the validation score of one image.

    0.5 and 0.8 Angstrom, but never more than half (0.7 times) the typical distance of neighbouring atoms,
    so that dense lattices and projected layers are not merged; at least one pixel.
    """

    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    spacing = math.inf
    if len(positions) > 1:
        spacing = float(np.percentile(cKDTree(positions).query(positions, k=2)[0][:, 1], 10))
    pixel_A = float(pixel_A) or 1.0
    return max(1.0, min(0.5 / pixel_A, 0.5 * spacing)), max(1.0, min(0.8 / pixel_A, 0.7 * spacing))


def make_sample(target, second, positions, intensities, scale: float, sigma_px: float,
                valid: Optional[np.ndarray] = None, holdout: float = 0.15, name: str = "") -> Sample:
    shape = np.asarray(target).shape
    valid = np.ones(shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    heights = np.asarray(intensities, dtype=np.float64).reshape(-1) / float(scale)
    label = labels.heatmap(shape, positions, heights, sigma_px)
    split = int(round(shape[0] * (1.0 - float(holdout)))) if holdout > 0 else shape[0]
    return Sample(inputs(target, second, valid), label, valid, positions, split, name)


@dataclasses.dataclass
class DetectorConfig:
    steps: int = 3000
    lr: float = 1e-3
    batch: int = 8
    patch: int = 128
    base: int = 32
    depth: int = 4
    val_every: int = 100
    noise: float = 0.05
    seed: Optional[int] = None


class DetectorTrainer:
    """Optimises a :class:`UNet` on random crops of the training part of ``samples`` (module docstring)."""

    def __init__(self, samples: Sequence[Sample], config: Optional[DetectorConfig] = None,
                 net: Optional[UNet] = None, device="cpu", tolerance_px: float = 2.0, threshold: float = 0.3,
                 min_distance_px: float = 2.0):
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("no samples")
        self.config = config or DetectorConfig()
        self.device = torch.device(device)
        channels = self.samples[0].image.shape[0]
        self.net = (net or UNet(channels, self.config.base, self.config.depth)).to(self.device)
        self.optimiser = torch.optim.AdamW(self.net.parameters(), lr=self.config.lr, weight_decay=1e-4)
        self.rng = np.random.default_rng(self.config.seed)
        self.generator = torch.Generator(device="cpu")
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            self.generator.manual_seed(self.config.seed)
        self.tolerance_px, self.threshold, self.min_distance_px = tolerance_px, threshold, min_distance_px
        self.steps = 0
        self.loss = math.nan
        self.val_loss: Optional[float] = None
        self.val_f1: Optional[float] = None
        self.history: List[Dict[str, Any]] = []
        multiple = self.net.multiple
        side = min(min(s.split_row, s.shape[1]) for s in self.samples)
        self.patch = max(multiple, min(int(self.config.patch), side) // multiple * multiple)
        if side < multiple:
            raise ValueError(f"the training part ({side} px) is smaller than {multiple} px")

    # ------------------------------------------------------------------ batches
    def _crop(self):
        sample = self.samples[int(self.rng.integers(len(self.samples)))]
        p = self.patch
        r = int(self.rng.integers(0, sample.split_row - p + 1))
        c = int(self.rng.integers(0, sample.shape[1] - p + 1))
        image = sample.image[:, r:r + p, c:c + p]
        label, valid = sample.label[r:r + p, c:c + p], sample.valid[r:r + p, c:c + p]
        k, flip = int(self.rng.integers(4)), bool(self.rng.integers(2))
        image, label, valid = (np.rot90(a, k, axes=(-2, -1)) for a in (image, label, valid))
        if flip:
            image, label, valid = (a[..., ::-1] for a in (image, label, valid))
        image = image * float(self.rng.uniform(0.9, 1.1)) + self.rng.normal(0.0, self.config.noise, image.shape)
        return image.astype(np.float32), np.ascontiguousarray(label), np.ascontiguousarray(valid)

    def _batch(self):
        parts = [self._crop() for _ in range(max(1, int(self.config.batch)))]
        to = lambda a, dtype: torch.as_tensor(np.stack(a), dtype=dtype, device=self.device)
        return (to([p[0] for p in parts], torch.float32), to([p[1] for p in parts], torch.float32),
                to([p[2] for p in parts], torch.float32))

    @staticmethod
    def weighted_loss(pred: torch.Tensor, label: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Eq. (8.2)."""

        top = label.amax().clamp_min(1e-6)
        weight = valid * (1.0 + FOREGROUND_WEIGHT * (label > 0.1 * top).float())
        return (weight * (pred - label) ** 2).sum() / weight.sum().clamp_min(1.0)

    # ------------------------------------------------------------------ steps
    def step(self) -> float:
        self.net.train()
        x, y, v = self._batch()
        loss = self.weighted_loss(self.net(x), y, v)
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        self.optimiser.step()
        self.steps += 1
        self.loss = float(loss)
        if self.config.val_every and self.steps % self.config.val_every == 0:
            self.validate()
        return self.loss

    def run(self, steps: int, budget_s: Optional[float] = None) -> int:
        """Up to ``steps`` steps, ending early after ``budget_s`` seconds; returns how many were made."""

        deadline = None if budget_s is None else time.monotonic() + budget_s
        done = 0
        while done < steps and (deadline is None or time.monotonic() < deadline):
            self.step()
            done += 1
        return done

    def validate(self) -> Optional[float]:
        """Loss (8.2) and F1 of the peaks on the held-out rows; ``None`` without hold-out."""

        losses, f1s = [], []
        for s in self.samples:
            if s.split_row >= s.shape[0]:
                continue
            # the strip with some context above it, so that its first rows do not see a border
            top = max(s.split_row - 32, 0)
            heat = predict(self.net, s.image[:, top:], self.device)[s.split_row - top:]
            label, valid = s.label[s.split_row:], s.valid[s.split_row:]
            pred = torch.as_tensor(heat)
            losses.append(float(self.weighted_loss(pred, torch.as_tensor(label), torch.as_tensor(valid, dtype=torch.float32))))
            found, _ = labels.find_peaks(heat, self.threshold, s.min_distance_px or self.min_distance_px, valid)
            truth = s.positions[s.positions[:, 0] >= s.split_row] - (s.split_row, 0)
            inside = valid[np.clip(truth[:, 0].astype(int), 0, valid.shape[0] - 1),
                           np.clip(truth[:, 1].astype(int), 0, valid.shape[1] - 1)] if len(truth) else np.zeros(0, bool)
            f1s.append(labels.scores(found, truth[inside], s.tolerance_px or self.tolerance_px)[2])
        if not losses:
            return None
        self.val_loss, self.val_f1 = float(np.mean(losses)), float(np.mean(f1s))
        self.history.append({"step": self.steps, "loss": self.loss, "val_loss": self.val_loss, "val_f1": self.val_f1})
        return self.val_loss


# ---------------------------------------------------------------------- files
def save_detector(directory: str, net: UNet, meta: Dict[str, Any]) -> str:
    os.makedirs(directory, exist_ok=True)
    torch.save({"config": net.config, "state": net.state_dict()}, os.path.join(directory, MODEL_FILE))
    with open(os.path.join(directory, META_FILE), "w") as f:
        json.dump(meta, f, indent=1)
    return directory


def load_detector(directory: str, device="cpu"):
    """``(net, meta)`` of a directory written by :func:`save_detector`."""

    blob = torch.load(os.path.join(directory, MODEL_FILE), map_location=device, weights_only=True)
    net = UNet(**blob["config"]).to(device)
    net.load_state_dict(blob["state"])
    return net, read_meta(directory)


def read_meta(directory: str) -> Dict[str, Any]:
    try:
        with open(os.path.join(directory, META_FILE)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def is_detector(directory: str) -> bool:
    return os.path.isfile(os.path.join(directory, MODEL_FILE))
