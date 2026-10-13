r"""Part II, Section 6.3: where to try a new atom.

Three proposal distributions, all in image pixels ``(row, col)``:

*Uniform.*  A point drawn uniformly from the allowed region (the mask, shrunk by
a margin at the image border).  Unbiased, but wasteful once most of the image is
explained.

*Neighbour.*  Atoms in a 2D material sit at a roughly fixed bond length
:math:`b` from their neighbours.  The atoms are visited one after the other
(round robin) and a point is drawn around the current atom at a radius
:math:`\rho\sim\mathcal N(b,\sigma_b^2)`, truncated to :math:`[\rho_{\min},\rho_{\max}]`
(resampled up to 50 times, then clipped), and a uniform angle,

.. math::

    \mathbf p = \mathbf q + \rho(\cos\theta,\sin\theta).
    \qquad (6.8)

A flake grows from its edge instead of dissolving into random points.

*Landscape (STEM).*  Section 6.4 gives the loss change :math:`g(\mathbf p)` of a new
atom at every pixel.  Only pixels with :math:`g<0` that are further than
:math:`d_{\rm merge}` from every atom are admitted, and a pixel is drawn with the
Boltzmann weight

.. math::

    P(\mathbf p) \propto \exp\!\Big(-\frac{g(\mathbf p)}{T\,\max|g|}\Big),
    \qquad (6.9)

then jittered by :math:`\mathcal N(0, 0.25^2)` px (clipped to :math:`\pm\tfrac12`) to
reach sub-pixel positions.  :math:`T` is a temperature relative to the best
possible gain: :math:`T\to0` always takes the best pixel, :math:`T\to\infty`
any admissible pixel.  Together with the neighbour mode the pixels are restricted
to the ring around the atom, the scale :math:`\max|g|` stays that of the whole
image so that :math:`T` means the same everywhere.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


class Region:
    """Allowed area for new atoms: image minus margin, intersected with an optional mask."""

    def __init__(self, shape: Tuple[int, int], mask: Optional[np.ndarray] = None):
        self.shape = (int(shape[0]), int(shape[1]))
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)
        if self.mask is not None and self.mask.shape != self.shape:
            raise ValueError(f"region {self.mask.shape} does not match the image {self.shape}")
        self._pixels = {}

    def valid(self, margin: float) -> np.ndarray:
        height, width = self.shape
        m = int(math.ceil(margin))
        valid = np.zeros(self.shape, dtype=bool)
        valid[m:height - m, m:width - m] = True
        return valid if self.mask is None else valid & self.mask

    def check(self, margin: float) -> None:
        if self.mask is not None and not self.valid(margin).any():
            raise ValueError("the mask leaves no room for atoms")

    def inside(self, x: float, y: float, margin: float) -> bool:
        height, width = self.shape
        if not (margin <= x < height - margin and margin <= y < width - margin):
            return False
        return self.mask is None or bool(self.mask[int(x), int(y)])

    def uniform(self, rng: np.random.Generator, margin: float) -> Tuple[float, float]:
        height, width = self.shape
        if self.mask is not None:
            m = int(math.ceil(margin))
            if m not in self._pixels:
                self._pixels[m] = np.argwhere(self.valid(margin))
            pixels = self._pixels[m]
            row, col = pixels[rng.integers(len(pixels))]
            return row + rng.uniform(0.0, 1.0), col + rng.uniform(0.0, 1.0)
        return rng.uniform(margin, height - margin), rng.uniform(margin, width - margin)

    def jitter(self, rng: np.random.Generator, row: int, col: int, margin: float) -> Tuple[float, float]:
        height, width = self.shape
        x = row + float(np.clip(rng.normal(0.0, 0.25), -0.5, 0.5))
        y = col + float(np.clip(rng.normal(0.0, 0.25), -0.5, 0.5))
        return (min(max(x, margin), height - margin - 1e-6), min(max(y, margin), width - margin - 1e-6))


def distance_map(shape: Tuple[int, int], points: np.ndarray) -> np.ndarray:
    """Distance [px] of every pixel to the nearest of ``points`` (rounded to pixels)."""

    height, width = shape
    free = np.ones(shape, dtype=bool)
    if len(points):
        rounded = np.rint(np.asarray(points)).astype(int)
        free[np.clip(rounded[:, 0], 0, height - 1), np.clip(rounded[:, 1], 0, width - 1)] = False
    return distance_transform_edt(free)


class Landscape:
    """A map of loss changes (``inf`` = not admissible) to draw pixels from, Eq. (6.9)."""

    def __init__(self, gain: np.ndarray):
        self.gain = gain
        finite = np.isfinite(gain)
        self.admissible = bool(finite.any())
        self.scale = max(float(np.abs(gain[finite]).max()), 1e-30) if self.admissible else 1.0
        self._cdf = None
        self._cdf_temperature = None

    def pixel(self, rng: np.random.Generator, temperature: float) -> Optional[Tuple[int, int]]:
        if not self.admissible:
            return None
        if self._cdf is None or self._cdf_temperature != temperature:
            finite = np.isfinite(self.gain)
            score = np.where(finite, -self.gain / (self.scale * max(temperature, 1e-3)), -np.inf)
            score -= score[finite].max()
            self._cdf = np.cumsum(np.exp(score).ravel())
            self._cdf_temperature = temperature
        flat = int(np.searchsorted(self._cdf, rng.uniform(0.0, self._cdf[-1]), side="right"))
        return divmod(min(flat, self._cdf.size - 1), self.gain.shape[1])

    def ring_pixel(self, rng: np.random.Generator, base: np.ndarray, low: float, high: float,
                   temperature: float) -> Optional[Tuple[int, int]]:
        height, width = self.gain.shape
        reach = int(math.ceil(high))
        r0, r1 = max(int(base[0]) - reach, 0), min(int(base[0]) + reach + 1, height)
        c0, c1 = max(int(base[1]) - reach, 0), min(int(base[1]) + reach + 1, width)
        if r1 <= r0 or c1 <= c0:
            return None
        window = self.gain[r0:r1, c0:c1]
        rows, cols = np.mgrid[r0:r1, c0:c1]
        distance = np.hypot(rows - base[0], cols - base[1])
        window = np.where((distance >= low) & (distance <= high), window, np.inf)
        finite = np.isfinite(window)
        if not finite.any():
            return None
        score = -window / (self.scale * max(temperature, 1e-3))
        score -= score[finite].max()
        cdf = np.cumsum(np.where(finite, np.exp(score), 0.0).ravel())
        flat = min(int(np.searchsorted(cdf, rng.uniform(0.0, cdf[-1]), side="right")), cdf.size - 1)
        return r0 + flat // (c1 - c0), c0 + flat % (c1 - c0)

    def draw_in(self, rng: np.random.Generator, rows: np.ndarray, cols: np.ndarray, temperature: float):
        """Pixel among ``(rows, cols)`` with weight ``exp(-(g - min) / (T s))``; ``s`` is the spread of the candidates."""

        gain = self.gain[rows, cols]
        low = float(gain.min())
        spread = float(np.percentile(gain, 90)) - low if low >= 0.0 else -low
        weights = np.exp(-(gain - low) / (max(spread, 1e-12) * max(temperature, 1e-3)))
        cdf = np.cumsum(weights)
        k = min(int(np.searchsorted(cdf, rng.uniform(0.0, cdf[-1]), side="right")), len(cdf) - 1)
        return int(rows[k]), int(cols[k])


class NeighbourSampler:
    """Round-robin ring sampler of Eq. (6.8)."""

    def __init__(self):
        self.cursor = 0

    def _bases(self, positions: np.ndarray, exclude: Optional[Sequence[int]]):
        n = len(positions)
        skip = set(int(i) for i in exclude) if exclude is not None else set()
        for _ in range(n):
            index = self.cursor % n
            self.cursor = index + 1
            if index not in skip:
                yield positions[index]

    def point(self, rng, positions, region: Region, margin, bond, sigma, low, high,
              exclude=None, landscape: Optional[Landscape] = None, temperature: float = 0.1):
        if len(positions) == 0 or high <= low:
            return None
        if landscape is not None and not landscape.admissible:
            return None
        for base in self._bases(positions, exclude):
            if landscape is not None:
                pixel = landscape.ring_pixel(rng, base, low, high, temperature)
                if pixel is not None:
                    return region.jitter(rng, pixel[0], pixel[1], margin)
                continue
            for _ in range(20):
                for _ in range(50):
                    radius = rng.normal(bond, sigma)
                    if low <= radius <= high:
                        break
                else:
                    radius = min(max(bond, low), high)
                angle = rng.uniform(0.0, 2.0 * math.pi)
                x, y = base[0] + radius * math.cos(angle), base[1] + radius * math.sin(angle)
                if region.inside(x, y, margin):
                    return x, y
        return None
