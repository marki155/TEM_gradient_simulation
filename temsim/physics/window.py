r"""Part I, Section 10: local window update of the image.

Motivation
==========
Random seeding (Part II) tries thousands of small changes of the atom set.  A
change influences the image only within a finite neighbourhood, so re-simulating
the whole field of view is wasteful.  This section shows that, for the forward
model of Sections 6-9, the *change* of the image can be computed exactly (up to
the truncation of kernels) in a small window.

Linearity of the chain
----------------------
*STEM.*  The HAADF image Eq. (8.2) is linear in the atom densities,

.. math::

    \Delta I = |P|^2 * \sum_e \sigma_e\,c\,\Delta\rho_e .
    \qquad (10.1)

*TEM.*  Fix all layers but layer :math:`\ell`.  The multislice recursion Eq.
(6.2)-(6.3) is *linear in the wave* (the potentials only enter as multipliers),
and so is the lens, Eq. (7.1).  If the potential of layer :math:`\ell` changes by
:math:`\Delta V`, the wave behind that layer changes by

.. math::

    \delta_\ell = \big(\mathrm e^{\mathrm i\sigma(V+\Delta V)} - \mathrm e^{\mathrm i\sigma V}\big)\,\psi_\ell,
    \qquad (10.2)

where :math:`\psi_\ell` is the (stored) wave entering the layer, and the change
travels on *linearly* through the layers behind it,

.. math::

    \delta_{j+1}^{\mathrm{in}} = P(\Delta z_j)\,\delta_j,\qquad
    \delta_{j+1} = t_{j+1}\,\delta_{j+1}^{\mathrm{in}},\qquad
    \delta_{\mathrm{img}} = h * \delta_{\mathrm{exit}}
    \qquad (10.3)

(:math:`h = \mathcal F^{-1}[TA]` the lens impulse response).  All three
operators are convolutions with kernels of finite effective range: the
propagator spreads a perturbation over :math:`\lambda k_{\max}\Delta z`, the
potential over the atom radius, the lens over the width of :math:`h`.  Truncating
each kernel at the radius that contains all but a fraction ``tail`` of its energy
yields a window of size :math:`2(\text{extent} + R_V + \sum R_P + R_h)` in which
Eqs. (10.1)-(10.3) are evaluated with FFTs of that window only.  The new image is
the old image plus the window change; for amplitude signals :math:`|a+\delta|`,
for intensity :math:`|a+\delta|^2` with the complex amplitude :math:`a`.

Compared with earlier versions (one layer, one element, identity group geometry)
the restriction is lifted as far as linearity allows:

* any number of layers, because Eq. (10.3) carries the change through the stack
  (the stored wave of every layer behind the change is updated on commit);
* mixed layers: a change is given per element, each with its own kernel;
* group transformations and strain do not matter here: the *caller* passes the
  atoms' effective positions (before and after), the window only needs where
  density is added or removed.

Remaining approximations (all removed by ``refresh``): the median subtracted in
Eq. (6.5) is kept fixed, kernels are truncated, and changes beyond the support
radius are ignored.  A change that cannot be handled in a window (``propose``
returns ``None``) is simulated in full by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .simulation import OpticsState, Simulator
from .structure import AtomLayer, element_densities

TAIL_TEM = 1e-4
TAIL_STEM = 1e-2
TAIL_POTENTIAL = 1e-4
TAIL_PROPAGATOR = 1e-4
QUANT = 8


def fast_length(n: int) -> int:
    """Smallest even number >= n with only the prime factors 2, 3, 5 (fast FFT)."""

    n = max(int(n), 2)
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1 and n % 2 == 0:
            return n
        n += 1


def kernel_radius(weight: torch.Tensor, tail: float) -> int:
    """Smallest radius [px] outside which at most ``tail`` of the weight lies (kernel centred at [0, 0])."""

    n_h, n_w = weight.shape
    iy = (torch.arange(n_h, device=weight.device) + n_h // 2) % n_h - n_h // 2
    ix = (torch.arange(n_w, device=weight.device) + n_w // 2) % n_w - n_w // 2
    radius = torch.sqrt(iy[:, None].double() ** 2 + ix[None, :].double() ** 2)
    histogram = torch.bincount(radius.long().flatten(), weights=weight.double().flatten())
    cumulative = torch.cumsum(histogram, 0)
    inside = cumulative >= (1.0 - tail) * cumulative[-1]
    return int(torch.argmax(inside.to(torch.int64))) + 1


def _axis(center: int, size: int, n: int) -> Tuple[int, int]:
    """(size, origin) of a window along one axis; the whole axis if it does not fit."""

    if size >= n:
        return n, 0
    return size, int(min(max(center - size // 2, 0), n - size))


@dataclass
class ImageChange:
    """Result of :meth:`LocalImage.propose`: the image region that changes, old and new values."""

    rows: slice
    cols: slice
    old: torch.Tensor
    new: torch.Tensor
    # bookkeeping for commit
    layer: int = -1
    amplitude_new: Optional[torch.Tensor] = None
    core: Optional[Tuple[slice, slice]] = None
    potential_delta: Optional[torch.Tensor] = None
    wave_deltas: Dict[int, torch.Tensor] = field(default_factory=dict)
    origin: Tuple[int, int] = (0, 0)
    core_abs: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None


class LocalImage:
    """Current simulated image plus the intermediate waves needed for local updates."""

    def __init__(self, simulator: Simulator, layers: Sequence[AtomLayer], state: OpticsState,
                 refresh_every: int = 300, max_extent_px: Optional[int] = None, radius_px: Optional[int] = None):
        self.sim = simulator
        self.refresh_every = int(refresh_every)
        self.max_extent_px = max_extent_px
        self.radius_override = radius_px
        self.device = simulator.device
        self.accepted_since_refresh = 0
        self.stem = simulator.stem_mode
        self.refresh(layers, state)

    # -- full state ----------------------------------------------------------------
    @torch.no_grad()
    def refresh(self, layers: Sequence[AtomLayer], state: OpticsState) -> torch.Tensor:
        """Rebuild everything from a full simulation; returns the image."""

        sim = self.sim
        self.layers = list(layers)
        self.state = state
        self.pad = sim.padding
        self.H, self.W = sim.shape
        self.Nh, self.Nw = sim.padded_shape
        self._cache: Dict[tuple, torch.Tensor] = {}
        self._radii: Dict[tuple, int] = {}
        self.accepted_since_refresh = 0

        if self.stem:
            self.kernel = sim.probe_intensity(state).detach()
            self.radius = self.radius_override or self._radius(("probe",), self.kernel, TAIL_STEM)
            self.image = sim.image(layers, state).detach().clone()
            return self.image

        potentials = [None if p is None else p.detach() for p in sim.potentials(layers)]
        ms = sim.multislice
        self.order, self.distances, self.exit_distance = ms.plan(potentials, [l.z_nm for l in layers],
                                                                 sim.reverse)
        self.potential = potentials
        self.transmission: Dict[int, torch.Tensor] = {}
        self.wave_in: Dict[int, torch.Tensor] = {}
        wave = torch.ones(sim.padded_shape, dtype=torch.complex64, device=self.device)
        for k, idx in enumerate(self.order):
            self.wave_in[idx] = wave
            self.transmission[idx] = ms.transmission(potentials[idx])
            wave = wave * self.transmission[idx]
            if k < len(self.order) - 1:
                wave = ms.propagate(wave, self.distances[k])
        if self.order and self.exit_distance > 0:
            wave = ms.propagate(wave, self.exit_distance)
        self.spectrum = sim.lens_spectrum(state).detach()
        self.lens_kernel = torch.fft.ifft2(self.spectrum)
        self.lens_radius = self.radius_override or self._radius(("lens",), self.lens_kernel.abs() ** 2, TAIL_TEM)
        amplitude = torch.fft.ifft2(torch.fft.fft2(wave) * self.spectrum)
        p = self.pad
        self.amplitude = amplitude[p:p + self.H, p:p + self.W].clone()
        self.image = self._signal(self.amplitude)
        return self.image

    def _signal(self, amplitude: torch.Tensor) -> torch.Tensor:
        if self.sim.setup.tem_signal == "intensity":
            return amplitude.real ** 2 + amplitude.imag ** 2
        return amplitude.abs()

    @property
    def needs_refresh(self) -> bool:
        return self.refresh_every > 0 and self.accepted_since_refresh >= self.refresh_every

    # -- kernels -------------------------------------------------------------------
    def _radius(self, key, weight, tail) -> int:
        if key not in self._radii:
            self._radii[key] = kernel_radius(weight, tail)
        return self._radii[key]

    def _cropped(self, full: torch.Tensor, radius: int, size: Tuple[int, int]) -> torch.Tensor:
        """Kernel centred at the origin, cut to ``radius`` and placed in a ``size`` array (origin stays at 0)."""

        out = torch.zeros(size, dtype=full.dtype, device=self.device)
        index = []
        for axis in range(2):
            if size[axis] >= full.shape[axis]:
                idx = torch.arange(full.shape[axis], device=self.device)
                index.append((idx, idx))
            else:
                idx = torch.arange(-radius, radius + 1, device=self.device)
                index.append((idx % full.shape[axis], idx % size[axis]))
        (rs, rd), (cs, cd) = index
        out[rd[:, None], cd[None, :]] = full[rs[:, None], cs[None, :]]
        return out

    def probe_kernel(self) -> torch.Tensor:
        """STEM: probe intensity :math:`|P|^2` cut to its effective radius, on the padded grid, origin at ``[0, 0]``."""

        if not self.stem:
            raise RuntimeError("the probe kernel exists only in STEM mode")
        return self._cropped(self.kernel, self.radius, (self.Nh, self.Nw))

    def _kernel_ft(self, key, full, radius, size, real: bool):
        cache_key = (key, size, real)
        if cache_key not in self._cache:
            cropped = self._cropped(full, radius, size)
            self._cache[cache_key] = torch.fft.rfft2(cropped) if real else torch.fft.fft2(cropped)
        return self._cache[cache_key]

    def _potential_kernel(self, element: str, u_A: float):
        key = ("potential", element, round(float(u_A), 4))
        if key not in self._cache:
            spectrum = self.sim.bank.spectrum(element, float(u_A) * 0.1)
            real = torch.fft.irfft2(spectrum, s=(self.Nh, self.Nw)).detach()
            self._cache[key] = real
            self._radii[key] = max(3, kernel_radius(real.abs(), TAIL_POTENTIAL))
        return key, self._cache[key], self._radii[key]

    def _propagator_kernel(self, dz: float):
        key = ("propagator", round(float(dz), 9))
        if key not in self._cache:
            self._cache[key] = torch.fft.ifft2(self.sim.multislice.propagator(dz)).detach()
            self._radii[key] = kernel_radius(self._cache[key].abs() ** 2, TAIL_PROPAGATOR)
        return key, self._cache[key], self._radii[key]

    # -- helpers -------------------------------------------------------------------
    def _splat(self, size, origin, points, weights) -> torch.Tensor:
        rho = torch.zeros(size, dtype=torch.float32, device=self.device)
        x, y = points[:, 0], points[:, 1]
        ix, iy = torch.floor(x).long(), torch.floor(y).long()
        fx, fy = x - ix, y - iy
        ix, iy = ix - origin[0], iy - origin[1]
        width = size[1]
        flat = ix * width + iy
        index = torch.cat([flat, flat + 1, flat + width, flat + width + 1])
        value = torch.cat([(1 - fx) * (1 - fy) * weights, (1 - fx) * fy * weights,
                           fx * (1 - fy) * weights, fx * fy * weights])
        rho.view(-1).index_add_(0, index, value)
        return rho

    def _geometry(self, points: np.ndarray, reach: int):
        pad = self.pad
        shifted = points.astype(np.float32) + np.float32(pad)
        lo = np.floor(shifted.min(axis=0)).astype(np.int64)
        hi = np.floor(shifted.max(axis=0)).astype(np.int64) + 1
        center = (lo + hi) // 2
        extent = np.maximum(center - lo, hi - center)
        if self.max_extent_px is not None and extent.max() > self.max_extent_px:
            return None
        quantised = np.maximum(np.ceil(extent / QUANT).astype(np.int64) * QUANT, QUANT)
        n_dims = (self.Nh, self.Nw)
        window = [_axis(int(center[a]), fast_length(2 * (int(quantised[a]) + reach) + 2), n_dims[a])
                  for a in range(2)]
        size = (window[0][0], window[1][0])
        origin = (window[0][1], window[1][1])
        support = int(max(extent)) + reach + 1
        core = []
        for a in range(2):
            low, high = origin[a], origin[a] + size[a]
            if size[a] < n_dims[a]:
                low = max(low, int(center[a]) - support)
                high = min(high, int(center[a]) + support + 1)
            core.append((low, high))
        return shifted, size, origin, core

    def _gather(self, changes: Mapping[str, Tuple[np.ndarray, np.ndarray]]):
        parts = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p, _w in changes.values()]
        allpoints = np.concatenate(parts) if parts else np.zeros((0, 2))
        return allpoints

    # -- proposal --------------------------------------------------------------------
    @torch.no_grad()
    def propose(self, layer: int, changes: Mapping[str, Tuple[np.ndarray, np.ndarray]]) -> Optional[ImageChange]:
        """Image after adding/removing density in layer ``layer``.

        ``changes[element] = (points, weights)``: effective positions (unpadded pixel frame) and
        *signed* weights (removal negative).  ``None`` if the change needs more than a window.
        """

        allpoints = self._gather(changes)
        if len(allpoints) == 0:
            return None
        if self.stem:
            return self._propose_stem(layer, changes, allpoints)
        return self._propose_tem(layer, changes, allpoints)

    def _density(self, size, origin, shifted_by_element):
        return {el: self._splat(size, origin, torch.as_tensor(pts, device=self.device),
                                torch.as_tensor(w, dtype=torch.float32, device=self.device))
                for el, (pts, w) in shifted_by_element.items()}

    def _shifted(self, changes):
        pad = self.pad
        return {el: (np.asarray(p, dtype=np.float64).reshape(-1, 2).astype(np.float32) + np.float32(pad),
                     np.asarray(w, dtype=np.float64).reshape(-1)) for el, (p, w) in changes.items()}

    def _image_bounds(self, core):
        pad = self.pad
        bounds = []
        for a, n in enumerate((self.H, self.W)):
            low = max(core[a][0], pad)
            high = min(core[a][1], pad + n)
            bounds.append((low, high))
        if bounds[0][1] <= bounds[0][0] or bounds[1][1] <= bounds[1][0]:
            return None
        return bounds

    def _propose_stem(self, layer, changes, allpoints):
        reach = self.radius
        geometry = self._geometry(allpoints, reach)
        if geometry is None:
            return None
        _shifted_all, size, origin, core = geometry
        bounds = self._image_bounds(core)
        if bounds is None:
            return None
        spec = self.layers[layer]
        kernel_ft = self._kernel_ft(("probe",), self.kernel, self.radius, size, real=True)
        total = None
        densities = self._density(size, origin, self._shifted(changes))
        for element, rho in densities.items():
            part = rho * (spec.contrast * self.sim.cross_sections(element, spec.debye_waller_u_A))
            total = part if total is None else total + part
        delta = torch.fft.irfft2(torch.fft.rfft2(total) * kernel_ft, s=size)
        rows, cols, wr, wc = self._slices(bounds, origin)
        old = self.image[rows, cols]
        return ImageChange(rows=rows, cols=cols, old=old, new=old + delta[wr, wc], layer=layer)

    def _slices(self, bounds, origin):
        pad = self.pad
        rows = slice(bounds[0][0] - pad, bounds[0][1] - pad)
        cols = slice(bounds[1][0] - pad, bounds[1][1] - pad)
        wr = slice(bounds[0][0] - origin[0], bounds[0][1] - origin[0])
        wc = slice(bounds[1][0] - origin[1], bounds[1][1] - origin[1])
        return rows, cols, wr, wc

    def _propose_tem(self, layer, changes, allpoints):
        if layer not in self.transmission:
            return None
        spec = self.layers[layer]
        sim = self.sim
        position = self.order.index(layer)
        downstream = self.order[position + 1:]
        flights = self.distances[position:] + ([self.exit_distance] if self.exit_distance > 0 else [])

        kernels = {el: self._potential_kernel(el, spec.debye_waller_u_A) for el in changes}
        potential_radius = max(k[2] for k in kernels.values())
        propagators = [self._propagator_kernel(dz) for dz in flights]
        reach = potential_radius + sum(k[2] for k in propagators) + self.lens_radius
        geometry = self._geometry(allpoints, reach)
        if geometry is None:
            return None
        _s, size, origin, core = geometry
        bounds = self._image_bounds(core)
        if bounds is None:
            return None

        rows_c = slice(origin[0], origin[0] + size[0])
        cols_c = slice(origin[1], origin[1] + size[1])
        density = self._density(size, origin, self._shifted(changes))
        delta_v = None
        for element, rho in density.items():
            key, full, radius = kernels[element]
            part = torch.fft.irfft2(torch.fft.rfft2(rho) * self._kernel_ft(key, full, radius, size, True), s=size)
            delta_v = part if delta_v is None else delta_v + part
        delta_v = delta_v * spec.contrast

        sigma = sim.multislice.sigma
        v_old = self.potential[layer][rows_c, cols_c]
        t_old = self.transmission[layer][rows_c, cols_c]
        cur = (torch.exp(1j * sigma * (v_old + delta_v)) - t_old) * self.wave_in[layer][rows_c, cols_c]

        wave_deltas: Dict[int, torch.Tensor] = {}
        flight = iter(zip(flights, propagators))
        for j in downstream:
            _dz, (key, full, radius) = next(flight)
            cur = torch.fft.ifft2(torch.fft.fft2(cur) * self._kernel_ft(key, full, radius, size, False))
            wave_deltas[j] = cur
            cur = cur * self.transmission[j][rows_c, cols_c]
        if self.exit_distance > 0:
            _dz, (key, full, radius) = next(flight)
            cur = torch.fft.ifft2(torch.fft.fft2(cur) * self._kernel_ft(key, full, radius, size, False))
        lens_ft = self._kernel_ft(("lens",), self.lens_kernel, self.lens_radius, size, False)
        d_amp = torch.fft.ifft2(torch.fft.fft2(cur) * lens_ft)

        rows, cols, wr, wc = self._slices(bounds, origin)
        amplitude_new = self.amplitude[rows, cols] + d_amp[wr, wc]
        old = self.image[rows, cols]
        new = self._signal(amplitude_new)
        core_window = (slice(core[0][0] - origin[0], core[0][1] - origin[0]),
                       slice(core[1][0] - origin[1], core[1][1] - origin[1]))
        change = ImageChange(rows=rows, cols=cols, old=old, new=new, layer=layer, amplitude_new=amplitude_new,
                             core=core_window)
        change.potential_delta = delta_v
        change.wave_deltas = wave_deltas
        change.origin, change.core_abs = origin, core
        return change

    # -- commit ----------------------------------------------------------------------
    @torch.no_grad()
    def commit(self, change: ImageChange) -> None:
        self.image[change.rows, change.cols] = change.new
        if change.amplitude_new is not None:
            self.amplitude[change.rows, change.cols] = change.amplitude_new
            core = change.core_abs
            rows = slice(core[0][0], core[0][1])
            cols = slice(core[1][0], core[1][1])
            win = change.core
            self.potential[change.layer][rows, cols] += change.potential_delta[win]
            self.transmission[change.layer][rows, cols] = torch.exp(
                1j * self.sim.multislice.sigma * self.potential[change.layer][rows, cols])
            for j, delta in change.wave_deltas.items():
                self.wave_in[j][rows, cols] += delta[win]
        self.accepted_since_refresh += 1
