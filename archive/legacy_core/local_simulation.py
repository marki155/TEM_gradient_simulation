"""Lokale Neusimulation fuer das Atom-Seeding.

Ein Seeding-Versuch aendert nur wenige Atome. Das Bild aendert sich dadurch
nur in deren Umgebung (Reichweite = Faltungskern von Potential bzw. Sonde /
Kontrastuebertragung) - eine Simulation des ganzen Bildes ist Verschwendung.
Dieser Simulator haelt deshalb das aktuelle Bild im Speicher und rechnet pro
Versuch nur ein kleines Fenster um die geaenderten Atome neu:

* **STEM**: das Bild ist linear in der Atomdichte, ``dI = dO (*) Sonde``.
* **TEM** (eine Schicht): die Austrittswelle ``psi = exp(i sigma V)`` aendert sich
  nur dort, wo sich ``V`` aendert. Die CTF ist eine lineare Faltung der Welle
  mit ``h = ifft(ctf * apertur)``, also ``A' = A + h (*) dpsi`` - das Bild ist
  ``|A'|``.

Die Normierung der Loss (z-Normierung auf der Referenzmenge, dann negative
Korrelation auf der Wertungsmenge) laeuft ueber fortlaufende Summen, so dass
auch sie nur das Fenster anfasst. Der Faltungskern wird auf die Reichweite
beschnitten, in der ``tail`` seiner Energie (bzw. Masse) liegt; der dabei
entstehende kleine Fehler summiert sich ueber viele angenommene Versuche, daher
baut ``refresh()`` alles in regelmaessigen Abstaenden aus einer vollstaendigen
Simulation neu auf.

Unterstuetzt wird: eine Schicht, ein Element, Identitaets-Geometrie der Gruppe
(kein Shift/Rotation/Scale/Strain). In allen anderen Faellen ist ``ok`` falsch
und der Seeder simuliert wie bisher das ganze Bild.
"""

from __future__ import annotations

import contextlib
import functools
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def fast_length(n: int) -> int:
    """Kleinste gerade Zahl >= n mit nur den Primfaktoren 2, 3, 5 (schnelle FFT)."""

    n = max(int(n), 2)
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1 and n % 2 == 0:
            return n
        n += 1


@contextlib.contextmanager
def _without_default_device():
    """``torch.set_default_device`` haengt an jede Operation einen Python-Hook (~0.2 ms).

    Der lokale Simulator gibt jedes Geraet explizit an und kommt ohne aus.
    """

    current = torch.get_default_device()
    if current.type == "cpu":
        yield
        return
    torch.set_default_device(None)
    try:
        yield
    finally:
        torch.set_default_device(current)


def _plain(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with _without_default_device():
            return method(self, *args, **kwargs)
    return wrapper


@dataclass
class _Proposal:
    loss: float
    rows: slice
    cols: slice
    sim_new: torch.Tensor
    sums: np.ndarray
    amplitude_new: Optional[torch.Tensor] = None
    potential_delta: Optional[torch.Tensor] = None
    potential_window: Optional[tuple] = None


class LocalSimulator:
    """Haelt Bild und Loss-Summen und bewertet Aenderungen lokal.

    ``store`` ist der ``AtomStore`` der Schicht (liefert Positionen/Gewichte der
    zu entfernenden Atome und die Atomzahl).
    """

    #: Anteil der Kern-Energie (TEM) bzw. -Masse (STEM), der ausserhalb bleiben darf
    TAIL_TEM = 1e-4
    TAIL_STEM = 1e-2
    TAIL_POTENTIAL = 1e-4
    QUANT = 8

    def __init__(self, model, target, store, radius_px=None, refresh_every=300,
                 max_extent_px=None):
        self.model = model
        self.target = target
        self.store = store
        self.radius_override = radius_px
        self.refresh_every = int(refresh_every)
        self.max_extent_px = max_extent_px
        self.device = target.device
        self.ok = False
        self.reason = ""
        self.loss = math.inf
        self.accepted_since_refresh = 0
        self.radius = 0
        self.potential_radius = 0

        reason = self._unsupported()
        if reason:
            self.reason = reason
            return
        self.ok = True

    # -- Voraussetzungen ---------------------------------------------------------
    def _unsupported(self):
        model = self.model
        pot = model.potential_layer
        if pot.N_l != 1:
            return "mehr als eine Schicht"
        if pot.mixes_elements(0):
            return "Elementmischung"
        if not pot.active_layer_mask()[0]:
            return "Schicht traegt nicht zum Bild bei"
        self.stem = model.config.imaging_mode == "stem_haadf"
        if not self.stem and model.config.imaging_mode != "tem":
            return f"Abbildungsmodus {model.config.imaging_mode}"
        return ""

    # -- Aufbau --------------------------------------------------------------------
    @torch.no_grad()
    def refresh(self) -> float:
        """Alles aus einer vollstaendigen Simulation neu aufbauen; liefert die Loss."""

        if not self.ok:
            return math.inf
        model = self.model
        pot = model.potential_layer
        tp = pot.tunable_parameters
        self.pad = pot.potential_pad
        self.H, self.W = pot.H, pot.W
        self.Nh, self.Nw = self.H + 2 * self.pad, self.W + 2 * self.pad

        positions = pot.get_positions()[0]
        base = pot.get_init_positions(0) + tp.position_shifts[0]
        if positions.numel() and float((positions - base).abs().max()) > 1e-3:
            self.ok = False
            self.reason = "Gruppe ist verschoben/gedreht/verzerrt"
            return math.inf

        layer = pot.layers[0]
        group = layer.group
        self.contrast = float(tp.contrast_factors[group].detach())
        self._windows = {}
        self._potential_kernels = {}

        self._load_target_and_masks()
        if self.stem:
            self._refresh_stem()
        else:
            self._refresh_tem()
        self._load_normalisation()
        self._rebuild_sums()
        self.accepted_since_refresh = 0
        self.loss = self._loss_from_sums()
        return self.loss

    def _load_target_and_masks(self):
        model = self.model
        tm = model.train_methods
        target, mask = model.get_current_target_and_mask_img(self.target)
        active = tm.active_group
        ref = tm.mask_for(active) if active is not None else model._union_mask()

        def to_float(m):
            if m is None:
                return torch.ones((self.H, self.W), dtype=torch.float32, device=self.device)
            return (m.to(self.device) > 0).to(torch.float32)

        self.t = target.to(self.device).to(torch.float32)
        self.ref_mask = to_float(ref)
        self.keep_mask = to_float(mask)

    def _load_normalisation(self):
        values = self.model.tunable_parameters.normalisation_parameters.detach().cpu().tolist()
        self.norm0, self.norm1 = float(values[0]), float(values[1])

    def _rebuild_sums(self):
        r = self.ref_mask.double()
        k = self.keep_mask.double()
        t = self.t.double()
        s = self.sim.double()
        self.n_ref = float(r.sum())
        self.n_keep = float(k.sum())
        self.t_ref = float((r * t).sum())
        self.tt_ref = float((r * t * t).sum())
        self.t_keep = float((k * t).sum())
        self.s_ref = float((r * s).sum())
        self.ss_ref = float((r * s * s).sum())
        self.s_keep = float((k * s).sum())
        self.st_keep = float((k * s * t).sum())

    def _refresh_tem(self):
        model = self.model
        pot = model.potential_layer
        ctf_layer = model.imaging_layer
        multislice = model.model[1].phase_object_layer
        self.sigma = float(multislice.sigma)

        potential_list, _ = pot(None)
        self.potential = potential_list[0].detach().clone()
        wave = torch.exp(1j * self.sigma * self.potential)
        ctf, _env = ctf_layer._build_ctf()
        spectrum = (ctf * ctf_layer.aperture).detach()
        amplitude = torch.fft.ifft2(torch.fft.fft2(wave) * spectrum)
        pad = self.pad
        self.amplitude = amplitude[pad:pad + self.H, pad:pad + self.W].clone()
        self.sim = self.amplitude.abs()

        self.kernel = torch.fft.ifft2(spectrum)
        self.radius = self.radius_override or self._radius(self.kernel.abs() ** 2, self.TAIL_TEM)

        kirkland = pot.kirkland_precalc_pot
        element = pot.layers[0].element
        real = torch.fft.irfft2(kirkland.kernels_ft[element], s=(self.Nh, self.Nw))
        self.potential_kernel = real.detach()
        self.potential_radius = max(3, self._radius(real.abs(), self.TAIL_POTENTIAL))

    def _refresh_stem(self):
        model = self.model
        pot = model.potential_layer
        haadf = model.imaging_layer
        sim, _ = model.model(None)
        self.sim = sim.detach().clone()

        self.kernel = haadf.probe_intensity().detach()
        self.radius = self.radius_override or self._radius(self.kernel, self.TAIL_STEM)
        layer = pot.layers[0]
        cross = haadf.cross_sections[haadf._key(layer.element, layer.debye_waller_u)]
        self.stem_scale = self.contrast * float(cross)

    def _radius(self, weight, tail):
        """Kleinster Radius (Pixel), ausserhalb dessen hoechstens ``tail`` des Gewichts liegt."""

        n_h, n_w = weight.shape
        iy = (torch.arange(n_h, device=weight.device) + n_h // 2) % n_h - n_h // 2
        ix = (torch.arange(n_w, device=weight.device) + n_w // 2) % n_w - n_w // 2
        radius = torch.sqrt(iy[:, None].double() ** 2 + ix[None, :].double() ** 2)
        histogram = torch.bincount(radius.long().flatten(),
                                   weights=weight.double().flatten())
        cumulative = torch.cumsum(histogram, 0)
        inside = cumulative >= (1.0 - tail) * cumulative[-1]
        return int(torch.argmax(inside.to(torch.int64))) + 1

    # -- Loss aus den Summen -------------------------------------------------------
    def _loss_from_sums(self, sums=None):
        if self.store.n == 0:
            return math.inf
        s_ref, ss_ref, s_keep, st_keep = (
            (self.s_ref, self.ss_ref, self.s_keep, self.st_keep) if sums is None else sums)
        return self._ncc(s_ref, ss_ref, s_keep, st_keep)

    def _ncc(self, s_ref, ss_ref, s_keep, st_keep):
        n = self.n_ref
        if n < 2 or self.n_keep < 1:
            return math.inf
        eps = 1e-8
        mean_s = s_ref / n
        std_s = math.sqrt(max((ss_ref - n * mean_s * mean_s) / (n - 1.0), 0.0))
        mean_t = self.t_ref / n
        std_t = math.sqrt(max((self.tt_ref - n * mean_t * mean_t) / (n - 1.0), 0.0))
        a = mean_s + self.norm0 * std_s
        sigma_s = max(std_s * max(1.0 + self.norm1, 0.1), eps)
        sigma_t = max(std_t, eps)
        cross = st_keep - a * self.t_keep - mean_t * s_keep + self.n_keep * a * mean_t
        loss = -cross / (self.n_keep * sigma_s * sigma_t)
        return loss if math.isfinite(loss) else math.inf

    @property
    def needs_refresh(self):
        return self.refresh_every > 0 and self.accepted_since_refresh >= self.refresh_every

    # -- Fenster -------------------------------------------------------------------
    @staticmethod
    def _axis(center, size, n):
        """(Groesse, Ursprung) eines Fensters entlang einer Achse; ganze Achse, wenn zu gross."""

        if size >= n:
            return n, 0
        return size, int(min(max(center - size // 2, 0), n - size))

    def _kernel_window(self, key, size, build):
        cached = self._windows.get(key)
        if cached is None:
            cached = build(size)
            self._windows[key] = cached
        return cached

    def _crop_indices(self, radius, size, n):
        if size >= n:
            index = torch.arange(n, device=self.device)
            return index, index
        index = torch.arange(-radius, radius + 1, device=self.device)
        return index % n, index % size

    def _cropped(self, full, radius, size):
        """Kern um den Ursprung auf Fenstergroesse ``size`` beschneiden (Ursprung bleibt bei 0)."""

        out = torch.zeros(size, dtype=full.dtype, device=self.device)
        rows_src, rows_dst = self._crop_indices(radius, size[0], full.shape[0])
        cols_src, cols_dst = self._crop_indices(radius, size[1], full.shape[1])
        out[rows_dst[:, None], cols_dst[None, :]] = full[rows_src[:, None], cols_src[None, :]]
        return out

    def _imaging_kernel_ft(self, size):
        key = ("imaging", size)
        cached = self._windows.get(key)
        if cached is None:
            cropped = self._cropped(self.kernel, self.radius, size)
            cached = torch.fft.rfft2(cropped) if self.stem else torch.fft.fft2(cropped)
            self._windows[key] = cached
        return cached

    def _potential_kernel_ft(self, size):
        cached = self._potential_kernels.get(size)
        if cached is None:
            cropped = self._cropped(self.potential_kernel, self.potential_radius, size)
            cached = torch.fft.rfft2(cropped)
            self._potential_kernels[size] = cached
        return cached

    # -- Vorschlag -----------------------------------------------------------------
    @_plain
    @torch.no_grad()
    def propose(self, remove, add_xy, add_weight) -> Optional[_Proposal]:
        """Loss, wenn die Atome ``remove`` (Indizes im Store) wegfallen und ``add_xy`` dazukommen.

        ``None``: Aenderung zu gross fuer ein Fenster (der Aufrufer rechnet dann voll).
        """

        if not self.ok:
            return None
        store = self.store
        remove = np.asarray(remove, dtype=np.int64)
        add_xy = np.asarray(add_xy, dtype=np.float64).reshape(-1, 2)
        points = np.concatenate([store.positions[remove], add_xy])
        if len(points) == 0:
            return None
        n_after = store.n - len(remove) + len(add_xy)
        weights = np.concatenate([-store.weights[remove],
                                  np.broadcast_to(np.asarray(add_weight, dtype=np.float64), (len(add_xy),))])

        pad = self.pad
        shifted = points.astype(np.float32) + np.float32(pad)
        lo = np.floor(shifted.min(axis=0)).astype(np.int64)
        hi = np.floor(shifted.max(axis=0)).astype(np.int64) + 1
        center = (lo + hi) // 2
        extent = np.maximum(center - lo, hi - center)
        if self.max_extent_px is not None and extent.max() > self.max_extent_px:
            return None
        quantised = (np.ceil(extent / self.QUANT).astype(np.int64) + 0) * self.QUANT
        quantised = np.maximum(quantised, self.QUANT)

        n_dims = (self.Nh, self.Nw)
        reach = self.radius + (0 if self.stem else self.potential_radius)

        window = [self._axis(int(center[a]),
                             fast_length(2 * (int(quantised[a]) + reach) + 2), n_dims[a])
                  for a in range(2)]
        size = (window[0][0], window[1][0])
        origin = (window[0][1], window[1][1])

        positions = torch.as_tensor(shifted, device=self.device)
        weight_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        kernel_ft = self._imaging_kernel_ft(size)

        if self.stem:
            density = self._splat(size, origin, positions, weight_t)
            delta = torch.fft.irfft2(torch.fft.rfft2(density) * kernel_ft, s=size)
            delta = delta * self.stem_scale
            potential_delta = potential_window = None
        else:
            v_window = [self._axis(int(center[a]),
                                   fast_length(2 * (int(quantised[a]) + self.potential_radius) + 2),
                                   n_dims[a]) for a in range(2)]
            v_size = (v_window[0][0], v_window[1][0])
            v_origin = (v_window[0][1], v_window[1][1])

            density = self._splat(v_size, v_origin, positions, weight_t)
            potential_delta = torch.fft.irfft2(
                torch.fft.rfft2(density) * self._potential_kernel_ft(v_size),
                s=v_size) * self.contrast
            rows = slice(v_origin[0], v_origin[0] + v_size[0])
            cols = slice(v_origin[1], v_origin[1] + v_size[1])
            potential_old = self.potential[rows, cols]
            wave_delta = (torch.exp(1j * self.sigma * (potential_old + potential_delta))
                          - torch.exp(1j * self.sigma * potential_old))

            big = torch.zeros(size, dtype=torch.complex64, device=self.device)
            offset = (v_origin[0] - origin[0], v_origin[1] - origin[1])
            big[offset[0]:offset[0] + v_size[0], offset[1]:offset[1] + v_size[1]] = wave_delta
            delta = torch.fft.ifft2(torch.fft.fft2(big) * kernel_ft)
            potential_window = (rows, cols)

        # Bereich, in dem sich das Bild tatsaechlich aendern kann: Wirkungsradius um das
        # Zentrum, im Fenster, im Bild. Nicht-zyklische Fenster koennen am Rand
        # Wrap-Artefakte im Rest des Fensters haben - die werden hier abgeschnitten.
        support = int(max(extent)) + reach + 1
        bounds = []
        for a in range(2):
            low, high = origin[a], origin[a] + size[a]
            if size[a] < n_dims[a]:
                low = max(low, int(center[a]) - support)
                high = min(high, int(center[a]) + support + 1)
            image_n = (self.H, self.W)[a]
            low = max(low, pad)
            high = min(high, pad + image_n)
            bounds.append((low, high))
        if bounds[0][1] <= bounds[0][0] or bounds[1][1] <= bounds[1][0]:
            return None

        rows_image = slice(bounds[0][0] - pad, bounds[0][1] - pad)
        cols_image = slice(bounds[1][0] - pad, bounds[1][1] - pad)
        window_rows = slice(bounds[0][0] - origin[0], bounds[0][1] - origin[0])
        window_cols = slice(bounds[1][0] - origin[1], bounds[1][1] - origin[1])
        change = delta[window_rows, window_cols]

        sim_old = self.sim[rows_image, cols_image]
        if self.stem:
            amplitude_new = None
            sim_new = sim_old + change
        else:
            amplitude_new = self.amplitude[rows_image, cols_image] + change
            sim_new = amplitude_new.abs()

        difference = (sim_new - sim_old).double()
        ref = self.ref_mask[rows_image, cols_image].double()
        keep = self.keep_mask[rows_image, cols_image].double()
        target = self.t[rows_image, cols_image].double()
        sums = torch.stack([
            (ref * difference).sum(),
            (ref * difference * (sim_new.double() + sim_old.double())).sum(),
            (keep * difference).sum(),
            (keep * target * difference).sum(),
        ]).cpu().numpy()

        if n_after == 0:
            loss = math.inf
        else:
            loss = self._ncc(self.s_ref + sums[0], self.ss_ref + sums[1],
                             self.s_keep + sums[2], self.st_keep + sums[3])
        return _Proposal(loss=loss, rows=rows_image, cols=cols_image, sim_new=sim_new,
                         sums=sums, amplitude_new=amplitude_new,
                         potential_delta=potential_delta, potential_window=potential_window)

    @_plain
    @torch.no_grad()
    def insertion_gain(self, weight):
        """Loss-Landschaft fuers Einfuegen (nur STEM): Loss-Aenderung fuer ein neues Atom
        mit ``weight`` an jedem Bildpixel, Form (H, W), exakt wie ``propose`` (ohne Kollisionsregel).

        Das Bild ist linear in den Atomen: ein Atom bei ``p`` addiert
        ``d(x) = weight * stem_scale * kernel(x - p)``. Die Loss haengt nur von vier
        Summen ab (``s_ref``, ``ss_ref``, ``s_keep``, ``st_keep``), und deren Aenderungen
        sind Kreuzkorrelationen fester Karten mit dem Kern bzw. seinem Quadrat - fuer alle
        ``p`` zugleich per FFT: sum(ref*d), sum(ref*(2 s d + d^2)), sum(keep*d), sum(keep*t*d).
        Keine Simulation, nur ein paar FFTs. ``None``, wenn nicht anwendbar.
        """

        if not (self.ok and self.stem) or self.store.n == 0 or self.kernel.shape != (self.Nh, self.Nw):
            return None
        pad, H, W = self.pad, self.H, self.W
        dtype = torch.float64
        kernel = self._cropped(self.kernel, self.radius, (self.Nh, self.Nw)).to(dtype)
        kernel_ft = torch.fft.rfft2(kernel).conj()
        kernel2_ft = torch.fft.rfft2(kernel * kernel).conj()

        def correlate(image, k_ft):
            padded = torch.zeros((self.Nh, self.Nw), dtype=dtype, device=self.device)
            padded[pad:pad + H, pad:pad + W] = image.to(dtype)
            out = torch.fft.irfft2(torch.fft.rfft2(padded) * k_ft, s=(self.Nh, self.Nw))
            return out[pad:pad + H, pad:pad + W].cpu().numpy()

        amplitude = float(weight) * self.stem_scale
        ref, keep = self.ref_mask, self.keep_mask
        d_s_ref = amplitude * correlate(ref, kernel_ft)
        d_ss_ref = (2.0 * amplitude * correlate(ref * self.sim, kernel_ft)
                    + amplitude ** 2 * correlate(ref, kernel2_ft))
        d_s_keep = amplitude * correlate(keep, kernel_ft)
        d_st_keep = amplitude * correlate(keep * self.t, kernel_ft)

        n, n_keep, eps = self.n_ref, self.n_keep, 1e-8
        if n < 2 or n_keep < 1:
            return None
        mean_t = self.t_ref / n
        std_t = math.sqrt(max((self.tt_ref - n * mean_t * mean_t) / (n - 1.0), 0.0))
        s_ref = self.s_ref + d_s_ref
        mean_s = s_ref / n
        var_s = (self.ss_ref + d_ss_ref - n * mean_s * mean_s) / (n - 1.0)
        std_s = np.sqrt(np.maximum(var_s, 0.0))
        a = mean_s + self.norm0 * std_s
        sigma_s = np.maximum(std_s * max(1.0 + self.norm1, 0.1), eps)
        cross = ((self.st_keep + d_st_keep) - a * self.t_keep - mean_t * (self.s_keep + d_s_keep)
                 + n_keep * a * mean_t)
        loss = -cross / (n_keep * sigma_s * max(std_t, eps))
        gain = (loss - self.loss).astype(np.float32)
        return np.where(np.isfinite(gain), gain, np.inf)

    def _splat(self, size, origin, positions, weights):
        """Atome bilinear auf ein Fenster rastern (wie ``rasterize_atoms_bilinear``)."""

        rho = torch.zeros(size, dtype=torch.float32, device=self.device)
        x, y = positions[:, 0], positions[:, 1]
        ix = torch.floor(x).long()
        iy = torch.floor(y).long()
        fx, fy = x - ix, y - iy
        ix = ix - origin[0]
        iy = iy - origin[1]
        width = size[1]
        flat = ix * width + iy
        index = torch.cat([flat, flat + 1, flat + width, flat + width + 1])
        value = torch.cat([(1 - fx) * (1 - fy) * weights, (1 - fx) * fy * weights,
                           fx * (1 - fy) * weights, fx * fy * weights])
        rho.view(-1).index_add_(0, index, value)
        return rho

    @contextlib.contextmanager
    def hypothetical(self, proposal: _Proposal):
        """Zustand nach ``proposal`` nur voruebergehend annehmen (z.B. fuer ``insertion_gain``)."""

        saved = (self.sim[proposal.rows, proposal.cols].clone(), proposal.rows, proposal.cols,
                 self.s_ref, self.ss_ref, self.s_keep, self.st_keep, self.loss,
                 self.accepted_since_refresh)
        self.commit(proposal)
        try:
            yield
        finally:
            sim, rows, cols = saved[:3]
            self.sim[rows, cols] = sim
            (self.s_ref, self.ss_ref, self.s_keep, self.st_keep, self.loss,
             self.accepted_since_refresh) = saved[3:]

    @_plain
    @torch.no_grad()
    def commit(self, proposal: _Proposal):
        self.sim[proposal.rows, proposal.cols] = proposal.sim_new
        if proposal.amplitude_new is not None:
            self.amplitude[proposal.rows, proposal.cols] = proposal.amplitude_new
            rows, cols = proposal.potential_window
            self.potential[rows, cols] += proposal.potential_delta
        s_ref, ss_ref, s_keep, st_keep = proposal.sums
        self.s_ref += float(s_ref)
        self.ss_ref += float(ss_ref)
        self.s_keep += float(s_keep)
        self.st_keep += float(st_keep)
        self.accepted_since_refresh += 1
        self.loss = proposal.loss
