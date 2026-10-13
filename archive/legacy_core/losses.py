import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalDifferenceVarianceLoss(nn.Module):
    def __init__(self, win_size=8, reduction="mean", minimize=True, eps=1e-8):
        """
        Varianz der Differenz zweier Bilder auf nicht überlappenden Patches.

        Args:
            win_size: Patchgröße (win_size x win_size)
            reduction: "mean", "sum" oder "none"
            minimize:
                True  -> kleine Varianz belohnen
                False -> große Varianz belohnen
            eps: numerische Stabilität
        """
        super().__init__()
        self.win_size = win_size
        self.reduction = reduction
        self.minimize = minimize
        self.eps = eps

    def _prepare_mask(self, mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Bringt mask auf Shape [B, C, H, W] und castet zu x.dtype.
        Erlaubte Eingaben:
          - [H, W]
          - [1, 1, H, W]
          - [B, 1, H, W]
          - [B, C, H, W]
        """
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim != 4:
            raise ValueError(f"mask must have ndim 2 or 4, got {mask.ndim}")

        if mask.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"mask spatial shape must match x1/x2. Got mask {mask.shape}, x {x.shape}"
            )

        mask = mask.to(device=x.device, dtype=x.dtype)

        try:
            mask = torch.broadcast_to(mask, x.shape)
        except RuntimeError as e:
            raise ValueError(
                f"mask with shape {mask.shape} is not broadcastable to x shape {x.shape}"
            ) from e

        return mask

    def forward(self, x1, x2, mask: torch.Tensor = None):
        """
        Args:
            x1, x2: Tensoren mit Shape (B, C, H, W)
            mask: gültige Pixel, 1=True/verwenden, 0=False/ignorieren

        Returns:
            Loss-Skalar oder Tensor mit Patch-Varianzen
        """
        if x1.shape != x2.shape:
            raise ValueError(
                f"x1 und x2 müssen dieselbe Shape haben, bekommen: "
                f"{x1.shape} vs {x2.shape}"
            )

        if x1.dim() != 4:
            raise ValueError(
                f"Erwartet Tensoren der Form (B, C, H, W), bekommen dim={x1.dim()}"
            )

        b, c, h, w = x1.shape
        ws = self.win_size

        if h < ws or w < ws:
            raise ValueError(
                f"Bild ist kleiner als win_size: Bild=({h}, {w}), win_size={ws}"
            )

        # Rand abschneiden, falls H oder W nicht durch win_size teilbar
        h_crop = (h // ws) * ws
        w_crop = (w // ws) * ws

        x1 = x1[:, :, :h_crop, :w_crop]
        x2 = x2[:, :, :h_crop, :w_crop]

        diff = x1 - x2

        if mask is None:
            # Originalverhalten
            diff = diff.view(b, c, h_crop // ws, ws, w_crop // ws, ws)
            diff = diff.permute(0, 1, 2, 4, 3, 5).contiguous()
            diff = diff.view(b, c, h_crop // ws, w_crop // ws, ws * ws)

            patch_var = diff.var(dim=-1, unbiased=False)
        else:
            # erst auf Original-Shape vorbereiten oder direkt passend croppen
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.ndim != 4:
                raise ValueError(f"mask must have ndim 2 or 4, got {mask.ndim}")

            # zuerst nur prüfen, dass die Original-Spatial-Dims passen
            if mask.shape[-2:] != (h, w):
                raise ValueError(
                    f"mask spatial shape must match original x1/x2 shape. "
                    f"Got mask {mask.shape}, expected (*, *, {h}, {w})"
                )

            mask = mask.to(device=x1.device, dtype=x1.dtype)

            # dann genauso croppen wie x1/x2
            mask = mask[:, :, :h_crop, :w_crop]

            # jetzt broadcast auf die gecroppte x1-Shape
            try:
                mask = torch.broadcast_to(mask, x1.shape)
            except RuntimeError as e:
                raise ValueError(
                    f"mask with shape {mask.shape} is not broadcastable to cropped x shape {x1.shape}"
                ) from e

            # In nicht überlappende Patches zerlegen
            diff = diff.view(b, c, h_crop // ws, ws, w_crop // ws, ws)
            diff = diff.permute(0, 1, 2, 4, 3, 5).contiguous()
            diff = diff.view(b, c, h_crop // ws, w_crop // ws, ws * ws)

            m = mask.view(b, c, h_crop // ws, ws, w_crop // ws, ws)
            m = m.permute(0, 1, 2, 4, 3, 5).contiguous()
            m = m.view(b, c, h_crop // ws, w_crop // ws, ws * ws)

            valid_count = m.sum(dim=-1)  # [B, C, nH, nW]
            valid_count_safe = valid_count.clamp_min(1.0)

            mean = (diff * m).sum(dim=-1) / valid_count_safe
            mean = mean.unsqueeze(-1)

            sq_dev = ((diff - mean) ** 2) * m
            patch_var = sq_dev.sum(dim=-1) / valid_count_safe

            # Leere Patches auf 0 setzen
            patch_var = torch.where(valid_count > 0, patch_var, torch.zeros_like(patch_var))

        if self.reduction == "mean":
            if mask is None:
                loss = patch_var.mean()
            else:
                # nur nicht-leere Patches mitteln
                valid_patch_mask = (valid_count > 0).to(patch_var.dtype)
                denom = valid_patch_mask.sum().clamp_min(1.0)
                loss = (patch_var * valid_patch_mask).sum() / denom

        elif self.reduction == "sum":
            if mask is None:
                loss = patch_var.sum()
            else:
                loss = patch_var.sum()

        elif self.reduction == "none":
            loss = patch_var

        else:
            raise ValueError(f"Unbekannte reduction: {self.reduction}")

        return loss if self.minimize else -loss


class RadialPowerSpectrumLoss(nn.Module):
    """
    Loss-Funktion basierend auf dem radialen Powerspektrum.

    Vergleicht die azimutal gemittelten Powerspektren P(q) von
    simuliertem und realem TEM-Bild.

    Args:
        log_scale:   Powerspektrum im log-Raum vergleichen (empfohlen für TEM)
        eps:         Numerische Stabilität im log
        weighting:   None | 'freq' (q-Gewichtung) | 'bragg' (mask-basiert)
        reduction:   'mean' | 'sum'
    """

    def __init__(
            self,
            log_scale: bool = True,
            eps: float = 1e-8,
            weighting: str = None,
            reduction: str = "mean",
    ):
        super().__init__()
        self.log_scale = log_scale
        self.eps = eps
        self.weighting = weighting
        self.reduction = reduction

    def _radial_power_spectrum(self, img: torch.Tensor) -> torch.Tensor:
        """
        Berechnet das radiale (azimutal gemittelte) Powerspektrum.

        Args:
            img: (B, 1, H, W) oder (B, H, W)
        Returns:
            P:   (B, n_bins) — Powerspektrum über Ortsfrequenzen q
        """
        if img.dim() == 3:
            img = img.unsqueeze(1)  # (B,1,H,W)

        B, C, H, W = img.shape
        assert C == 1, "Erwartet Graustufenbild (C=1)"

        # 2D FFT → Powerspektrum
        fft = torch.fft.fft2(img.squeeze(1))  # (B, H, W) komplex
        fft = torch.fft.fftshift(fft, dim=(-2, -1))  # DC in die Mitte
        power = fft.real ** 2 + fft.imag ** 2  # (B, H, W)

        # Radiale Koordinaten (einmal berechnen, auf device legen)
        cy, cx = H / 2.0, W / 2.0
        y = torch.arange(H, device=img.device, dtype=torch.float32) - cy
        x = torch.arange(W, device=img.device, dtype=torch.float32) - cx
        r = torch.sqrt(y[:, None] ** 2 + x[None, :] ** 2)  # (H, W)

        r_max = min(H, W) // 2
        n_bins = r_max

        # Azimutal mitteln: für jeden Ring r ∈ [k, k+1) den Mittelwert bilden
        P = torch.zeros(B, n_bins, device=img.device)
        for k in range(n_bins):
            mask = (r >= k) & (r < k + 1)  # (H, W) bool
            if mask.sum() == 0:
                continue
            P[:, k] = power[:, mask].mean(dim=-1)

        return P, torch.abs(fft)  # (B, n_bins)

    def _get_weights(self, n_bins: int, device: torch.device) -> torch.Tensor:
        """Optionale Frequenz-Gewichtung."""
        q = torch.arange(n_bins, device=device, dtype=torch.float32)
        if self.weighting == "freq":
            # Höhere Frequenzen stärker gewichten (schärft Gitterreflexe)
            w = q / (q.max() + 1e-8)
        elif self.weighting == "bragg":
            # Beispiel: Bragg-Peaks bei typischen TEM-Frequenzen betonen
            # → anpassen an konkrete q_hkl deines Materials!
            w = torch.ones(n_bins, device=device)
            peak_bins = [n_bins // 8, n_bins // 4, n_bins // 3]
            for p in peak_bins:
                idx = torch.arange(max(0, p - 2), min(n_bins, p + 3), device=device)
                w[idx] = 5.0
        else:
            w = torch.ones(n_bins, device=device)
        return w

    def forward(
            self,
            img_sim: torch.Tensor,
            img_real: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            img_sim:  (B, 1, H, W)  simuliertes TEM-Bild  [float32]
            img_real: (B, 1, H, W)  echtes TEM-Bild        [float32]
        Returns:
            loss: Skalar
        """
        #P_sim = self._radial_power_spectrum(img_sim)
        #P_real = self._radial_power_spectrum(img_real)

        diff = img_sim - img_real
        P_diff, fft = self._radial_power_spectrum(diff)


        if self.log_scale:
            P_diff = torch.log(P_diff + self.eps)
            #P_sim = torch.log(P_sim + self.eps)
            #P_real = torch.log(P_real + self.eps)

        # Frequenz-Gewichtung
        #w = self._get_weights(P_diff.shape[-1], P_diff.device)  # (n_bins,)
        #diff = P_diff * w.unsqueeze(0)  # (B, n_bins)
        fft = fft[:,130:170, 130:170]  # Nur bis Nyquist-Frequenz

        loss = fft.max() #if self.reduction == "mean" else diff.sum()
        return loss, fft

class NCC(nn.Module):
    def __init__(self, eps: float = 1e-8, size_average: bool = True):
        super().__init__()
        self.eps = eps
        self.size_average = size_average

    def _prepare_mask(self, mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Bringt mask auf Shape [B, C, H, W] und castet zu x.dtype.
        Erlaubte Eingaben:
          - [H, W]
          - [1, 1, H, W]
          - [B, 1, H, W]
          - [B, C, H, W]
        """
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        elif mask.ndim != 4:
            raise ValueError(f"mask must have ndim 2 or 4, got {mask.ndim}")

        if mask.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"mask spatial shape must match x/y. Got mask {mask.shape}, x {x.shape}"
            )

        mask = mask.to(device=x.device, dtype=x.dtype)

        # Broadcast auf [B, C, H, W]
        try:
            mask = torch.broadcast_to(mask, x.shape)
        except RuntimeError as e:
            raise ValueError(
                f"mask with shape {mask.shape} is not broadcastable to x shape {x.shape}"
            ) from e

        return mask

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        """
        x, y: [B, C, H, W]
        mask: gültige Pixel, 1=True/verwenden, 0=False/ignorieren
        returns normalized cross-correlation in [-1, 1]
        """
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensors [B,C,H,W], got ndim={x.ndim}")

        if mask is None:
            x_flat = x.flatten(start_dim=1)
            y_flat = y.flatten(start_dim=1)

            x_mean = x_flat.mean(dim=1, keepdim=True)
            y_mean = y_flat.mean(dim=1, keepdim=True)

            x_centered = x_flat - x_mean
            y_centered = y_flat - y_mean

            numerator = (x_centered * y_centered).sum(dim=1)
            denominator = torch.sqrt(
                (x_centered ** 2).sum(dim=1) * (y_centered ** 2).sum(dim=1)
            ) + self.eps

            ncc = numerator / denominator
        else:
            mask = self._prepare_mask(mask, x)

            x_flat = x.flatten(start_dim=1)          # [B, N]
            y_flat = y.flatten(start_dim=1)          # [B, N]
            m_flat = mask.flatten(start_dim=1)       # [B, N]

            valid_count = m_flat.sum(dim=1, keepdim=True)  # [B, 1]

            # Schutz gegen leere Masken
            valid_count_safe = valid_count.clamp_min(1.0)

            x_mean = (x_flat * m_flat).sum(dim=1, keepdim=True) / valid_count_safe
            y_mean = (y_flat * m_flat).sum(dim=1, keepdim=True) / valid_count_safe

            x_centered = (x_flat - x_mean) * m_flat
            y_centered = (y_flat - y_mean) * m_flat

            numerator = (x_centered * y_centered).sum(dim=1)
            denominator = torch.sqrt(
                (x_centered ** 2).sum(dim=1) * (y_centered ** 2).sum(dim=1)
            ) + self.eps

            ncc = numerator / denominator

            # Falls ein Sample gar keine gültigen Pixel hat -> NCC = 0
            has_valid = (valid_count.squeeze(1) > 0)
            ncc = torch.where(has_valid, ncc, torch.zeros_like(ncc))

        if self.size_average:
            return ncc.mean()
        else:
            return ncc

class NCCLoss(nn.Module):
    def __init__(self, eps: float = 1e-8, size_average: bool = True):
        super().__init__()
        self.ncc = NCC(eps=eps, size_average=size_average)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        # Maximieren von NCC = Minimieren von 1 - NCC
        return 1.0 - self.ncc(x, y, mask)


import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(window_size: int, sigma: float, device=None, dtype=None):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def create_2d_window(window_size: int, sigma: float, channels: int, device=None, dtype=None):
    g = gaussian_kernel(window_size, sigma, device=device, dtype=dtype)
    window_2d = torch.outer(g, g)
    window_2d = window_2d / window_2d.sum()
    window_2d = window_2d.view(1, 1, window_size, window_size)
    window_2d = window_2d.repeat(channels, 1, 1, 1)
    return window_2d


class SSIM(nn.Module):
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
        size_average: bool = True,
    ):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.size_average = size_average

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        x, y: shape [B, C, H, W]
        returns: scalar SSIM if size_average=True, else map mean per image
        """
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensors [B,C,H,W], got ndim={x.ndim}")

        b, c, h, w = x.shape
        window = create_2d_window(
            self.window_size,
            self.sigma,
            c,
            device=x.device,
            dtype=x.dtype,
        )

        padding = self.window_size // 2

        mu_x = F.conv2d(x, window, padding=padding, groups=c)
        mu_y = F.conv2d(y, window, padding=padding, groups=c)

        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=c) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=c) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=padding, groups=c) - mu_xy

        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        )

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(dim=(1, 2, 3))


class SSIMLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ssim = SSIM(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return 1.0 - self.ssim(x, y)

class LocalNCC(nn.Module):
    def __init__(self, window_size: int = 11, eps: float = 1e-8, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.eps = eps
        self.size_average = size_average

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        x, y: [B, C, H, W]
        """
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensors [B,C,H,W], got ndim={x.ndim}")

        b, c, h, w = x.shape
        padding = self.window_size // 2

        filt = torch.ones(
            (c, 1, self.window_size, self.window_size),
            device=x.device,
            dtype=x.dtype,
        )

        win_area = float(self.window_size * self.window_size)

        sum_x = F.conv2d(x, filt, padding=padding, groups=c)
        sum_y = F.conv2d(y, filt, padding=padding, groups=c)
        sum_x2 = F.conv2d(x * x, filt, padding=padding, groups=c)
        sum_y2 = F.conv2d(y * y, filt, padding=padding, groups=c)
        sum_xy = F.conv2d(x * y, filt, padding=padding, groups=c)

        mean_x = sum_x / win_area
        mean_y = sum_y / win_area

        var_x = sum_x2 / win_area - mean_x ** 2
        var_y = sum_y2 / win_area - mean_y ** 2
        cov_xy = sum_xy / win_area - mean_x * mean_y

        ncc_map = cov_xy / (torch.sqrt(var_x * var_y + self.eps))

        if self.size_average:
            return ncc_map.mean()
        else:
            return ncc_map.mean(dim=(1, 2, 3))


class LocalNCCLoss(nn.Module):
    def __init__(self, window_size: int = 11, eps: float = 1e-8, size_average: bool = True):
        super().__init__()
        self.local_ncc = LocalNCC(window_size=window_size, eps=eps, size_average=size_average)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return 1.0 - self.local_ncc(x, y)


class FFTBraggLoss(nn.Module):
    def __init__(self,
                 norm="ortho",
                 log_scale=True,
                 bragg_weight=2.0,
                 eps=1e-8):
        super().__init__()
        self.norm = norm
        self.log_scale = log_scale
        self.bragg_weight = bragg_weight
        self.eps = eps

    def forward(self, sim, target):
        if sim.ndim == 3:
            sim = sim.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        Fsim = torch.fft.fft2(sim, norm=self.norm)
        Ftar = torch.fft.fft2(target, norm=self.norm)

        # lineare Powerspektren
        Psim_lin = torch.abs(Fsim) ** 2
        Ptar_lin = torch.abs(Ftar) ** 2

        # Gewicht immer aus NICHTNEGATIVER linearer Power bauen
        norm_tar = Ptar_lin / (Ptar_lin.amax(dim=(-2, -1), keepdim=True) + self.eps)
        weight = 1.0 + self.bragg_weight * norm_tar

        # Fehlerraum optional logarithmisch
        if self.log_scale:
            Psim = torch.log1p(Psim_lin)
            Ptar = torch.log1p(Ptar_lin)
        else:
            Psim = Psim_lin
            Ptar = Ptar_lin

        loss = ((Psim - Ptar) ** 2 * weight).mean()
        return loss, weight