import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        x, y: [B, C, H, W]
        returns normalized cross-correlation in [-1, 1]
        """
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensors [B,C,H,W], got ndim={x.ndim}")

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

        if self.size_average:
            return ncc.mean()
        else:
            return ncc

class NCCLoss(nn.Module):
    def __init__(self, eps: float = 1e-8, size_average: bool = True):
        super().__init__()
        self.ncc = NCC(eps=eps, size_average=size_average)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # Maximieren von NCC = Minimieren von 1 - NCC
        return 1.0 - self.ncc(x, y)


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