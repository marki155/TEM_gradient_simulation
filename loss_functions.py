import torch
import torch.nn as nn


class _BaseLocalWindowLoss(nn.Module):
    def __init__(self, win_size=8, reduction="mean", eps=1e-8):
        super().__init__()
        self.win_size = win_size
        self.reduction = reduction
        self.eps = eps

    def _prepare_inputs(self, x, y, mask=None):
        if x.shape != y.shape:
            raise ValueError(f"x and y must have same shape, got {x.shape} vs {y.shape}")
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensors [B,C,H,W], got ndim={x.ndim}")

        b, c, h, w = x.shape
        ws = self.win_size

        if h < ws or w < ws:
            raise ValueError(
                f"Image smaller than win_size: image=({h}, {w}), win_size={ws}"
            )

        h_crop = (h // ws) * ws
        w_crop = (w // ws) * ws

        x = x[:, :, :h_crop, :w_crop]
        y = y[:, :, :h_crop, :w_crop]

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            elif mask.ndim != 4:
                raise ValueError(f"mask must have ndim 2 or 4, got {mask.ndim}")

            if mask.shape[-2] < h_crop or mask.shape[-1] < w_crop:
                raise ValueError(
                    f"mask is smaller than required crop. Got {mask.shape}, "
                    f"need at least (..., ..., {h_crop}, {w_crop})"
                )

            mask = mask.to(device=x.device, dtype=x.dtype)
            mask = mask[:, :, :h_crop, :w_crop]

            try:
                mask = torch.broadcast_to(mask, x.shape)
            except RuntimeError as e:
                raise ValueError(
                    f"mask with shape {mask.shape} is not broadcastable to x shape {x.shape}"
                ) from e

        return x, y, mask, b, c, h_crop, w_crop, ws

    def _to_patches(self, t, b, c, h_crop, w_crop, ws):
        # (B,C,H,W) -> (B,C,nH,nW,ws*ws)
        t = t.view(b, c, h_crop // ws, ws, w_crop // ws, ws)
        t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
        t = t.view(b, c, h_crop // ws, w_crop // ws, ws * ws)
        return t

    def _reduce(self, patch_values, valid_patch_mask=None):
        if self.reduction == "none":
            return patch_values

        if self.reduction == "sum":
            if valid_patch_mask is None:
                return patch_values.sum()
            return (patch_values * valid_patch_mask).sum()

        if self.reduction == "mean":
            if valid_patch_mask is None:
                return patch_values.mean()
            denom = valid_patch_mask.sum().clamp_min(1.0)
            return (patch_values * valid_patch_mask).sum() / denom

        raise ValueError(f"Unknown reduction: {self.reduction}")

class _MaskedLossBase(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
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
                f"mask spatial shape must match x. Got mask {mask.shape}, x {x.shape}"
            )

        mask = mask.to(device=x.device, dtype=x.dtype)

        try:
            mask = torch.broadcast_to(mask, x.shape)
        except RuntimeError as e:
            raise ValueError(
                f"mask with shape {mask.shape} is not broadcastable to x shape {x.shape}"
            ) from e

        return mask

class MaskedMSELoss(_MaskedLossBase):
    def __init__(self, reduction: str = "mean", eps: float = 1e-8):
        super().__init__(eps=eps)
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        if x.shape != y.shape:
            raise ValueError(f"x and y must have same shape, got {x.shape} vs {y.shape}")
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensors [B,C,H,W], got ndim={x.ndim}")

        diff2 = (x - y) ** 2

        if mask is None:
            if self.reduction == "mean":
                return diff2.mean()
            if self.reduction == "sum":
                return diff2.sum()
            if self.reduction == "none":
                return diff2
            raise ValueError(f"Unknown reduction: {self.reduction}")

        mask = self._prepare_mask(mask, x)
        diff2 = diff2 * mask

        if self.reduction == "mean":
            return diff2.sum() / mask.sum().clamp_min(self.eps)
        if self.reduction == "sum":
            return diff2.sum()
        if self.reduction == "none":
            return diff2
        raise ValueError(f"Unknown reduction: {self.reduction}")

class MaskedL1Loss(_MaskedLossBase):
    def __init__(self, reduction: str = "mean", eps: float = 1e-8):
        super().__init__(eps=eps)
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        if x.shape != y.shape:
            raise ValueError(f"x and y must have same shape, got {x.shape} vs {y.shape}")
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensors [B,C,H,W], got ndim={x.ndim}")

        diff = torch.abs(x - y)

        if mask is None:
            if self.reduction == "mean":
                return diff.mean()
            if self.reduction == "sum":
                return diff.sum()
            if self.reduction == "none":
                return diff
            raise ValueError(f"Unknown reduction: {self.reduction}")

        mask = self._prepare_mask(mask, x)
        diff = diff * mask

        if self.reduction == "mean":
            return diff.sum() / mask.sum().clamp_min(self.eps)
        if self.reduction == "sum":
            return diff.sum()
        if self.reduction == "none":
            return diff
        raise ValueError(f"Unknown reduction: {self.reduction}")

class NCC(_MaskedLossBase):
    def __init__(self, eps: float = 1e-8, size_average: bool = True):
        super().__init__(eps=eps)
        self.size_average = size_average

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
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
            ).clamp_min(self.eps)

            ncc = numerator / denominator
        else:
            mask = self._prepare_mask(mask, x)

            x_flat = x.flatten(start_dim=1)
            y_flat = y.flatten(start_dim=1)
            m_flat = mask.flatten(start_dim=1)

            valid_count = m_flat.sum(dim=1, keepdim=True)
            valid_count_safe = valid_count.clamp_min(1.0)

            x_mean = (x_flat * m_flat).sum(dim=1, keepdim=True) / valid_count_safe
            y_mean = (y_flat * m_flat).sum(dim=1, keepdim=True) / valid_count_safe

            x_centered = (x_flat - x_mean) * m_flat
            y_centered = (y_flat - y_mean) * m_flat

            numerator = (x_centered * y_centered).sum(dim=1)
            denominator = torch.sqrt(
                (x_centered ** 2).sum(dim=1) * (y_centered ** 2).sum(dim=1)
            ).clamp_min(self.eps)

            ncc = numerator / denominator
            ncc = torch.where(valid_count.squeeze(1) > 0, ncc, torch.zeros_like(ncc))

        return ncc.mean() if self.size_average else ncc


class NCCLoss(nn.Module):
    def __init__(self, eps: float = 1e-8, size_average: bool = True):
        super().__init__()
        self.ncc = NCC(eps=eps, size_average=size_average)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        return 1.0 - self.ncc(x, y, mask=mask)


class LocalMSELoss(_BaseLocalWindowLoss):
    def forward(self, x, y, mask=None):
        x, y, mask, b, c, h_crop, w_crop, ws = self._prepare_inputs(x, y, mask)

        diff2 = (x - y) ** 2
        diff2 = self._to_patches(diff2, b, c, h_crop, w_crop, ws)

        if mask is None:
            patch_mse = diff2.mean(dim=-1)
            return self._reduce(patch_mse)

        m = self._to_patches(mask, b, c, h_crop, w_crop, ws)
        valid_count = m.sum(dim=-1)
        valid_count_safe = valid_count.clamp_min(1.0)

        patch_mse = (diff2 * m).sum(dim=-1) / valid_count_safe
        valid_patch_mask = (valid_count > 0).to(x.dtype)
        return self._reduce(patch_mse, valid_patch_mask)

class LocalL1Loss(_BaseLocalWindowLoss):
    def forward(self, x, y, mask=None):
        x, y, mask, b, c, h_crop, w_crop, ws = self._prepare_inputs(x, y, mask)

        diff = torch.abs(x - y)
        diff = self._to_patches(diff, b, c, h_crop, w_crop, ws)

        if mask is None:
            patch_l1 = diff.mean(dim=-1)
            return self._reduce(patch_l1)

        m = self._to_patches(mask, b, c, h_crop, w_crop, ws)
        valid_count = m.sum(dim=-1)
        valid_count_safe = valid_count.clamp_min(1.0)

        patch_l1 = (diff * m).sum(dim=-1) / valid_count_safe
        valid_patch_mask = (valid_count > 0).to(x.dtype)
        return self._reduce(patch_l1, valid_patch_mask)

class LocalDifferenceVarianceLoss(_BaseLocalWindowLoss):
    def __init__(self, win_size=8, reduction="mean", minimize=True, eps=1e-8):
        super().__init__(win_size=win_size, reduction=reduction, eps=eps)
        self.minimize = minimize

    def forward(self, x1, x2, mask: torch.Tensor = None):
        x1, x2, mask, b, c, h_crop, w_crop, ws = self._prepare_inputs(x1, x2, mask)

        diff = x1 - x2
        diff = self._to_patches(diff, b, c, h_crop, w_crop, ws)

        if mask is None:
            patch_var = diff.var(dim=-1, unbiased=False)
            loss = self._reduce(patch_var)
            return loss if self.minimize else -loss

        m = self._to_patches(mask, b, c, h_crop, w_crop, ws)
        valid_count = m.sum(dim=-1)
        valid_count_safe = valid_count.clamp_min(1.0)

        mean = (diff * m).sum(dim=-1, keepdim=True) / valid_count_safe.unsqueeze(-1)
        sq_dev = ((diff - mean) ** 2) * m
        patch_var = sq_dev.sum(dim=-1) / valid_count_safe

        valid_patch_mask = (valid_count > 0).to(diff.dtype)
        loss = self._reduce(patch_var, valid_patch_mask)
        return loss if self.minimize else -loss

class GlobalDifferenceVarianceLoss(_MaskedLossBase):
    def __init__(self, reduction="mean", minimize=True, eps=1e-8):
        """
        Varianz der Differenz zweier Bilder über das gesamte Bild.

        Args:
            reduction: "mean", "sum" oder "none"
            minimize:
                True  -> kleine Varianz belohnen
                False -> große Varianz belohnen
            eps: numerische Stabilität
        """
        super().__init__(eps=eps)
        self.reduction = reduction
        self.minimize = minimize

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, mask: torch.Tensor = None):
        if x1.shape != x2.shape:
            raise ValueError(f"x1 and x2 must have same shape, got {x1.shape} vs {x2.shape}")
        if x1.ndim != 4:
            raise ValueError(f"Expected 4D tensors [B,C,H,W], got ndim={x1.ndim}")

        diff = x1 - x2  # [B,C,H,W]

        if mask is None:
            diff_flat = diff.flatten(start_dim=1)  # [B, N]
            mean = diff_flat.mean(dim=1, keepdim=True)
            var = ((diff_flat - mean) ** 2).mean(dim=1)  # [B]
        else:
            mask = self._prepare_mask(mask, x1)
            diff_flat = diff.flatten(start_dim=1)        # [B, N]
            m_flat = mask.flatten(start_dim=1)           # [B, N]

            valid_count = m_flat.sum(dim=1, keepdim=True)           # [B,1]
            valid_count_safe = valid_count.clamp_min(1.0)

            mean = (diff_flat * m_flat).sum(dim=1, keepdim=True) / valid_count_safe
            sq_dev = ((diff_flat - mean) ** 2) * m_flat
            var = sq_dev.sum(dim=1) / valid_count_safe.squeeze(1)

            # Falls gar keine gültigen Pixel vorhanden sind
            var = torch.where(valid_count.squeeze(1) > 0, var, torch.zeros_like(var))

        if self.reduction == "mean":
            loss = var.mean()
        elif self.reduction == "sum":
            loss = var.sum()
        elif self.reduction == "none":
            loss = var
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        return loss if self.minimize else -loss

class LocalNCC(_BaseLocalWindowLoss):
    def __init__(self, win_size: int = 11, eps: float = 1e-8, size_average: bool = True):
        reduction = "mean" if size_average else "none"
        super().__init__(win_size=win_size, reduction=reduction, eps=eps)
        self.size_average = size_average

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        x, y, mask, b, c, h_crop, w_crop, ws = self._prepare_inputs(x, y, mask)

        # -> [B, C, nH, nW, ws*ws]
        x = self._to_patches(x, b, c, h_crop, w_crop, ws)
        y = self._to_patches(y, b, c, h_crop, w_crop, ws)

        if mask is None:
            x_mean = x.mean(dim=-1, keepdim=True)
            y_mean = y.mean(dim=-1, keepdim=True)

            x_centered = x - x_mean
            y_centered = y - y_mean

            numerator = (x_centered * y_centered).sum(dim=-1)
            denominator = torch.sqrt(
                (x_centered ** 2).sum(dim=-1) * (y_centered ** 2).sum(dim=-1)
            ).clamp_min(self.eps)

            patch_ncc = numerator / denominator

            if self.size_average:
                return patch_ncc.mean()
            return patch_ncc

        m = self._to_patches(mask, b, c, h_crop, w_crop, ws)

        valid_count = m.sum(dim=-1, keepdim=True)
        valid_count_safe = valid_count.clamp_min(1.0)

        x_mean = (x * m).sum(dim=-1, keepdim=True) / valid_count_safe
        y_mean = (y * m).sum(dim=-1, keepdim=True) / valid_count_safe

        x_centered = (x - x_mean) * m
        y_centered = (y - y_mean) * m

        numerator = (x_centered * y_centered).sum(dim=-1)
        denominator = torch.sqrt(
            (x_centered ** 2).sum(dim=-1) * (y_centered ** 2).sum(dim=-1)
        ).clamp_min(self.eps)

        patch_ncc = numerator / denominator

        valid_patch_mask = (valid_count.squeeze(-1) > 0)
        patch_ncc = torch.where(valid_patch_mask, patch_ncc, torch.zeros_like(patch_ncc))

        if self.size_average:
            valid_patch_mask_f = valid_patch_mask.to(patch_ncc.dtype)
            denom = valid_patch_mask_f.sum().clamp_min(1.0)
            return (patch_ncc * valid_patch_mask_f).sum() / denom

        return patch_ncc

class LocalNCCLoss(nn.Module):
    def __init__(self, win_size: int = 11, eps: float = 1e-8, size_average: bool = True):
        super().__init__()
        self.local_ncc = LocalNCC(win_size=win_size, eps=eps, size_average=size_average)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        return 1.0 - self.local_ncc(x, y, mask=mask)

class LocalSSIM(_BaseLocalWindowLoss):
    def __init__(
        self,
        win_size: int = 11,
        data_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
        eps: float = 1e-8,
        size_average: bool = True,
    ):
        reduction = "mean" if size_average else "none"
        super().__init__(win_size=win_size, reduction=reduction, eps=eps)
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.size_average = size_average

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        x, y, mask, b, c, h_crop, w_crop, ws = self._prepare_inputs(x, y, mask)

        # [B, C, nH, nW, P]
        x = self._to_patches(x, b, c, h_crop, w_crop, ws)
        y = self._to_patches(y, b, c, h_crop, w_crop, ws)

        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2

        if mask is None:
            mu_x = x.mean(dim=-1)
            mu_y = y.mean(dim=-1)

            mu_x2 = mu_x ** 2
            mu_y2 = mu_y ** 2
            mu_xy = mu_x * mu_y

            var_x = ((x - mu_x.unsqueeze(-1)) ** 2).mean(dim=-1)
            var_y = ((y - mu_y.unsqueeze(-1)) ** 2).mean(dim=-1)
            cov_xy = ((x - mu_x.unsqueeze(-1)) * (y - mu_y.unsqueeze(-1))).mean(dim=-1)

            ssim_patch = ((2 * mu_xy + c1) * (2 * cov_xy + c2)) / (
                (mu_x2 + mu_y2 + c1) * (var_x + var_y + c2) + self.eps
            )

            if self.size_average:
                return ssim_patch.mean()
            return ssim_patch

        m = self._to_patches(mask, b, c, h_crop, w_crop, ws)  # [B,C,nH,nW,P]
        valid_count = m.sum(dim=-1)                            # [B,C,nH,nW]
        valid_count_safe = valid_count.clamp_min(1.0)

        mu_x = (x * m).sum(dim=-1) / valid_count_safe
        mu_y = (y * m).sum(dim=-1) / valid_count_safe

        x_centered = (x - mu_x.unsqueeze(-1)) * m
        y_centered = (y - mu_y.unsqueeze(-1)) * m

        var_x = (x_centered ** 2).sum(dim=-1) / valid_count_safe
        var_y = (y_centered ** 2).sum(dim=-1) / valid_count_safe
        cov_xy = (x_centered * y_centered).sum(dim=-1) / valid_count_safe

        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        ssim_patch = ((2 * mu_xy + c1) * (2 * cov_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (var_x + var_y + c2) + self.eps
        )

        valid_patch_mask = (valid_count > 0)
        ssim_patch = torch.where(valid_patch_mask, ssim_patch, torch.zeros_like(ssim_patch))

        if self.size_average:
            valid_patch_mask_f = valid_patch_mask.to(ssim_patch.dtype)
            denom = valid_patch_mask_f.sum().clamp_min(1.0)
            return (ssim_patch * valid_patch_mask_f).sum() / denom

        return ssim_patch

class LocalSSIMLoss(nn.Module):
    def __init__(
        self,
        win_size: int = 11,
        data_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
        eps: float = 1e-8,
        size_average: bool = True,
    ):
        super().__init__()
        self.local_ssim = LocalSSIM(
            win_size=win_size,
            data_range=data_range,
            k1=k1,
            k2=k2,
            eps=eps,
            size_average=size_average,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        return 1.0 - self.local_ssim(x, y, mask=mask)


class GlobalSSIM(_MaskedLossBase):
    def __init__(
        self,
        data_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
        eps: float = 1e-8,
        size_average: bool = True,
    ):
        super().__init__(eps=eps)
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.size_average = size_average

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        """
        x, y: [B, C, H, W]
        mask: optional, gleiche Semantik wie bei den anderen Losses
        """
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensors [B,C,H,W], got ndim={x.ndim}")

        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2

        x_flat = x.flatten(start_dim=1)   # [B, N]
        y_flat = y.flatten(start_dim=1)   # [B, N]

        if mask is None:
            mu_x = x_flat.mean(dim=1, keepdim=True)
            mu_y = y_flat.mean(dim=1, keepdim=True)

            x_centered = x_flat - mu_x
            y_centered = y_flat - mu_y

            var_x = (x_centered ** 2).mean(dim=1)
            var_y = (y_centered ** 2).mean(dim=1)
            cov_xy = (x_centered * y_centered).mean(dim=1)

            mu_x = mu_x.squeeze(1)
            mu_y = mu_y.squeeze(1)
        else:
            mask = self._prepare_mask(mask, x)
            m_flat = mask.flatten(start_dim=1)  # [B, N]

            valid_count = m_flat.sum(dim=1, keepdim=True)   # [B,1]
            valid_count_safe = valid_count.clamp_min(1.0)

            mu_x = (x_flat * m_flat).sum(dim=1, keepdim=True) / valid_count_safe
            mu_y = (y_flat * m_flat).sum(dim=1, keepdim=True) / valid_count_safe

            x_centered = (x_flat - mu_x) * m_flat
            y_centered = (y_flat - mu_y) * m_flat

            var_x = (x_centered ** 2).sum(dim=1) / valid_count_safe.squeeze(1)
            var_y = (y_centered ** 2).sum(dim=1) / valid_count_safe.squeeze(1)
            cov_xy = (x_centered * y_centered).sum(dim=1) / valid_count_safe.squeeze(1)

            mu_x = mu_x.squeeze(1)
            mu_y = mu_y.squeeze(1)

            # Falls keine gültigen Pixel
            has_valid = valid_count.squeeze(1) > 0
            var_x = torch.where(has_valid, var_x, torch.zeros_like(var_x))
            var_y = torch.where(has_valid, var_y, torch.zeros_like(var_y))
            cov_xy = torch.where(has_valid, cov_xy, torch.zeros_like(cov_xy))
            mu_x = torch.where(has_valid, mu_x, torch.zeros_like(mu_x))
            mu_y = torch.where(has_valid, mu_y, torch.zeros_like(mu_y))

        ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
            (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2) + self.eps
        )

        if self.size_average:
            return ssim.mean()
        return ssim


class GlobalSSIMLoss(nn.Module):
    def __init__(
        self,
        data_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
        eps: float = 1e-8,
        size_average: bool = True,
    ):
        super().__init__()
        self.global_ssim = GlobalSSIM(
            data_range=data_range,
            k1=k1,
            k2=k2,
            eps=eps,
            size_average=size_average,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        return 1.0 - self.global_ssim(x, y, mask=mask)