import torch
import torch.nn as nn
import torch.nn.functional as F


class SplineDisplacementField2D(nn.Module):
    """
    Glattes 2D-Verschiebungsfeld u(x,y), parametrisiert über ein grobes
    Kontrollgitter und bilineare Interpolation.

    Koordinaten sind hier in Pixeln.
    """

    def __init__(
        self,
        image_shape,
        control_shape=(8, 8),
        init_scale=0.0,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()

        self.H, self.W = image_shape
        self.control_shape = control_shape
        self.device = device
        self.dtype = dtype

        Hc, Wc = control_shape

        control = torch.zeros((1, 2, Hc, Wc), dtype=dtype, device=device)
        if init_scale > 0:
            control = control + init_scale * torch.randn_like(control)

        self.control = nn.Parameter(control)

    def normalize_coords(self, coords_xy):
        """
        coords_xy: (N, 2) in pixel coordinates
        output: (N, 2) in [-1, 1]
        """
        x = coords_xy[:, 0]
        y = coords_xy[:, 1]

        x_norm = 2.0 * x / max(self.W - 1, 1) - 1.0
        y_norm = 2.0 * y / max(self.H - 1, 1) - 1.0

        return torch.stack([x_norm, y_norm], dim=-1)

    def sample(self, coords_xy):
        """
        coords_xy: (N, 2) in pixel coordinates
        returns u: (N, 2)
        """
        grid = self.normalize_coords(coords_xy).view(1, -1, 1, 2)

        sampled = F.grid_sample(
            self.control,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        # (1, 2, N, 1) -> (N, 2)
        u = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
        return u

    def forward(self, coords_xy):
        return coords_xy + self.sample(coords_xy)

    def jacobian(self, coords_xy):
        """
        J[n, i, j] = d u_i / d x_j
        coords_xy must be shape (N, 2)
        """
        if not coords_xy.requires_grad:
            coords_xy = coords_xy.clone().detach().requires_grad_(True)

        u = self.sample(coords_xy)  # (N, 2)

        rows = []
        for i in range(2):
            grad_i = torch.autograd.grad(
                u[:, i].sum(),
                coords_xy,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            rows.append(grad_i)

        J = torch.stack(rows, dim=1)  # (N, 2, 2)
        return J

    def strain_tensor(self, coords_xy):
        J = self.jacobian(coords_xy)
        return 0.5 * (J + J.transpose(-1, -2))

    def rotation_tensor(self, coords_xy):
        J = self.jacobian(coords_xy)
        return 0.5 * (J - J.transpose(-1, -2))

    def elastic_energy_density(self, coords_xy, lambda_, mu):
        """
        lineare isotrope 2D-Elastizität
        """
        eps = self.strain_tensor(coords_xy)
        tr_eps = eps[:, 0, 0] + eps[:, 1, 1]
        frob_sq = (eps ** 2).sum(dim=(-2, -1))
        W = 0.5 * lambda_ * tr_eps**2 + mu * frob_sq
        return W

    def smoothness_loss(self):
        """
        Numerische Regularisierung auf dem Kontrollgitter
        """
        u = self.control
        dx = u[:, :, :, 1:] - u[:, :, :, :-1]
        dy = u[:, :, 1:, :] - u[:, :, :-1, :]
        return (dx**2).mean() + (dy**2).mean()

    def make_coordinate_grid(self, H=None, W=None, pixel_centers=True):
        H = self.H if H is None else H
        W = self.W if W is None else W

        if pixel_centers:
            xs = torch.arange(W, device=self.device, dtype=self.dtype) + 0.5
            ys = torch.arange(H, device=self.device, dtype=self.dtype) + 0.5
        else:
            xs = torch.linspace(0, self.W - 1, W, device=self.device, dtype=self.dtype)
            ys = torch.linspace(0, self.H - 1, H, device=self.device, dtype=self.dtype)

        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # [x, y]
        return coords

    def strain_maps(self, H=None, W=None, margin=0.0, lambda_=None, mu=None):
        """
        Berechnet Karten des Verschiebungs- und Strainfelds auf einem
        regulären Gitter.

        Returns dict mit 2D-Tensoren der Form (H, W):
            ux, uy,
            eps_xx, eps_yy, eps_xy,
            rot,
            div,
            energy (optional)
        """
        H = self.H if H is None else H
        W = self.W if W is None else W

        coords = self.make_coordinate_grid(H=H, W=W)
        coords = coords.clone().detach().requires_grad_(True)

        u = self.sample(coords)                    # (N, 2)
        eps = self.strain_tensor(coords)           # (N, 2, 2)
        omega = self.rotation_tensor(coords)       # (N, 2, 2)

        ux = u[:, 0].reshape(H, W)
        uy = u[:, 1].reshape(H, W)

        eps_xx = eps[:, 0, 0].reshape(H, W)
        eps_yy = eps[:, 1, 1].reshape(H, W)
        eps_xy = eps[:, 0, 1].reshape(H, W)

        # 2D-Rotation als Skalar
        rot = omega[:, 1, 0].reshape(H, W)

        # Divergenz = Spur von grad(u) = eps_xx + eps_yy
        div = (eps[:, 0, 0] + eps[:, 1, 1]).reshape(H, W)

        out = {
            "ux": ux.detach().cpu().numpy(),
            "uy": uy.detach().cpu().numpy(),
            "eps_xx": eps_xx.detach().cpu().numpy(),
            "eps_yy": eps_yy.detach().cpu().numpy(),
            "eps_xy": eps_xy.detach().cpu().numpy(),
            "rot": rot.detach().cpu().numpy(),
            "div": div.detach().cpu().numpy(),
        }

        if lambda_ is not None and mu is not None:
            energy = self.elastic_energy_density(coords, lambda_=lambda_, mu=mu)
            out["energy"] = energy.reshape(H, W)

        return out

    def smooth_map(self, m):
        return F.avg_pool2d(m.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1)[0, 0]

    def self_test(self, H=None, W=None, device=None, dtype=None):
        """
        Führt einen einfachen internen Konsistenztest aus.

        Testidee:
        1. Setzt ein analytisch einfaches Feld auf das Kontrollgitter:
               u_x = a * x
               u_y = b * y
           Erwartung:
               eps_xx ~ a
               eps_yy ~ b
               eps_xy ~ 0
               rot    ~ 0

        2. Berechnet daraus strain maps.
        3. Gibt Kennzahlen und Maps zurück.

        Returns:
            result: dict mit
                passed: bool
                stats: dict
                maps: dict[str, torch.Tensor]
        """
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        H = self.H if H is None else H
        W = self.W if W is None else W

        # Kleine Testdehnungen
        a = 0.01
        b = -0.02

        # Alten Zustand sichern
        old_control = self.control.detach().clone()

        try:
            with torch.no_grad():
                Hc, Wc = self.control_shape

                # Kontrollgitter in Pixelkoordinaten
                xs = torch.linspace(0, self.W - 1, Wc, device=device, dtype=dtype)
                ys = torch.linspace(0, self.H - 1, Hc, device=device, dtype=dtype)
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")

                # Erwartetes lineares Feld:
                # u_x = a * x
                # u_y = b * y
                ux = a * xx
                uy = b * yy

                self.control.zero_()
                self.control[0, 0] = ux
                self.control[0, 1] = uy

            maps = self.strain_maps(H=H, W=W)

            # Innenbereich auswerten, damit Randartefakte nicht dominieren
            pad = max(2, min(H, W) // 16)
            sl = (slice(pad, H - pad), slice(pad, W - pad))

            eps_xx_mean = maps["eps_xx"][sl].mean().item()
            eps_yy_mean = maps["eps_yy"][sl].mean().item()
            eps_xy_mean = maps["eps_xy"][sl].mean().item()
            rot_mean = maps["rot"][sl].mean().item()

            eps_xx_std = maps["eps_xx"][sl].std().item()
            eps_yy_std = maps["eps_yy"][sl].std().item()
            eps_xy_std = maps["eps_xy"][sl].std().item()
            rot_std = maps["rot"][sl].std().item()

            # Toleranzen bewusst nicht zu hart, weil bilineare Interpolation + Rastereffekte
            tol_mean = 5e-3
            tol_zero = 5e-3
            tol_std = 5e-3

            passed = (
                    abs(eps_xx_mean - a) < tol_mean and
                    abs(eps_yy_mean - b) < tol_mean and
                    abs(eps_xy_mean) < tol_zero and
                    abs(rot_mean) < tol_zero and
                    eps_xx_std < tol_std and
                    eps_yy_std < tol_std and
                    eps_xy_std < tol_std and
                    rot_std < tol_std
            )

            stats = {
                "target_eps_xx": a,
                "target_eps_yy": b,
                "target_eps_xy": 0.0,
                "target_rot": 0.0,
                "eps_xx_mean": eps_xx_mean,
                "eps_yy_mean": eps_yy_mean,
                "eps_xy_mean": eps_xy_mean,
                "rot_mean": rot_mean,
                "eps_xx_std": eps_xx_std,
                "eps_yy_std": eps_yy_std,
                "eps_xy_std": eps_xy_std,
                "rot_std": rot_std,
                "pad_used": pad,
            }

            return {
                "passed": passed,
                "stats": stats,
                "maps": {k: v for k, v in maps.items()},
            }

        finally:
            with torch.no_grad():
                self.control.copy_(old_control)

    def plot_self_test(self, H=None, W=None):
        import matplotlib.pyplot as plt

        result = self.self_test(H=H, W=W)
        maps = result["maps"]
        stats = result["stats"]

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        im = axes[0, 0].imshow(maps["ux"].detach().cpu().numpy(), origin="lower")
        axes[0, 0].set_title("ux")
        plt.colorbar(im, ax=axes[0, 0])

        im = axes[0, 1].imshow(maps["uy"].detach().cpu().numpy(), origin="lower")
        axes[0, 1].set_title("uy")
        plt.colorbar(im, ax=axes[0, 1])

        im = axes[0, 2].imshow(maps["eps_xx"], origin="lower")
        axes[0, 2].set_title(f"eps_xx mean={stats['eps_xx_mean']:.4f}")
        plt.colorbar(im, ax=axes[0, 2])

        im = axes[1, 0].imshow(maps["eps_yy"], origin="lower")
        axes[1, 0].set_title(f"eps_yy mean={stats['eps_yy_mean']:.4f}")
        plt.colorbar(im, ax=axes[1, 0])

        im = axes[1, 1].imshow(self.smooth_map(maps["eps_xy"]), origin="lower")
        axes[1, 1].set_title(f"eps_xy mean={stats['eps_xy_mean']:.4f}")
        plt.colorbar(im, ax=axes[1, 1])

        im = axes[1, 2].imshow(maps["rot"], origin="lower")
        axes[1, 2].set_title(f"rot mean={stats['rot_mean']:.4f}")
        plt.colorbar(im, ax=axes[1, 2])

        fig.suptitle(f"SplineDisplacementField2D self test: passed={result['passed']}")
        plt.tight_layout()
        plt.show()

        return result

if __name__ == "__main__":
    field = SplineDisplacementField2D(
        image_shape=(256, 256),
        control_shape=(100,100),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    result = field.self_test()
    print(result["passed"])
    print(result["stats"])

    field.plot_self_test()