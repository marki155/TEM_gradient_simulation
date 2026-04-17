import math
import numpy as np
import torch
from scipy.special import k0

from abtem.parametrizations import KirklandParametrization


def _get_kirkland_projected_parameters(atom_type: str):
    """
    Holt die von abTEM bereits skalierten Kirkland-Parameter
    für das projizierte Potential.

    Rückgabe:
        p: np.ndarray shape (4, 3)
           Reihen entsprechen [a, b, c, d]
    """
    kp = KirklandParametrization()
    p = kp.scaled_parameters(atom_type, "projected_potential")
    return np.asarray(p, dtype=np.float64)


def _evaluate_kirkland_projected_potential_from_radius(R_A, atom_type: str):
    """
    Evaluiert das projizierte Kirkland-Potential auf einem gegebenen Radius-Array.

    Parameter
    ---------
    R_A : np.ndarray
        Radius in Å
    atom_type : str
        Elementsymbol, z. B. "C", "S", "W"

    Returns
    -------
    np.ndarray
        Potential in abTEM-Konvention: (eV/e)*Å == V*Å
    """
    p = _get_kirkland_projected_parameters(atom_type)
    a, b, c, d = p[0], p[1], p[2], p[3]

    V = np.zeros_like(R_A, dtype=np.float64)
    for j in range(3):
        V += 2.0 * a[j] * k0(b[j] * R_A)
        V += math.sqrt(math.pi / d[j]) * c[j] * np.exp(-d[j] * R_A**2)
    return V


def _supersampled_pixel_average_kirkland(
    xc_A,
    yc_A,
    pixel_size_A,
    atom_type: str,
    supersample_factor: int,
    softcore_fraction: float,
):
    """
    Mittelt den Kirkland-Kernel über ein einzelnes Pixel per Supersampling.

    Das Pixel ist um (xc_A, yc_A) zentriert und hat Kantenlänge pixel_size_A.
    Die Singularität wird innerhalb jedes Subsamples weich regularisiert mit:
        r_eff = sqrt(r^2 + a_soft^2)

    Returns
    -------
    float
        Pixelgemitteltes projiziertes Potential in V*Å
    """
    n = supersample_factor
    # Subpixelzentren in [-0.5, 0.5) * pixel_size_A
    offs = ((np.arange(n, dtype=np.float64) + 0.5) / n - 0.5) * pixel_size_A
    dY, dX = np.meshgrid(offs, offs, indexing="ij")

    Xs = xc_A + dX
    Ys = yc_A + dY

    R = np.sqrt(Xs**2 + Ys**2)

    a_soft = max(pixel_size_A * softcore_fraction, 1e-8)
    R_eff = np.sqrt(R**2 + a_soft**2)

    V_sub = _evaluate_kirkland_projected_potential_from_radius(R_eff, atom_type)
    return float(V_sub.mean())


def projected_atomic_potential_kernel_kirkland(
    H,
    W,
    atom_type: str,
    pixel_size_nm,
    device,
    supersample_radius_px: float = 2.0,
    supersample_factor: int = 8,
    softcore_fraction: float = 0.1,
):
    """
    Baut den projizierten Atomkernel V_proj(r) auf deinem (H, W)-Raster.

    Verwendet die von abTEM skalierten Kirkland-Parameter und die Form:
        V_proj(r) = sum_j [ 2*a_j*K0(b_j*r) + sqrt(pi/d_j)*c_j*exp(-d_j*r^2) ]

    Einheiten:
    - Radius intern in Å
    - Rückgabe in V·nm, passend zu sigma in rad / (V·nm)

    Numerik:
    - im Fernfeld normale Punktauswertung
    - im Nahfeld um das Zentrum Pixelmittelung per Supersampling
    - innerhalb jedes Subsamples weiche Soft-Core-Regularisierung

    Parameter
    ---------
    supersample_radius_px : float
        Radius um das Zentrum in Pixeln, innerhalb dessen supersampled wird.
    supersample_factor : int
        Unterteilungen pro Pixelkante für die Pixelmittelung.
    softcore_fraction : float
        Weiche Regularisierung als Bruchteil der Pixelgröße in Å.
    """
    pixel_size_A = float(pixel_size_nm) * 10.0  # nm -> Å

    # Pixelzentren auf dem HxW-Gitter
    x = (torch.arange(H, device=device, dtype=torch.float32) - H // 2) * pixel_size_A
    y = (torch.arange(W, device=device, dtype=torch.float32) - W // 2) * pixel_size_A
    X, Y = torch.meshgrid(x, y, indexing="ij")
    R = torch.sqrt(X**2 + Y**2)

    # Fernfeld: normale Auswertung mit weicher Regularisierung
    a_soft = max(pixel_size_A * softcore_fraction, 1e-8)
    R_eff = torch.sqrt(R**2 + a_soft**2)
    R_cpu = R_eff.detach().cpu().numpy()

    V = _evaluate_kirkland_projected_potential_from_radius(R_cpu, atom_type)

    # Nahfeld: Pixelmittelung per Supersampling
    if supersample_radius_px is not None and supersample_radius_px > 0 and supersample_factor > 1:
        radius_A = supersample_radius_px * pixel_size_A

        x_centers_A = ((np.arange(H, dtype=np.float64) - H // 2) * pixel_size_A)
        y_centers_A = ((np.arange(W, dtype=np.float64) - W // 2) * pixel_size_A)

        for i, xc_A in enumerate(x_centers_A):
            for j, yc_A in enumerate(y_centers_A):
                if math.hypot(xc_A, yc_A) <= radius_A:
                    V[i, j] = _supersampled_pixel_average_kirkland(
                        xc_A=xc_A,
                        yc_A=yc_A,
                        pixel_size_A=pixel_size_A,
                        atom_type=atom_type,
                        supersample_factor=supersample_factor,
                        softcore_fraction=softcore_fraction,
                    )

    # (eV/e)*Å == V*Å  ->  V*nm
    V = torch.tensor(V, dtype=torch.float32, device=device) * 0.1

    # Zentrum nach (0,0) für FFT-Faltung
    V = torch.fft.ifftshift(V)
    return V


class KirklandPrecalcPotential:
    def __init__(
        self,
        H,
        W,
        pixel_size_nm,
        device,
        supersample_radius_px: float = 2.0,
        supersample_factor: int = 8,
        softcore_fraction: float = 0.001,
    ):
        self.H = H
        self.W = W
        self.pixel_size_nm = pixel_size_nm
        self.device = device

        self.supersample_radius_px = supersample_radius_px
        self.supersample_factor = supersample_factor
        self.softcore_fraction = softcore_fraction

        self.kernels = {
            "C": projected_atomic_potential_kernel_kirkland(
                H, W, "C", pixel_size_nm, device,
                supersample_radius_px=supersample_radius_px,
                supersample_factor=supersample_factor,
                softcore_fraction=softcore_fraction,
            ),
            "S": projected_atomic_potential_kernel_kirkland(
                H, W, "S", pixel_size_nm, device,
                supersample_radius_px=supersample_radius_px,
                supersample_factor=supersample_factor,
                softcore_fraction=softcore_fraction,
            ),
            "W": projected_atomic_potential_kernel_kirkland(
                H, W, "W", pixel_size_nm, device,
                supersample_radius_px=supersample_radius_px,
                supersample_factor=supersample_factor,
                softcore_fraction=softcore_fraction,
            ),
        }

    def projected_slice_potential_kirkland(self, rho, atom_type: str, pixel_size_nm=None, device=None):
        kernel = self.kernels[atom_type]
        V = torch.fft.ifft2(torch.fft.fft2(rho) * torch.fft.fft2(kernel)).real
        return V


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    atom_type = "C"
    pixel_size_nm = 0.02
    H = 257
    W = 257
    device = "cpu"

    kernel = projected_atomic_potential_kernel_kirkland(
        H=H,
        W=W,
        atom_type=atom_type,
        pixel_size_nm=pixel_size_nm,
        device=device,
        supersample_radius_px=2.0,
        supersample_factor=8,
        softcore_fraction=0.01,
    )

    # zurück in "zentrierte Bildlage" nur fürs Plotten
    kernel_centered = torch.fft.fftshift(kernel).cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(kernel_centered, origin="lower")
    plt.colorbar()
    plt.title(f"Kirkland kernel ({atom_type}), supersampled center")
    plt.tight_layout()
    plt.show()

    # 1D-Referenzvergleich gegen abTEM-Funktion
    kp = KirklandParametrization()
    f_ref = kp.projected_potential(atom_type)

    r_A = np.linspace(0.01, 5.0, 500)
    v_ref = f_ref(r_A) * 0.1  # V·Å -> V·nm

    p = _get_kirkland_projected_parameters(atom_type)
    a, b, c, d = p[0], p[1], p[2], p[3]

    v_own = np.zeros_like(r_A)
    for j in range(3):
        v_own += 2.0 * a[j] * k0(b[j] * r_A)
        v_own += np.sqrt(np.pi / d[j]) * c[j] * np.exp(-d[j] * r_A**2)
    v_own *= 0.1

    plt.figure(figsize=(6, 4))
    plt.plot(r_A, v_ref, label="abTEM ref")
    plt.plot(r_A, v_own, "--", label="own analytic")
    plt.xlabel("r [Å]")
    plt.ylabel("projected potential [V·nm]")
    plt.legend()
    plt.tight_layout()
    plt.show()