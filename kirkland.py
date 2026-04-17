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



def projected_atomic_potential_kernel_kirkland(H, W, atom_type: str, pixel_size_nm, device):
    """
    Baut den projizierten Atomkernel V_proj(r) auf deinem (H, W)-Raster.

    Benutzt die von abTEM skalierten Kirkland-Parameter und die
    abTEM-Form des projizierten Potentials:

        V_proj(r) = sum_j [ 2*a_j*K0(b_j*r) + sqrt(pi/d_j)*c_j*exp(-d_j*r^2) ]

    Wichtige Notiz:
    K0 divergiert für r -> 0 logarithmisch. Deshalb wird r unten mit
    einem kleinen floor versehen.
    """

    pixel_size_A = pixel_size_nm * 10.0  # nm -> Å für Kirkland-Parameter


    x = (torch.arange(H, device=device, dtype=torch.float32) - H // 2) * pixel_size_A
    y = (torch.arange(W, device=device, dtype=torch.float32) - W // 2) * pixel_size_A
    X, Y = torch.meshgrid(x, y, indexing="ij")
    R = torch.sqrt(X**2 + Y**2)

    # K0 hat bei r=0 eine Singularität -> numerischer floor
    #r_floor = max(pixel_size_A * 0.25, 1e-6)
    #R_safe = torch.clamp(R, min=r_floor)

    a = pixel_size_A * 0.01
    R_eff = torch.sqrt(R ** 2 + a ** 2)

    p = _get_kirkland_projected_parameters(atom_type)
    a = p[0]
    b = p[1]
    c = p[2]
    d = p[3]

    # scipy.special.k0 läuft auf CPU/NumPy; Kernel ist fest, daher okay
    R_cpu = R_eff.detach().cpu().numpy()



    V = np.zeros((H, W), dtype=np.float64)

    for j in range(3):
        V += 2.0 * a[j] * k0(b[j] * R_cpu)
        V += math.sqrt(math.pi / d[j]) * c[j] * np.exp(-d[j] * R_cpu**2)



    V = torch.tensor(V, dtype=torch.float32, device=device) * 0.1  # Å -> nm für Ausgabe
    V = torch.fft.ifftshift(V)

    #R_cut = 300
    #V = V * (torch.tensor(R_cpu) <= R_cut)

    return V

class KirklandPrecalcPotential:
    def __init__(self, H, W, pixel_size_nm, device):
        KERNELS = { "C": projected_atomic_potential_kernel_kirkland(H, W, "C", pixel_size_nm, device),
                    "S": projected_atomic_potential_kernel_kirkland(H, W, "S", pixel_size_nm, device),
                    "W": projected_atomic_potential_kernel_kirkland(H, W, "W", pixel_size_nm, device) }
        self.kernels = KERNELS

    def projected_slice_potential_kirkland(self, rho, atom_type: str, pixel_size_nm, device):
        kernel = self.kernels[atom_type]
        #kernel = kernel - kernel.mean()
        V = torch.fft.ifft2(torch.fft.fft2(rho) * torch.fft.fft2(kernel)).real

        return V


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from abtem.parametrizations import KirklandParametrization

    kp = KirklandParametrization()
    f_ref = kp.projected_potential("C")  # direkte abTEM-Referenzfunktion

    r_A = np.linspace(0.01, 5.0, 500)  # Å
    v_ref = f_ref(r_A)  # Referenz

    # deine Implementierung als 1D-Kurve:
    p = _get_kirkland_projected_parameters("C")
    a, b, c, d = p[0], p[1], p[2], p[3]

    from scipy.special import k0

    v_own = np.zeros_like(r_A)
    for j in range(3):
        v_own += 2.0 * a[j] * k0(b[j] * r_A)
        v_own += np.sqrt(np.pi / d[j]) * c[j] * np.exp(-d[j] * r_A ** 2)

    plt.plot(r_A, v_ref, label="abTEM ref")
    plt.plot(r_A, v_own, "--", label="own")
    plt.legend()
    plt.show()