import math
import torch

# Rohwerte direkt aus Tabelle 1 des Papers
LOBATO_TABLE = {
    "C": {
        "a": [
            1.244660886213433e+02,
           -2.203528570789638e+02,
            1.952353522804791e+02,
           -9.810793612697997e+01,
            1.420230412136232e-02,
        ],
        "b_A2": [
            2.421208492560056e+00,
            2.305379437524258e+00,
            2.048519321065642e+00,
            1.933525529175474e+00,
            7.689768184783397e-02,
        ],
    },
    "S": {
        "a": [
            2.679714156101995e+00,
           -4.742528222307552e-01,
            5.148359489896877e-01,
           -9.583600249907225e-02,
            2.488719638189352e-03,
        ],
        "b_A2": [
            3.068891211999711e+00,
            3.782167021858090e-01,
            1.887218119025488e-01,
            9.233705900314941e-02,
            1.119208772411441e-02,
        ],
    },
    "W": {
        "a": [
            3.976297158190774e+00,
            1.522926842573418e+00,
            7.977062463905613e-01,
            2.136667415586852e-02,
            1.030786893857173e-04,
        ],
        "b_A2": [
            6.296857792287749e+00,
            1.112899511576911e+00,
            2.310570406784734e-01,
            1.510135455672045e-02,
            4.746824771050258e-04,
        ],
    },
}


def b_A2_to_nm2(b_A2: float) -> float:
    return b_A2 * 0.01

def a_A_to_nm(a: float) -> float:
    return a * 0.1


def get_lobato_params_nm(atom_type: str):
    raw = LOBATO_TABLE[atom_type]
    a = [a_A_to_nm(x) for x in raw["a"]]
    b_nm2 = [b_A2_to_nm2(x) for x in raw["b_A2"]]
    return a, b_nm2


def eta_lobato_SI():
    eps0 = 8.8541878128e-12 * 1e-9  # F/m
    a0 = 5.29177210903e-11 * 1e9  # m
    e = 1.602176634e-19       # C
    return 2.0 * eps0 / (a0 * e)


def projected_atomic_potential_kernel_lobato(H, W, pixel_size_nm, atom_type, device):
    y = (torch.arange(H, device=device) - H // 2) * pixel_size_nm
    x = (torch.arange(W, device=device) - W // 2) * pixel_size_nm
    Y, X = torch.meshgrid(y, x, indexing="ij")
    R = torch.sqrt(X**2 + Y**2 + 1e-20)

    a_vals, b_vals_nm2 = get_lobato_params_nm(atom_type)
    eta = eta_lobato_SI()

    V = torch.zeros((H, W), dtype=torch.float32, device=device)

    for a_i, b_i in zip(a_vals, b_vals_nm2):
        sqrt_b = math.sqrt(b_i)
        u = 2.0 * R / sqrt_b

        # Achtung:
        # torch.special.modified_bessel_k0 / k1 je nach Torch-Version prüfen
        k0 = torch.special.modified_bessel_k0(u)
        k1 = torch.special.modified_bessel_k1(u)

        term = (2.0 * a_i / (b_i ** 1.5)) * (k0 + (R / sqrt_b) * k1)
        V = V + term

    V = V / eta
    return torch.fft.ifftshift(V)