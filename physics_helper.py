import math


def relativistic_wavelength_nm(voltage_kV):
    """Relativistische Elektronenwellenlänge in nm."""
    V = voltage_kV * 1e3
    h = 6.62607015e-34
    m0 = 9.10938356e-31
    e = 1.602176634e-19
    c = 299792458.0
    E = e * V
    p = math.sqrt(E**2 + 2 * E * m0 * c**2) / c
    lam_m = h / p
    return lam_m * 1e9  # nm