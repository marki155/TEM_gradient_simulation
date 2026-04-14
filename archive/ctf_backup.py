TEM_TO_ZERNIKE = {
    "C10": (2, 0),   # Defokus
    "C12": (2, 2),   # 2-fach Astigmatismus
    "C21": (3, 1),   # Koma
    "C23": (3, 3),   # 3-fach Astigmatismus
    "C30": (4, 0),   # Sphärische Aberration
    "C32": (4, 2),   # 4-fach Astigmatismus
    "C34": (4, 4),   # 4-fach Astigmatismus
}

# Radialteil
def R_nm(n, m, rho):
    m = abs(m)
    radial = torch.zeros_like(rho)
    for k in range((n - m)//2 + 1):
        coef = ((-1)**k * math.factorial(n - k)) / (
            math.factorial(k) *
            math.factorial((n + m)//2 - k) *
            math.factorial((n - m)//2 - k)
        )
        radial += coef * rho**(n - 2*k)
    return radial

# Zernike Polynom
def Z_nm(n, m, rho, theta):
    if m > 0:
        return R_nm(n, m, rho) * torch.cos(m * theta)
    elif m < 0:
        return R_nm(n, -m, rho) * torch.sin(-m * theta)
    else:
        return R_nm(n, 0, rho)

# Phase auf kx/ky-Gitter
def zernike_phase_(kx, ky, lam_nm, aberrations):
    rho = torch.sqrt(kx**2 + ky**2)
    theta = torch.atan2(ky, kx)
    # Normierung auf Einheitskreis: kmax = 1/lam (optional, hier einfach rho unbegrenzt)
    phase = torch.zeros_like(rho)
    for key, c in aberrations.items():
        if key not in TEM_TO_ZERNIKE:
            continue
        n, m = TEM_TO_ZERNIKE[key]
        phase += c * Z_nm(n, m, rho, theta)
    return phase