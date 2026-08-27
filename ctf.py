from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

from physics_helper import relativistic_wavelength_nm


def scherzer_defocus(C30_um, lam_nm):
    C30_nm = C30_um*1000.
    return - 1.2 * math.sqrt(C30_nm * lam_nm)

def build_ctf_envelope_on_k(kx, ky, voltage_kV, lam, params=None, deltaE=None, Cc_um=None,
                            cache=None, defocus_nm=None, cs_nm=None):
    """Gauss-Envelopes im k-Raum.

    ``cache``: optionales dict. Alles was nur am Gitter haengt (k, k2, k3, die
    Instabilitaets-Envelope) wird dort abgelegt und beim naechsten Aufruf
    wiederverwendet.  Die chromatische Envelope haengt an deltaE/Cc, die
    raeumliche an Defokus und Cs - beide werden jedes Mal neu berechnet.

    ``defocus_nm`` / ``cs_nm``: fuer die raeumliche Kohaerenz. Sie daempft ueber
    den Gradienten der Aberrationsfunktion,

        E_s(k) = exp[ -(pi*alpha/lam)^2 * (Cs*lam^3*k^3 + df*lam*k)^2 ]

    mit alpha als Quell-Halbwinkel in rad.  (Frueher stand hier
    exp(-(pi*alpha*k)^2), was alpha wie eine Laenge in nm behandelt hat - bei
    0.1 mrad war die Daempfung damit exakt 1, die Envelope also wirkungslos.)
    """

    if params is None:
        params = {}
    if cache is None:
        cache = {}

    if "k" in cache:
        k2, k = cache["k2"], cache["k"]
    else:
        k2 = kx**2 + ky**2
        k = torch.sqrt(k2 + 1e-30)
        cache["k2"], cache["k"] = k2, k
    eV = voltage_kV * 1e3

    # Spatial (source size / convergence), alpha als Halbwinkel in rad
    E_spatial = 1.0
    alpha = float(params.get('source_size', 0.0)) if params.get('use_spatial', True) else 0.0
    if alpha > 0:
        k3 = cache.get("k3")
        if k3 is None:
            k3 = k ** 3
            cache["k3"] = k3

        gradient = torch.zeros_like(k)
        if defocus_nm is not None:
            gradient = gradient + defocus_nm * lam * k
        if cs_nm is not None:
            gradient = gradient + cs_nm * lam ** 3 * k3

        E_spatial = torch.exp(-((math.pi * alpha / lam) ** 2) * gradient ** 2)

    # Chromatic (temporal) via defocus spread Δf = Cc * ΔE / E0
    E_chrom = 1.0
    if params.get('use_chromatic', True):
        if Cc_um is None:
            Cc_nm = float(params.get('Cc_um', 1e7))
        else:
            Cc_nm = float(Cc_um) * 1e3   # nm (10 mm)
        if deltaE is not None:
            dE = deltaE
        else:
            float(params.get('deltaE', 0.7))        # eV
        delta_f_nm = Cc_nm * (dE / eV)   # nm
        E_chrom = torch.exp(- (math.pi * lam * delta_f_nm * k2)**2)

    # Instability (extra defocus jitter)
    E_inst = cache.get("E_inst", 1.0)
    if "E_inst" not in cache and params.get('use_instability', True):
        sigma_df_nm = float(params.get('instab_defocus_nm', 0.0))
        if sigma_df_nm > 0:
            E_inst = torch.exp(- (math.pi * lam * sigma_df_nm * k2)**2)
        else:
            E_inst = torch.ones_like(k)
        cache["E_inst"] = E_inst

    # Detector (optional, Gaussian in k)
    E_det = cache.get("E_det", 1.0)
    if "E_det" not in cache and params.get('use_detector', False):
        sigma_k = params.get('detector_sigma_k', None)
        if sigma_k is not None:
            sigma_k = float(sigma_k)
            E_det = torch.exp(-(k**2) / (2.0 * sigma_k**2))
        else:
            E_det = torch.ones_like(k)
        cache["E_det"] = E_det

    env = E_spatial * E_chrom * E_inst * E_det
    if not torch.is_tensor(env):
        # alle Envelopes abgeschaltet -> konstante Eins in Gitterform
        env = torch.full_like(k, float(env))
    return torch.clamp(env, 0.0, 1.0)





#: Aberrations-Terme: key -> (Potenz von alpha, Vorfaktor, Einheiten-Skalierung,
#: Multiplizitaet m des Azimutterms).  m = 0 heisst rotationssymmetrisch (kein
#: Winkel).  Die Werte sind exakt die, die vorher ausgeschrieben im Code standen
#: (C30/C34/C45 stehen in um, deshalb die 1e3).
ABERRATION_TERMS = {
    "C10": (2, 0.5, 1.0, 0),      # Defokus
    "C12": (2, 0.5, 1.0, 2),      # 2-zaehliger Astigmatismus
    "C21": (3, 1 / 3, 1.0, 1),    # Koma
    "C23": (3, 1 / 3, 1.0, 3),    # 3-zaehliger Astigmatismus
    "C30": (4, 0.25, 1e3, 0),     # Spaerische Aberration
    "C32": (4, 0.25, 1.0, 2),
    "C34": (4, 0.25, 1e3, 4),
    "C41": (5, 0.2, 1.0, 1),
    "C43": (5, 0.2, 1.0, 3),
    "C45": (5, 0.2, 1e3, 5),
    "C50": (6, 1 / 6, 1.0, 0),
    "C52": (6, 1 / 6, 1.0, 2),
    "C54": (6, 1 / 6, 1.0, 4),
    "C56": (6, 1 / 6, 1.0, 6),
}

#: Wertebereiche, die frueher direkt in zernike_phase geklammert wurden
ABERRATION_CLAMPS = {"C10": (-1000., 1000.), "C30": (-20., 20.)}

PARAMETRIZATION_POLAR = "polar"
PARAMETRIZATION_CARTESIAN = "cartesian"


def angle_key(magnitude: str) -> str:
    """``C12`` -> ``phi12``"""

    return "phi" + magnitude[1:]


def cartesian_keys(magnitude: str):
    """``C12`` -> ``('a12', 'b12')``"""

    return "a" + magnitude[1:], "b" + magnitude[1:]


def is_angle_key(key: str) -> bool:
    """``phi12`` & Co. sind die Azimutwinkel der polaren Schreibweise."""

    return key.startswith("phi")


def magnitude_key(key: str) -> str:
    """Jeder Aberrations-Key auf seinen ``C..``-Key abgebildet.

    ``phi12`` / ``a12`` / ``b12`` gehoeren alle zu ``C12``.
    """

    if is_angle_key(key):
        return "C" + key[3:]
    if key[0] in ("a", "b") and key[1:].isdigit():
        return "C" + key[1:]
    return key


def polar_to_cartesian_aberrations(coeffs: dict) -> dict:
    """{'C12': C, 'phi12': phi} -> {'a12': C*cos(m*phi), 'b12': C*sin(m*phi)}

    Rotationssymmetrische Terme (C10, C30, ...) bleiben unveraendert.
    """

    out = {}
    for key, value in coeffs.items():
        if is_angle_key(key):
            continue
        term = ABERRATION_TERMS.get(key)
        if term is None or term[3] == 0:
            out[key] = value
            continue
        m = term[3]
        phi = float(coeffs.get(angle_key(key), 0.0))
        c = float(value)
        a_key, b_key = cartesian_keys(key)
        out[a_key] = c * math.cos(m * phi)
        out[b_key] = c * math.sin(m * phi)
    return out


def cartesian_to_polar_aberrations(coeffs: dict) -> dict:
    """Umkehrung: {'a12': a, 'b12': b} -> {'C12': hypot(a,b), 'phi12': atan2(b,a)/m}"""

    out = {}
    for key, value in coeffs.items():
        if key[0] in ("a", "b") and key[1:].isdigit():
            continue
        out[key] = value

    for magnitude, (_power, _pref, _scale, m) in ABERRATION_TERMS.items():
        if m == 0:
            continue
        a_key, b_key = cartesian_keys(magnitude)
        if a_key not in coeffs or b_key not in coeffs:
            continue
        a = float(coeffs[a_key])
        b = float(coeffs[b_key])
        out[magnitude] = math.hypot(a, b)
        out[angle_key(magnitude)] = math.atan2(b, a) / m
    return out


def convert_aberrations(coeffs: dict, parametrization: str) -> dict:
    """Bringt ein Aberrations-Dict in die gewuenschte Schreibweise."""

    if parametrization == PARAMETRIZATION_CARTESIAN:
        return polar_to_cartesian_aberrations(coeffs)
    if parametrization == PARAMETRIZATION_POLAR:
        return cartesian_to_polar_aberrations(coeffs)
    raise ValueError(f"Unbekannte Parametrisierung: {parametrization!r}")


def aberration_basis_maps(alpha, phi, keys=None):
    """Vorberechnete Ortsfunktionen der Aberrationsterme.

    Fuer jede Ordnung ``Cnm`` liefert das die fertigen Karten

        a-Karte = vorfaktor * alpha**n * cos(m*phi)
        b-Karte = vorfaktor * alpha**n * sin(m*phi)

    (bzw. nur die radiale Karte fuer m = 0).  Sie haengen ausschliesslich am
    Gitter, nicht an den Aberrationswerten - die Phase ist danach nur noch eine
    Linearkombination dieser Karten.  Das spart pro Aufruf die Potenzen und die
    Winkelfunktionen auf dem ganzen Gitter.
    """

    if keys is None:
        magnitudes = list(ABERRATION_TERMS)
    else:
        magnitudes = []
        for key in keys:
            magnitude = magnitude_key(key)
            if magnitude in ABERRATION_TERMS and magnitude not in magnitudes:
                magnitudes.append(magnitude)

    radial_cache = {}

    def radial(power):
        if power not in radial_cache:
            value = alpha * alpha
            for _ in range(power - 2):
                value = value * alpha
            radial_cache[power] = value
        return radial_cache[power]

    maps = {}
    for magnitude in magnitudes:
        power, prefactor, scale, m = ABERRATION_TERMS[magnitude]
        base = (prefactor * scale) * radial(power)

        if m == 0:
            maps[magnitude] = base
        else:
            a_key, b_key = cartesian_keys(magnitude)
            maps[a_key] = base * torch.cos(m * phi)
            maps[b_key] = base * torch.sin(m * phi)

    return maps


def zernike_phase(alpha, phi, coeffs, device='cpu', basis=None):
    """Aberrationsphase chi(alpha, phi) / (2 pi / lambda).

    Versteht beide Schreibweisen:

    * polar     ``C12 * cos(m * (phi - phi12))``
    * kartesisch ``a12 * cos(m * phi) + b12 * sin(m * phi)``

    Beides ist dieselbe Funktion (a = C cos(m phi12), b = C sin(m phi12)) - die
    kartesische Form ist nur besser zu optimieren: der Gradient nach dem Winkel
    skaliert in der polaren Form mit dem Betrag und verschwindet, wenn der
    Betrag klein wird.
    """

    if basis is None:
        basis = aberration_basis_maps(alpha, phi, coeffs.keys())

    # Device von alpha uebernehmen - der device-Parameter ist historisch und
    # stand per Default auf "cpu", was auf der GPU zu Mischbetrieb gefuehrt hat
    phase = torch.zeros_like(alpha, dtype=torch.float32)

    for magnitude, (_power, _prefactor, _scale, m) in ABERRATION_TERMS.items():
        if m == 0:
            if magnitude not in coeffs or magnitude not in basis:
                continue
            value = coeffs[magnitude]
            clamp = ABERRATION_CLAMPS.get(magnitude)
            if clamp is not None:
                value = torch.clamp(value, min=clamp[0], max=clamp[1])
            phase = phase + basis[magnitude] * value
            continue

        a_key, b_key = cartesian_keys(magnitude)
        if a_key not in basis:
            continue

        if a_key in coeffs and b_key in coeffs:
            a, b = coeffs[a_key], coeffs[b_key]
        elif magnitude in coeffs and angle_key(magnitude) in coeffs:
            # polare Schreibweise: cos(m*(phi-phi_m)) ausmultipliziert, damit
            # dieselben vorberechneten Karten benutzt werden koennen
            amplitude = coeffs[magnitude]
            argument = m * coeffs[angle_key(magnitude)]
            a = amplitude * torch.cos(argument)
            b = amplitude * torch.sin(argument)
        else:
            continue

        phase = phase + basis[a_key] * a + basis[b_key] * b

    return phase


DEFAULT_ABERRATIONS = {
    "C10": -13.07,
    'C12': 0.35, "phi12": 0.,
    'C23': 0.17, 'phi23': 0.,
    'C21': 3.71, 'phi21': 0.,
    "C30": 4.16,  # um
    'C34': -0.2, 'phi34': 0.,  # um
    'C45': 0.25, 'phi45': 0.,
}


def make_param_dict(a=None, parametrization=PARAMETRIZATION_CARTESIAN, device=None):
    if a is None:
        a = DEFAULT_ABERRATIONS

    a = convert_aberrations(a, parametrization)

    return nn.ParameterDict({
        k: nn.Parameter(torch.tensor(float(v), dtype=torch.float32, device=device))
        for k, v in a.items()
    })


def migrate_serialized_ctf_parameters(source, parametrization=PARAMETRIZATION_CARTESIAN):
    """Rechnet die Aberrationen einer gespeicherten Parameter-Datei um.

    Alte Dateien stehen in polarer Form (C12/phi12), neue in kartesischer
    (a12/b12) - je nach ``parametrization`` wird in die passende umgerechnet.
    """

    if not isinstance(source, dict):
        return source

    aberrations = source.get("aberrations")
    if not isinstance(aberrations, dict):
        return source

    out = dict(source)
    out["aberrations"] = convert_aberrations(aberrations, parametrization)
    return out


@dataclass
class TunableCTFParameters:
    # default_factory, sonst teilen sich alle Instanzen dieselben Parameter
    amp_contrast: nn.Parameter = field(
        default_factory=lambda: nn.Parameter(torch.tensor(1.0, dtype=torch.float32)))
    deltaE: nn.Parameter = field(
        default_factory=lambda: nn.Parameter(torch.tensor(0.45, dtype=torch.float32)))
    Cc_um: nn.Parameter = field(
        default_factory=lambda: nn.Parameter(torch.tensor(1.e3, dtype=torch.float32)))
    aberrations: nn.ParameterDict = field(default_factory=make_param_dict)  # nm bzw. um


class CTFLayer(nn.Module):
    """Kontrastuebertragung - und im STEM-Zweig der Sondenformer.

    ``aberrations`` ueberschreibt ``config.init_aberrations``: die Sonde im STEM
    hat andere Werte als die Abbildung im TEM (dort steht der Defokus typisch
    bei -13 nm, die Sonde will ihn nahe null).
    """

    def __init__(self, config, train_method=None, device=None, aberrations=None):
        super().__init__()
        self.H, self.W = config.shape
        self.pixel_size_nm = config.pixel_size_nm
        self.voltage_kV = config.voltage_kV
        #self.amp_contrast = nn.Parameter(torch.tensor(amp_contrast, device=device))
        self.envelope_params = config.init_ctf_env_params
        #self.deltaE = nn.Parameter(torch.tensor(envelope_params["deltaE"], device=device))
        #self.Cc_um = nn.Parameter(torch.tensor(envelope_params["Cc_um"], device=device))
        self.lam = relativistic_wavelength_nm(config.voltage_kV)
        self.semiangle_cutoff_mrad = None
        self.alpha_max_mrad = config.ctf_alpha_max_mrad
        self.device = device
        self.train_method = train_method
        self.parametrization = getattr(config, "ctf_parametrization", PARAMETRIZATION_CARTESIAN)

        # device explizit mitgeben: potential.py setzt beim Import ein globales
        # Default-Device, das nicht zwingend das dieses Layers ist
        self.tunable_parameters = TunableCTFParameters(
            amp_contrast=nn.Parameter(torch.tensor(config.init_amp_contrast,
                                                   dtype=torch.float32, device=device)),
            deltaE=nn.Parameter(torch.tensor(self.envelope_params["deltaE"],
                                             dtype=torch.float32, device=device)),
            Cc_um=nn.Parameter(torch.tensor(self.envelope_params["Cc_um"],
                                            dtype=torch.float32, device=device)),
            aberrations=nn.ParameterDict(make_param_dict(
                config.init_aberrations if aberrations is None else aberrations,
                self.parametrization, device=device)),
        )

        pad = config.padding
        self.pad = pad


        # Frequenzen in 1/nm
        fx = torch.fft.fftshift(torch.linspace(-.5/config.pixel_size_nm, .5/config.pixel_size_nm, self.H + 2*pad, device=device))
        fy = torch.fft.fftshift(torch.linspace(-.5/config.pixel_size_nm, .5/config.pixel_size_nm, self.W + 2*pad, device=device))
        kx, ky = torch.meshgrid(fx, fy, indexing='ij')

        k = torch.sqrt(kx ** 2 + ky ** 2)

        # maximal zugelassener Frequenzradius
        k_max = (self.alpha_max_mrad * 1e-3) / self.lam  # [1/nm]

        aperture = (k <= k_max).float()

        self.register_buffer("aperture", aperture)


        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

        # alpha/phi und die Ortsfunktionen der Aberrationen haengen nur am
        # Gitter - einmal berechnen statt bei jedem Forward.
        alpha = self.lam * torch.sqrt(kx ** 2 + ky ** 2)
        phi = torch.atan2(ky, kx)
        self.register_buffer('alpha', alpha)
        self.register_buffer('phi', phi)
        self.aberration_basis = aberration_basis_maps(
            alpha, phi, self.tunable_parameters.aberrations.keys())
        self._envelope_cache = {}

        for name, value in vars(self.tunable_parameters).items():
            if isinstance(value, (nn.Parameter, nn.ParameterList, nn.ParameterDict)):
                setattr(self, name, value)

    def init_aberrations(self, aberrations=None):
        # Wenn keine Aberrationen angegeben: leerer Dict
        if aberrations is None:
            aberrations = {}

        # Alle Aberrations-Cij als nn.Parameter speichern
        # z.B. 'C10', 'C12', 'C23', ...
        self.aberr_params = nn.ParameterDict()
        for key, val in aberrations.items():
            if key == "C10" and val == "scherzer":
                self.aberr_params[key] = nn.Parameter(torch.tensor(scherzer_defocus(aberrations["C30"],
                                                                                    self.lam), device=device))
            else:
                self.aberr_params[key] = nn.Parameter(torch.tensor(val, device=device))


    def get_tunable_parameters(self):
        return self.tunable_parameters

    def aberration_is_trainable(self, key: str) -> bool:
        """Wird dieser Aberrations-Parameter gerade mitoptimiert?

        Betrag und Azimutwinkel einer Ordnung gehoeren zusammen: ``phi12``
        folgt der Freigabe von ``C12`` in ``train_aberrations``, das Ganze
        haengt am Schalter ``train_ctf_aberrations``.
        """

        tm = self.train_method
        if tm is None:
            return True

        if not tm.train_ctf_aberrations.enabled:
            return False

        return bool(tm.train_aberrations.get(magnitude_key(key), False))



    def gated_aberrations(self):
        """Aberrationswerte, je nach Trainierbarkeit angehaengt oder abgeloest."""

        return {k: (v if self.aberration_is_trainable(k) else v.detach())
                for k, v in self.tunable_parameters.aberrations.items()}

    def aberration_phase(self, kx=None, ky=None):
        """chi(k) in rad auf dem eigenen Gitter.

        Dieselbe Phase, die in die CTF geht - der STEM-Zweig baut daraus seine
        Sonde, damit Defokus und Astigmatismus zwischen beiden geteilt sind.
        """

        own_grid = kx is None and ky is None
        if own_grid:
            alpha, phi, basis = self.alpha, self.phi, self.aberration_basis
        else:
            kx = self.kx if kx is None else kx
            ky = self.ky if ky is None else ky
            alpha = self.lam * torch.sqrt(kx ** 2 + ky ** 2)
            phi = torch.atan2(ky, kx)
            basis = None

        aberr_phase = zernike_phase(alpha, phi, self.gated_aberrations(),
                                    device=self.device, basis=basis)
        return 2.0 * math.pi * aberr_phase / self.lam

    def _build_ctf(self, kx=None, ky=None):
        # eigenes Gitter -> vorberechnete Karten benutzen; ein uebergebenes
        # Gitter (z.B. fuer die Beugungsbild-Ansicht) wird frisch gerechnet
        own_grid = kx is None and ky is None
        if kx is None:
            kx = self.kx
        if ky is None:
            ky = self.ky

        aberrations = self.gated_aberrations()
        lam = self.lam  # in nm

        #kx = ax / lam_nm
        #ky = ay / lam_nm


        if own_grid:
            alpha, phi = self.alpha, self.phi
            basis = self.aberration_basis
            envelope_cache = self._envelope_cache
        else:
            alpha = self.lam * torch.sqrt(kx ** 2 + ky ** 2)  # Radialabstand
            phi = torch.atan2(ky, kx)
            basis = None
            envelope_cache = None

        aberr_phase = zernike_phase(alpha, phi, aberrations, device=self.device, basis=basis)
        chi = 2.0 * math.pi * aberr_phase / lam  # klassische TEM-Phase
        if self.train_method is not None and self.train_method.train_amplitude_contrast.enabled:
            Q = torch.clamp(self.tunable_parameters.amp_contrast, 0.0, 1.0)
        else:
            Q = torch.clamp(self.tunable_parameters.amp_contrast.detach(), 0.0, 1.0)
        ac = math.sqrt(1.0 - Q ** 2) - 1j * Q
        ctf = ac * torch.exp(-1j * chi)

        if self.train_method.train_env_params["deltaE"] and self.train_method.train_env_parameters.enabled:
            deltaE = torch.clamp(self.tunable_parameters.deltaE, min=0.25, max=0.45)
        else:
            deltaE = self.tunable_parameters.deltaE.detach()

        if self.train_method.train_env_params["Cc"] and self.train_method.train_env_parameters.enabled:
            Cc_um = torch.clamp(self.tunable_parameters.Cc_um, min=1.e2, max=1.e4)
        else:
            Cc_um = self.tunable_parameters.Cc_um.detach()

        # zentrieren
        env = None
        if self.envelope_params is not None:
            # die raeumliche Kohaerenz haengt an Defokus und Cs (in nm)
            defocus_nm = aberrations.get("C10")
            if defocus_nm is not None:
                low, high = ABERRATION_CLAMPS["C10"]
                defocus_nm = torch.clamp(defocus_nm, min=low, max=high)
            cs_nm = aberrations.get("C30")
            if cs_nm is not None:
                low, high = ABERRATION_CLAMPS["C30"]
                cs_nm = torch.clamp(cs_nm, min=low, max=high) * ABERRATION_TERMS["C30"][2]

            env = build_ctf_envelope_on_k(kx, ky, self.voltage_kV, self.lam, self.envelope_params,
                                          deltaE=deltaE, Cc_um=Cc_um, cache=envelope_cache,
                                          defocus_nm=defocus_nm, cs_nm=cs_nm)
            ctf = ctf * env.to(ctf.device).type(ctf.dtype)
        # ctf = torch.fft.fftshift(ctf)
        return ctf, env

    def forward(self, input):
        wave = input[0]
        ctf, _ = self._build_ctf()  # Berechne CTF mit Envelope

        wave_ft = torch.fft.fft2(wave)
        wave_out_ft = wave_ft * ctf * self.aperture


        wave_out = torch.fft.ifft2(wave_out_ft)
        pad = self.pad
        return wave_out.abs()[pad:self.H+pad, pad:self.W+pad], input[1]

    def to_diffraction_patterns(self, max_angle_mrad=None, gpts=None):
        """
        Erzeuge ein abTEM-ähnliches 'Diffraction Pattern' der CTF:
        - baue ein Winkelgitter bis max_angle_mrad,
        - k = alpha / lambda (alpha in rad, lambda in nm) → k in 1/nm,
        - berechne CTF(kx,ky) (inkl. Envelope),
        - gib FFT-shifted Array zurück.
        """
        # Default-Logik wie in abTEM: wenn max_angle nicht angegeben und kein Cutoff bekannt → Fehler




        if max_angle_mrad is None:
            if self.semiangle_cutoff_mrad is not None:
                max_angle_mrad = float(self.semiangle_cutoff_mrad)
            else:
                raise RuntimeError("max_angle_mrad muss angegeben werden (kein semiangle_cutoff gesetzt).")

        # gpts-Default wie abTEM (128), erlaubt auch Tuple
        if gpts is None:
            gpts = 128
        if isinstance(gpts, int):
            gH = gW = gpts
        else:
            gH, gW = gpts

        # Winkelgitter (mrad → rad)
        alpha_x = torch.linspace(-max_angle_mrad, max_angle_mrad, gW, device=self.device) * 1e-3
        alpha_y = torch.linspace(-max_angle_mrad, max_angle_mrad, gH, device=self.device) * 1e-3
        ax, ay = torch.meshgrid(alpha_x, alpha_y, indexing='xy')

        # k-Gitter in 1/nm: k = alpha / lambda
        lam_nm = self.lam
        kx = ax / lam_nm
        ky = ay / lam_nm

        k = torch.sqrt(kx ** 2 + ky ** 2)

        # maximal zugelassener Frequenzradius
        k_max = (self.alpha_max_mrad * 1e-3) / self.lam  # [1/nm]

        aperture = (k <= k_max).float()

        # CTF auf diesem Gitter
        ctf, env = self._build_ctf(kx, ky)
        ctf = ctf * aperture

        # FFT-shift wie abTEM
        #ctf = torch.fft.fftshift(ctf)

        k_ring = 5.5  # [1/nm]
        delta_k = 0.1  # Ringbreite, Toleranz

        ring_mask = ((k > (k_ring - delta_k)) & (k < (k_ring + delta_k))).float()

        psf = torch.fft.fftshift(torch.fft.fft2(ctf))

        # Rückgabe: (array, sampling_info) analog – hier minimal:
        # In abTEM wird auch Sampling/Metadaten gesetzt; du kannst das bei Bedarf ergänzen.
        return ctf, env, ring_mask, psf



if __name__ == "__main__":
    H, W = 512, 512
          # 0.1 nm / Pixel (1 Å/Pixel)
    voltage_kV = 80.         # 300 kV TEM
    df_nm = -57               # Defokus: 1500 nm = 1.5 µm
    Cs_nm = 5.0e3

    pixel_size_nm = 4/512

    semiangle = 42  # 42 mrad
    zero_env = False
    Cc = 1.0
    dE = 0.35

    ccv = 0.0 if zero_env else Cc
    spv = 0.0 if zero_env else dE
    focal_spread = (ccv * 1e7) * spv / 80e3 if ccv > 0 else 0.0


    #ctf_layer = CTFLayer(H, W, pixel_size_nm, voltage_kV,
    #                     df_nm, Cs_nm, amp_contrast)

    aberrations = {'C10': -57., 'C12': 100.0, 'C23': 3.0e3,
                   'C21': 50.0e3, 'C30': 5.0e6, 'C34': 20.0e3, 'C32': 50.0e3}
    aberrations = {"C10": -57.,
                   'C12': 1.0, "phi12": 0.,
                    'C23': .5, 'phi23': 0.,
                    'C21': .5, 'phi21': 0.,
                    "C30": 2.}

    #aberrations = {'C10': -57.,'C30': 5.0e3}

    env_params = {
        'source_size': .1e-3,  # rad
        'Cc': Cc,  # nm
        'deltaE': dE,  # eV
        'instab_defocus': 0.5,  # nm
        'use_detector': False,
    }

    ctf_layer = CTFLayer(H=512, W=512, pixel_size_nm=pixel_size_nm,
                         voltage_kV=voltage_kV,
                         amp_contrast=0.5, device='cpu',
                         envelope_params=env_params,
                        aberrations=aberrations,
                         semiangle_cutoff_mrad=semiangle)

    dp = ctf_layer.to_diffraction_patterns(max_angle_mrad=semiangle, gpts=H)

    ctf_map = dp.detach().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(8, 8))

    axs[0].imshow(np.angle(ctf_map), cmap='RdBu', vmin=-np.pi, vmax=np.pi)
    #axs[0].colorbar()
    axs[0].set_title(f"CTF at {voltage_kV} kV, df={df_nm} nm, Cs={Cs_nm/1e6} mm")







    ctf = abtem.CTF(
        energy=80e3,
        aberration_coefficients=aberrations,
        semiangle_cutoff=semiangle,
        focal_spread=focal_spread,
        angular_spread=1.0,
        sampling=pixel_size_nm*10
    )

    # Compute diffraction pattern and phase
    #pattern = ctf.to_diffraction_patterns().array

    pattern = ctf.to_diffraction_patterns(
        max_angle=ctf.semiangle_cutoff * 1.01,
        gpts=512
    ).array
    phase = np.angle(pattern)

    axs[1].imshow(phase, cmap='RdBu', vmin=-np.pi, vmax=np.pi)

    plt.show()