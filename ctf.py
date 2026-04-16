from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np


def scherzer_defocus(C30_um, lam_nm):
    C30_nm = C30_um*1000.
    return - 1.2 * math.sqrt(C30_nm * lam_nm)

def build_ctf_envelope_on_k(kx, ky, voltage_kV, lam, params=None, deltaE=None, Cc_um=None):
    # gleiche Gauß-Envelopes wie zuvor; minimal gehalten
    if params is None:
        params = {}
    k2 = kx**2 + ky**2
    k = torch.sqrt(k2 + 1e-30)
    eV = voltage_kV * 1e3

    # Spatial (source size / convergence)  [rad]
    E_spatial = 1.0
    if params.get('use_spatial', True):
        alpha = float(params.get('source_size', 0.0))
        E_spatial = torch.exp(- (math.pi * alpha * k)**2)

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
    E_inst = 1.0
    if params.get('use_instability', True):
        sigma_df_nm = float(params.get('instab_defocus_nm', 0.0))
        if sigma_df_nm > 0:
            E_inst = torch.exp(- (math.pi * lam * sigma_df_nm * k2)**2)
        else:
            E_inst = torch.ones_like(k)

    # Detector (optional, Gaussian in k)
    E_det = 1.0
    if params.get('use_detector', False):
        sigma_k = params.get('detector_sigma_k', None)
        if sigma_k is not None:
            sigma_k = float(sigma_k)
            E_det = torch.exp(-(k**2) / (2.0 * sigma_k**2))
        else:
            E_det = torch.ones_like(k)

    env = torch.clamp(E_spatial * E_chrom * E_inst * E_det, 0.0, 1.0)
    return env





def zernike_phase(alpha, phi, coeffs, device='cpu'):
    """
    Berechnet die Zernike-Phasenfunktion für gegebene Aberrationskoeffizienten.

    Parameters:
    - alpha: Radialer Abstand im k-Raum (Tensor, shape HxW)
    - phi: Azimutwinkel im k-Raum (Tensor, shape HxW)
    - coeffs: dict mit Aberrationen, z.B. {'C10': ..., 'C12': ..., 'phi12': ...}

    Returns:
    - complex CTF Tensor: exp(-i * chi(alpha, phi))
    """
    phase = torch.zeros_like(alpha, dtype=torch.float32, device=device)

    # ---- 2. Ordnung ----
    if "C10" in coeffs:  # Defokus
        phase += 0.5 * torch.clamp(coeffs["C10"], min=-1000., max=1000.) * alpha**2
    if "C12" in coeffs and "phi12" in coeffs:  # Astigmatismus 2. Ordnung
        phase += 0.5 * coeffs["C12"] * alpha**2 * torch.cos(2 * (phi - coeffs["phi12"]))

    # ---- 3. Ordnung ----
    if "C21" in coeffs and "phi21" in coeffs:  # Koma
        phase += (1/3) * coeffs["C21"] * alpha**3 * torch.cos(phi - coeffs["phi21"])
    if "C23" in coeffs and "phi23" in coeffs:  # 3-fach Astigmatismus
        phase += (1/3) * coeffs["C23"] * alpha**3 * torch.cos(3 * (phi - coeffs["phi23"]))

    # ---- 4. Ordnung ----
    if "C30" in coeffs:  # Sphärische Aberration
        phase += 0.25 * torch.clamp(coeffs["C30"], min=-20.0, max=20.0) * 1e3 * alpha**4
    if "C32" in coeffs and "phi32" in coeffs:
        phase += 0.25 * coeffs["C32"] * alpha**4 * torch.cos(2 * (phi - coeffs["phi32"]))
    if "C34" in coeffs and "phi34" in coeffs:
        phase += 0.25 * coeffs["C34"] * 1e3 *  alpha**4 * torch.cos(4 * (phi - coeffs["phi34"]))

    # ---- 5. Ordnung ----
    if "C41" in coeffs and "phi41" in coeffs:
        phase += 0.2 * coeffs["C41"] * alpha**5 * torch.cos(phi - coeffs["phi41"])
    if "C43" in coeffs and "phi43" in coeffs:
        phase += 0.2 * coeffs["C43"] * alpha**5 * torch.cos(3 * (phi - coeffs["phi43"]))
    if "C45" in coeffs and "phi45" in coeffs:
        phase += 0.2 * coeffs["C45"] * 1e3 * alpha**5 * torch.cos(5 * (phi - coeffs["phi45"]))

    # ---- 6. Ordnung ----
    if "C50" in coeffs:  # Sphärisch 2. Ordnung
        phase += (1/6) * coeffs["C50"] * alpha**6
    if "C52" in coeffs and "phi52" in coeffs:
        phase += (1/6) * coeffs["C52"] * alpha**6 * torch.cos(2 * (phi - coeffs["phi52"]))
    if "C54" in coeffs and "phi54" in coeffs:
        phase += (1/6) * coeffs["C54"] * alpha**6 * torch.cos(4 * (phi - coeffs["phi54"]))
    if "C56" in coeffs and "phi56" in coeffs:
        phase += (1/6) * coeffs["C56"] * alpha**6 * torch.cos(6 * (phi - coeffs["phi56"]))

    return phase

def make_param_dict(a=None):
    if a is None:
        a = {"C10": -13.07,
                               'C12': 0.35, "phi12": 0.,
                               'C23': 0.17, 'phi23': 0.,
                               'C21': 3.71, 'phi21': 0.,
                               "C30": 4.16, #um
                               'C34': -0.2, 'phi34': 0., #um
                               'C45': 0.25, 'phi45': 0.}

    return nn.ParameterDict({
        k: nn.Parameter(torch.tensor(v, dtype=torch.float32))
        for k, v in a.items()
    })

@dataclass
class TunableCTFParameters:
    amp_contrast: nn.Parameter = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    deltaE: nn.Parameter = nn.Parameter(torch.tensor(0.45, dtype=torch.float32))
    Cc_um: nn.Parameter = nn.Parameter(torch.tensor(1.e3, dtype=torch.float32))
    aberrations: nn.ParameterDict = field(default_factory=make_param_dict)#um



class CTFLayer(nn.Module):
    def __init__(self, config, train_method=None,device=None):
        super().__init__()
        self.H, self.W = config.shape
        self.pixel_size_nm = config.pixel_size_nm
        self.voltage_kV = config.voltage_kV
        #self.amp_contrast = nn.Parameter(torch.tensor(amp_contrast, device=device))
        self.envelope_params = config.init_ctf_env_params
        #self.deltaE = nn.Parameter(torch.tensor(envelope_params["deltaE"], device=device))
        #self.Cc_um = nn.Parameter(torch.tensor(envelope_params["Cc_um"], device=device))
        self.lam = config.wavelength_nm
        self.semiangle_cutoff_mrad = None
        self.alpha_max_mrad = config.ctf_alpha_max_mrad
        self.device = device
        self.train_method = train_method

        self.tunable_parameters = TunableCTFParameters(
            amp_contrast=nn.Parameter(torch.tensor(config.init_amp_contrast, dtype=torch.float32)),
            deltaE = nn.Parameter(torch.tensor(self.envelope_params["deltaE"], dtype=torch.float32)),
            Cc_um = nn.Parameter(torch.tensor(self.envelope_params["Cc_um"], dtype=torch.float32)),
            aberrations = nn.ParameterDict(make_param_dict(config.init_aberrations))
        )

        self.tunable_parameters = TunableCTFParameters()

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



    def _build_ctf(self, kx=None, ky=None):
        if kx is None:
            kx = self.kx
        if ky is None:
            ky = self.ky

        aberrations = {}
        for k, v in self.tunable_parameters.aberrations.items():
            if self.train_method is not None and self.train_method.train_ctf_aberrations.enabled:
                if k in self.train_method.train_aberrations and self.train_method.train_aberrations[k]:
                    aberrations[k] = v
                else:
                    aberrations[k] = v.detach()  # nicht trainierbar
            else:
                aberrations[k] = v.detach()
        #aberrations = {k: v for k, v in self.aberr_params.items()}
        lam = self.lam  # in nm

        #kx = ax / lam_nm
        #ky = ay / lam_nm


        alpha = self.lam * torch.sqrt(kx ** 2 + ky ** 2)  # Radialabstand
        alpha.to(device=self.device)
        phi = torch.atan2(ky, kx)
        phi.to(device=self.device)

        aberr_phase = zernike_phase(alpha, phi, aberrations, device=self.device)
        chi = 2.0 * math.pi * aberr_phase / lam  # klassische TEM-Phase
        if self.train_method is not None and self.train_method.train_amplitude_contrast:
            Q = torch.clamp(self.tunable_parameters.amp_contrast, 0.0, 1.0)
        else:
            Q = torch.clamp(self.tunable_parameters.amp_contrast.detach(), 0.0, 1.0)
        ac = math.sqrt(1.0 - Q ** 2) - 1j * Q
        ctf = ac * torch.exp(-1j * chi)

        if self.train_method.train_env_params["deltaE"] and self.train_method.train_env_parameters:
            deltaE = torch.clamp(self.tunable_parameters.deltaE, min=0.25, max=0.45)
        else:
            deltaE = self.tunable_parameters.deltaE.detach()

        if self.train_method.train_env_params["Cc"] and self.train_method.train_env_parameters:
            Cc_um = torch.clamp(self.tunable_parameters.Cc_um, min=1.e2, max=1.e4)
        else:
            Cc_um = self.tunable_parameters.Cc_um.detach()

        # zentrieren
        if self.envelope_params is not None:
            env = build_ctf_envelope_on_k(kx, ky, self.voltage_kV, self.lam, self.envelope_params, deltaE=deltaE,
                                          Cc_um=Cc_um)
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