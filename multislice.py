from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from physics_helper import relativistic_wavelength_nm


def electron_wavelength_nm(voltage_kV):
    h = 6.62607015e-34
    m0 = 9.1093837015e-31
    e = 1.602176634e-19
    c = 299792458.0

    U = voltage_kV * 1e3
    p = math.sqrt(2 * m0 * e * U * (1 + e * U / (2 * m0 * c * c)))
    lam_m = h / p
    return lam_m * 1e9

def sigma_electron(wavelength_m):
    m = 9.10938356e-31      # Elektronenmasse (kg)
    e = 1.602176634e-19     # Elementarladung (C)
    h = 6.62607015e-34      # Planck-Konstante (Js)

    sigma = (m * e * wavelength_m) / (h ** 2)
    return sigma

def interaction_parameter_rad_per_Vnm(voltage_kV):
    h = 6.62607015e-34
    m0 = 9.1093837015e-31
    e = 1.602176634e-19
    c = 299792458.0

    U = voltage_kV * 1e3
    E0 = m0 * c * c
    E = e * U

    p = math.sqrt(2 * m0 * E * (1 + E / (2 * E0)))
    lam_m = h / p
    gamma = 1.0 + E / E0

    sigma_SI_rad_Vm = 2 * math.pi * gamma * m0 * e * lam_m / (h * h)  # rad / (V*m)
    sigma_SI_rad_Vnm = sigma_SI_rad_Vm * 1e-9  # rad / (V*nm)
    return sigma_SI_rad_Vnm


def fresnel_propagator(H, W, dz_nm, pixel_size_nm, wavelength_nm, device=None):
    """Fresnel-Propagator im k-Raum. Haengt nur an der Geometrie, nicht an der Welle."""

    fy = torch.fft.fftfreq(H, d=pixel_size_nm, device=device)
    fx = torch.fft.fftfreq(W, d=pixel_size_nm, device=device)
    Y, X = torch.meshgrid(fy, fx, indexing="ij")
    k2 = X ** 2 + Y ** 2

    return torch.exp(-1j * math.pi * wavelength_nm * dz_nm * k2)


def propagate_fresnel(wave, dz_nm, pixel_size_nm, wavelength_nm, propagator=None):
    if propagator is None:
        propagator = fresnel_propagator(
            wave.shape[0], wave.shape[1], dz_nm, pixel_size_nm, wavelength_nm, wave.device)

    wave_ft = torch.fft.fft2(wave)
    return torch.fft.ifft2(wave_ft * propagator)




#: Prueft nach jedem Slice, ob die Phase den sinnvollen Bereich verlaesst.
#: Kostet pro Schritt eine GPU->CPU Kopie je Schicht - deshalb per Default aus.
CHECK_PHASE = False


class PhysicalPhaseObjectLayer(nn.Module):
    """Transmission durch beliebig viele Schichten inkl. Fresnel-Propagation."""

    def __init__(self, H, W, pixel_size_nm, voltage_kV, device, layers=None):
        super().__init__()
        self.H = H
        self.W = W
        self.pixel_size_nm = pixel_size_nm
        self.voltage_kV = voltage_kV
        self.device = device

        self.lambda_nm = electron_wavelength_nm(voltage_kV)
        self.sigma = interaction_parameter_rad_per_Vnm(voltage_kV)

        self.layers = list(layers or [])
        self.layer_names = [l.name for l in self.layers]
        self.atom_names_by_layer = [l.element for l in self.layers]

        #: Propagatoren je Schichtabstand - konstant ueber das ganze Training
        self._propagators = {}

    def get_propagator(self, dz_nm, device):
        key = round(float(dz_nm), 9)
        propagator = self._propagators.get(key)
        if propagator is None:
            propagator = fresnel_propagator(
                self.H, self.W, key, self.pixel_size_nm, self.lambda_nm, device)
            self._propagators[key] = propagator
        return propagator

    def forward(self, potential_slices, z_positions_nm, reverse=False):
        """
        potential_slices: Liste (H,W) - eine pro Schicht, in Config-Reihenfolge
        z_positions_nm:   z-Koordinate je Schicht in nm, gleiche Reihenfolge
        reverse:          Strahl laeuft von der hoechsten zur niedrigsten
                          z-Position (statt umgekehrt)

        Die Slices werden anhand ihrer eigenen z-Position sortiert, jede Schicht
        behaelt also ihren Abstand zur naechsten. (Frueher wurde die Liste vor
        dem Aufruf umgedreht, waehrend die z-Werte aufsteigend blieben - dadurch
        bekamen die Schichten die Abstaende der jeweils anderen Seite.)
        """

        if len(potential_slices) != len(z_positions_nm):
            raise ValueError(
                f"{len(potential_slices)} Potential-Slices, aber "
                f"{len(z_positions_nm)} z-Positionen.")

        full_order = sorted(range(len(z_positions_nm)), key=lambda i: z_positions_nm[i])
        if reverse:
            full_order = full_order[::-1]

        # Schichten ohne Potential (None, z.B. isolierte Gruppe) ueberspringen:
        # ihre Transmission waere die Identitaet. Die Propagationsstrecke bleibt
        # erhalten - die Austrittsebene liegt weiterhin am Ende des Stapels.
        order = [i for i in full_order if potential_slices[i] is not None]
        if not order:
            return torch.ones((self.H, self.W), dtype=torch.complex64, device=self.device)

        wave = torch.ones((self.H, self.W), dtype=torch.complex64, device=self.device)

        for k, idx in enumerate(order):
            V_slice = potential_slices[idx]

            phase = self.sigma * V_slice

            if CHECK_PHASE:
                phase_np = phase.detach().cpu().numpy()
                mask_outside = (phase_np < 2 * -np.pi) | (phase_np > 2 * np.pi)
                if mask_outside.any():
                    name = self.layer_names[idx] if idx < len(self.layer_names) else idx
                    print(f"PHASE too high (Schicht {name}):")
                    print(phase_np[mask_outside])
                    print(np.argwhere(mask_outside))

            T = torch.exp(1j * phase)
            wave = wave * T

            if k < len(order) - 1:
                dz_nm = abs(z_positions_nm[order[k + 1]] - z_positions_nm[idx])
                wave = propagate_fresnel(
                    wave=wave,
                    dz_nm=dz_nm,
                    pixel_size_nm=self.pixel_size_nm,
                    wavelength_nm=self.lambda_nm,
                    propagator=self.get_propagator(dz_nm, wave.device),
                )

        # Reststrecke von der letzten aktiven Schicht bis zum Ende des Stapels
        # (Fresnel-Propagatoren multiplizieren sich, P(a)*P(b) = P(a+b), das
        # Ergebnis ist also identisch zum Durchlaufen leerer Schichten)
        dz_exit = abs(z_positions_nm[full_order[-1]] - z_positions_nm[order[-1]])
        if dz_exit > 0:
            wave = propagate_fresnel(
                wave=wave,
                dz_nm=dz_exit,
                pixel_size_nm=self.pixel_size_nm,
                wavelength_nm=self.lambda_nm,
                propagator=self.get_propagator(dz_exit, wave.device),
            )

        return wave


class DifferentiableMultislice(nn.Module):
    def __init__(self, config, train_method, device):
        super().__init__()

        self.layers = list(config.layers)
        self.z_positions_nm = [float(l.z_nm) for l in self.layers]
        self.reverse = str(getattr(config, "beam_direction", "+z")).strip() == "-z"

        self.pixel_size_nm = config.pixel_size_nm
        self.wavelength_nm = relativistic_wavelength_nm(config.voltage_kV)
        self.device = device

        self.H, self.W = config.shape

        self.train_method = train_method
        pad = config.padding
        self.phase_object_layer = PhysicalPhaseObjectLayer(
            self.H + 2 * pad, self.W + 2 * pad,
            config.pixel_size_nm, config.voltage_kV, device,
            layers=self.layers,
        )

    def forward(self, input):
        potential_list, pot_aux_loss = input

        exit_wave = self.phase_object_layer(
            potential_slices=potential_list,
            z_positions_nm=self.z_positions_nm,
            reverse=self.reverse,
        )

        return exit_wave, pot_aux_loss
