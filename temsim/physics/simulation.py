r"""Part I, Section 9: the complete forward model.

Theory
======

The chain of the preceding sections is evaluated for one *state* of the
specimen (a list of :class:`~temsim.physics.structure.AtomLayer`) and one state
of the microscope (:class:`OpticsState`):

.. math::

    \text{atoms}
    \xrightarrow{(4.1),(6.5)} V^{\mathrm{proj}}_\ell
    \xrightarrow{(6.2),(6.3)} \psi_{\mathrm{exit}}
    \xrightarrow{(7.4),(8.1)} I_{\mathrm{TEM}},
    \qquad
    \text{atoms}
    \xrightarrow{(4.1),(8.4)} O
    \xrightarrow{(8.2)} I_{\mathrm{HAADF}}.
    \qquad (9.1)

Everything is a differentiable tensor operation, so the derivative of any scalar
function of the image with respect to atom positions, weights, aberrations or
envelope parameters is obtained by back-propagation (Part II).

The simulator owns only *fixed* quantities (grids, kernels, caches).  Parameters
enter through the arguments, never as hidden state; the same object therefore
serves the full simulation, the training loop and the local window update
(:mod:`temsim.physics.window`) without duplicating any physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import torch

from . import electron, imaging, optics
from .multislice import Multislice, slice_potential
from .optics import Envelope, FourierGrid
from .scattering import KernelBank
from .structure import AtomLayer, element_densities


@dataclass
class OpticsState:
    """Microscope parameters that are fitted: aberrations, amplitude contrast, energy spread, Cc.

    ``aberrations`` are the coefficients used for the *image-forming* lens in TEM
    and for the *probe* in STEM (the caller chooses which set to pass).
    """

    aberrations: Mapping[str, object] = field(default_factory=dict)
    amp_contrast: object = 0.1
    delta_E_eV: object = 0.35
    Cc_um: object = 1500.0


class Simulator:
    """Forward model for a fixed :class:`~temsim.data.schema.Imaging` setup."""

    def __init__(self, imaging_setup, device="cpu"):
        self.setup = imaging_setup
        self.device = device
        self.shape = tuple(imaging_setup.shape)
        self.padding = int(imaging_setup.padding)
        self.padded_shape = (self.shape[0] + 2 * self.padding, self.shape[1] + 2 * self.padding)
        self.pixel_size_nm = float(imaging_setup.pixel_size_nm)
        self.voltage_kV = float(imaging_setup.voltage_kV)
        self.wavelength_nm = electron.wavelength_nm(self.voltage_kV)
        self.stem_mode = imaging_setup.mode == "stem_haadf"
        self.reverse = imaging_setup.beam_direction == "-z"

        self.bank = KernelBank(self.padded_shape, self.pixel_size_nm, imaging_setup.potential_parametrization,
                               device=device)
        self.multislice = Multislice(self.padded_shape, self.pixel_size_nm, self.voltage_kV,
                                     band_limit=imaging_setup.band_limit, device=device)
        self.grid = FourierGrid(self.padded_shape, self.pixel_size_nm, self.wavelength_nm, device)
        self.envelope_settings = Envelope.from_dict(imaging_setup.envelope)
        self.objective_aperture = optics.aperture(self.grid, imaging_setup.alpha_max_mrad)

        if self.stem_mode:
            stem = imaging_setup.stem
            self.probe_aperture = optics.aperture(self.grid, stem.convergence_mrad)
            self.cross_sections = imaging.CrossSections(*stem.detector_mrad, self.voltage_kV,
                                                        imaging_setup.potential_parametrization)

    # -- specimen -----------------------------------------------------------------
    def potentials(self, layers: Sequence[AtomLayer]) -> List[Optional[torch.Tensor]]:
        """Eq. (6.5) per layer (``None`` for layers that do not contribute)."""

        return [slice_potential(layer, self.bank, self.shape, self.padding) if layer.active else None
                for layer in layers]

    def exit_wave(self, layers: Sequence[AtomLayer], potentials=None) -> torch.Tensor:
        potentials = self.potentials(layers) if potentials is None else potentials
        return self.multislice.exit_wave(potentials, [layer.z_nm for layer in layers], reverse=self.reverse)

    # -- microscope ------------------------------------------------------------------
    def transfer(self, state: OpticsState) -> torch.Tensor:
        """Eq. (7.4) including the envelopes, without the aperture."""

        coeffs = state.aberrations
        cs = coeffs.get("C30")
        env = optics.envelope(self.grid, self.envelope_settings, self.voltage_kV, state.delta_E_eV,
                              state.Cc_um, defocus_nm=coeffs.get("C10"),
                              cs_nm=None if cs is None else cs * 1e3)
        return optics.transfer_function(self.grid, coeffs, state.amp_contrast, env)

    def lens_spectrum(self, state: OpticsState) -> torch.Tensor:
        """:math:`T\\,A` of Eq. (7.1)."""

        return self.transfer(state) * self.objective_aperture

    def probe_intensity(self, state: OpticsState) -> torch.Tensor:
        """Eq. (8.3), FFT order."""

        stem = self.setup.stem
        chi = optics.phase_error(self.grid, state.aberrations)
        return imaging.probe_intensity(chi, self.probe_aperture, stem.source_size_A,
                                       self.pixel_size_nm * 10.0)

    # -- images ----------------------------------------------------------------------
    def object_map(self, layers: Sequence[AtomLayer]) -> torch.Tensor:
        """Eq. (8.2): cross-section density, contrast factors included."""

        densities = []
        for layer in layers:
            if not layer.active:
                densities.append(None)
                continue
            densities.append({e: rho * layer.contrast
                              for e, rho in element_densities(layer, self.shape, self.padding).items()})
        return imaging.object_map(densities, [layer.debye_waller_u_A for layer in layers],
                                  self.cross_sections, self.padded_shape, self.device)

    def image(self, layers: Sequence[AtomLayer], state: OpticsState) -> torch.Tensor:
        """Simulated image of shape ``Imaging.shape``."""

        if self.stem_mode:
            return imaging.stem_image(self.object_map(layers), self.probe_intensity(state),
                                      self.shape, self.padding)
        wave = self.exit_wave(layers)
        return imaging.tem_image(wave, self.transfer(state), self.objective_aperture, self.shape,
                                 self.padding, self.setup.tem_signal)
