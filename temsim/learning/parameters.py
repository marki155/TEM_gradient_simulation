r"""Part II, Section 1: the parameters of the model.

Theory
======

Learning as inverse problem
---------------------------
Part I defines a forward map :math:`\mathcal F` from the *state* of specimen and
microscope to a simulated image :math:`I_{\rm sim}`.  A measurement
:math:`I_{\rm exp}` is given; the reconstruction looks for the state
:math:`\theta` that minimises a scalar objective,

.. math::

    \theta^\ast = \arg\min_\theta\;
    \mathcal L_{\rm data}\big(\mathcal F(\theta), I_{\rm exp}\big)
    + \mathcal R(\theta).
    \qquad (1.1)

:math:`\mathcal L_{\rm data}` (Section 2) measures the misfit, :math:`\mathcal R`
(Section 3) encodes prior knowledge that the image alone cannot supply (atoms are
either present or absent, neighbours do not overlap, a crystal is only gently
strained).  Section 4 explains how (1.1) is minimised, Section 5 how atoms are
added to and removed from the model - a *discrete* change that gradient descent
cannot make.

Which numbers are free
----------------------
The state is a list of tensors, each with a fixed meaning.  For atom :math:`i` of
layer :math:`\ell`

* a position shift :math:`\delta\mathbf r_{\ell i}` added to its initial position
  :math:`\mathbf r^0_{\ell i}`, Eq. (3.2) of Part III,
* an *intensity* :math:`I_{\ell i}`, a continuous atomic number that starts at
  :math:`I^0_{\ell i}` (the atomic number of the layer's element, so that the
  quotient below is 1 at the start),
* an occupancy logit :math:`o_{\ell i}`,

and for each group :math:`g`: a translation :math:`\mathbf t_g`, a rotation angle
:math:`\vartheta_g`, an anisotropic scale :math:`\mathbf s_g`, a contrast factor
:math:`c_g` and the control nodes :math:`\mathbf U_g` of the strain field of
Part I, Section 5.  Globally: the aberration coefficients
:math:`\{C_{nm}, \varphi_{nm}\}` (or their Cartesian form), the amplitude contrast
:math:`Q`, the energy spread :math:`\Delta E` and chromatic aberration
:math:`C_c`, and two numbers :math:`(n_0, n_1)` that adjust brightness and
contrast of the simulation (Section 2).

Effective atoms
---------------
The step from these tensors to the atoms that Part I simulates is

.. math::

    \mathbf r^{\rm eff}_{\ell i}
      = \mathcal S_g\!\Big(\mathbf c + \mathbf s_g \odot
        R(\vartheta_g)\,(\mathbf r^0_{\ell i} + \delta\mathbf r_{\ell i} - \mathbf c)
        + \mathbf t_g\Big),
    \qquad
    w_{\ell i} = \frac{I_{\ell i}}{I^0_{\ell i}}\;\sigma(o_{\ell i}),
    \qquad (1.2)

with :math:`\mathbf c` the image centre, :math:`\mathcal S_g` the strain
displacement (Part I, Eq. 5.1) and the logistic function
:math:`\sigma(o) = 1/(1+e^{-o})`.  The weight :math:`w` is the occupancy times the
*relative* scattering strength (Part I, Eq. 4.3).  In mixed layers (several
elements) the intensity does not scale the amplitude but selects the element
(Part I, Eq. 4.4) and :math:`w = \sigma(o)`.

Why a logistic occupancy?  An atom is either present or not, a binary decision.
The logit makes the decision differentiable; the entropy prior of Section 3
drives :math:`\sigma(o)` towards 0 or 1, and atoms with :math:`\sigma(o) < 1/2`
are removed from the model when it is *synchronised* (Section 5).

Bounds
------
Some parameters leave the region where the physics is meaningful when the
optimiser takes a large step.  The following bounds are applied *where the
parameter is used*, never to the stored value, so that a bound does not erase
information and the gradient vanishes outside it (as for any clamp):

* rotation :math:`\vartheta_g` within :math:`\pm` ``max_rotation`` and translation
  :math:`\mathbf t_g` within :math:`\pm` ``max_shift``, if the group sets them;
* scale :math:`\mathbf s_g` within [0.8, 1.2], if it is trained;
* defocus :math:`C_{10}` within :math:`\pm 1000` nm and spherical aberration
  :math:`C_{30}` within :math:`\pm 20` µm, always;
* energy spread within [0.25, 0.45] eV and :math:`C_c` within [100, 10000] µm, if
  trained;
* the amplitude contrast within [0, 1], always (Part I, Eq. 7.4).

Implementation
==============

:class:`ParameterSet` owns every trainable tensor as a separate *leaf*.  Whether a
leaf is trained is decided by ``requires_grad`` (:meth:`ParameterSet.apply_flags`
reads the :class:`~temsim.data.schema.TrainSettings`); the earlier program
instead detached tensors inside the forward pass, which is equivalent but obscures
what the optimiser sees.  A frozen tensor is still *used*, with its stored value.

Atoms are added and removed through :meth:`ParameterSet.append_atoms` and
:meth:`ParameterSet.remove_atoms`.  Both change tensor sizes, so the optimiser
state of the affected parameters is invalid afterwards; the trainer rebuilds the
optimiser (Section 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..data.schema import Experiment, Group, ModelState, Structure, TrainSettings
from ..data.schema import OpticsState as StoredOptics
from ..physics import optics, strain
from ..physics.simulation import OpticsState
from ..physics.structure import AtomLayer, atomic_number_of, relative_weights, sorted_elements, transform

#: Bounds of Section 1 (applied at the point of use).
SCALE_RANGE = (0.8, 1.2)
DEFOCUS_RANGE_NM = (-1000.0, 1000.0)
SPHERICAL_RANGE_UM = (-20.0, 20.0)
DELTA_E_RANGE_EV = (0.25, 0.45)
CC_RANGE_UM = (1.0e2, 1.0e4)

_DEFAULT_DELTA_E = 0.45
_DEFAULT_CC_UM = 1000.0


def _leaf(values, device, requires_grad: bool = False) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(values, dtype=np.float32), device=device).clone()
    return t.requires_grad_(requires_grad)


@dataclass
class LayerParams:
    """Tensors of one layer (``n`` atoms)."""

    name: str
    #: initial positions r0, (n, 2), never trained
    base: torch.Tensor
    #: I0, (n,), never trained
    reference: torch.Tensor
    #: delta r, (n, 2)
    shifts: torch.Tensor
    #: I, (n,)
    intensity: torch.Tensor
    #: o, (n,)
    logit: torch.Tensor

    def __len__(self) -> int:
        return int(self.base.shape[0])


@dataclass
class GroupParams:
    """Tensors of one group."""

    name: str
    shift: torch.Tensor          # (2,)
    rotation: torch.Tensor       # ()
    scale: torch.Tensor          # (2,)
    contrast: torch.Tensor       # ()
    strain: torch.Tensor         # (1, 2, Hc, Wc)


class ParameterSet:
    """All trainable tensors of one reconstruction and the map to effective atoms."""

    def __init__(self, experiment: Experiment, state: Optional[ModelState] = None, device="cpu"):
        self.experiment = experiment
        self.device = device
        self.shape = tuple(experiment.imaging.shape)
        self.parametrization = experiment.imaging.ctf_parametrization
        self.layers: List[LayerParams] = []
        self.groups: Dict[str, GroupParams] = {}
        self.aberrations: Dict[str, torch.Tensor] = {}
        self.amp_contrast = torch.zeros((), device=device)
        self.delta_E = torch.zeros((), device=device)
        self.Cc_um = torch.zeros((), device=device)
        self.normalisation = torch.zeros(2, device=device)
        self._build(state or ModelState())

    # ------------------------------------------------------------------ building
    def _build(self, state: ModelState) -> None:
        exp, device = self.experiment, self.device
        structure = exp.structure
        for i, layer in enumerate(exp.layers):
            base = np.asarray(structure.positions[i], dtype=np.float64).reshape(-1, 2)
            n = len(base)
            reference = np.asarray(structure.atomic_numbers[i], dtype=np.float64).reshape(-1)
            factors = None if structure.weight_factors is None else structure.weight_factors[i]
            intensity = reference * (1.0 if factors is None else np.asarray(factors).reshape(-1))
            shifts = np.zeros((n, 2))
            logit = np.full(n, float(layer.init_occupancy))
            for source, target in ((state.position_shifts, "shifts"), (state.intensities, "intensity"),
                                   (state.occupancies, "logit")):
                if i < len(source) and len(source[i]) > 0:
                    value = np.asarray(source[i], dtype=np.float64)
                    if len(value) != n:
                        raise ValueError(f"state has {len(value)} {target} for layer '{layer.name}' "
                                         f"but the structure has {n} atoms")
                    if target == "shifts":
                        shifts = value.reshape(n, 2)
                    elif target == "intensity":
                        intensity = value.reshape(n)
                    else:
                        logit = value.reshape(n)
            self.layers.append(LayerParams(
                layer.name, _leaf(base, device), _leaf(reference, device), _leaf(shifts, device),
                _leaf(intensity, device), _leaf(logit, device)))

        for g in exp.groups:
            self.groups[g.name] = self._build_group(g, state)

        im = exp.imaging
        stored = state.optics
        if stored is not None and stored.aberrations:
            aberrations = dict(stored.aberrations)
        elif im.mode == "stem_haadf":
            aberrations = dict(im.stem.aberrations or optics.DEFAULT_PROBE_ABERRATIONS)
        else:
            aberrations = dict(im.aberrations or optics.DEFAULT_ABERRATIONS)
        aberrations = optics.convert(aberrations, self.parametrization)
        self.aberrations = {k: _leaf(float(v), device) for k, v in aberrations.items()}
        if stored is not None:
            amp, dE, Cc = stored.amp_contrast, stored.delta_E, stored.Cc_um
        else:
            amp = im.amp_contrast
            dE = im.envelope.get("deltaE", _DEFAULT_DELTA_E)
            Cc = im.envelope.get("Cc_um", _DEFAULT_CC_UM)
        self.amp_contrast = _leaf(float(amp), device)
        self.delta_E = _leaf(float(dE), device)
        self.Cc_um = _leaf(float(Cc), device)
        norm = np.zeros(2) if state.normalisation is None else np.asarray(state.normalisation, dtype=np.float64)
        self.normalisation = _leaf(norm.reshape(2), device)

    def _build_group(self, g: Group, state: ModelState) -> GroupParams:
        device = self.device
        shift = np.asarray(state.shifts.get(g.name, g.init_shift), dtype=np.float64).reshape(-1)
        shift = np.full(2, shift[0]) if shift.size == 1 else shift
        scale = np.asarray(state.scales.get(g.name, g.init_scale), dtype=np.float64).reshape(-1)
        scale = np.full(2, scale[0]) if scale.size == 1 else scale
        rotation = float(state.rotations.get(g.name, g.init_rotation))
        contrast = float(state.contrast_factors.get(g.name, g.init_contrast_factor))
        nodes = (int(g.control_points[0]), int(g.control_points[1]))
        control = state.strain_control.get(g.name)
        if control is None:
            field = torch.zeros((1, 2) + nodes)
        else:
            arr = np.asarray(control, dtype=np.float32)
            field = torch.as_tensor(arr.reshape(1, 2, *arr.shape[-2:]))
            if state.strain_convention == "legacy":
                field = strain.from_legacy_control(field, self.shape, self.experiment.imaging.padding)
            if tuple(field.shape[-2:]) != nodes:
                field = strain.resample_control(field, nodes)
        return GroupParams(g.name, _leaf(shift, device), _leaf(rotation, device), _leaf(scale, device),
                           _leaf(contrast, device), _leaf(field.numpy(), device))

    # ------------------------------------------------------------------ flags / registry
    def layer_index(self, name: str) -> int:
        return self.experiment.layer_names.index(name)

    def aberration_trainable(self, key: str, settings: TrainSettings) -> bool:
        """An aberration entry is trained if the global switch and the flag of its magnitude are on."""

        return bool(settings.aberrations.enabled and settings.aberration_flags.get(optics.magnitude_key(key), False))

    def tensors(self, key: str, settings: TrainSettings) -> List[torch.Tensor]:
        """Leaves belonging to a key of :meth:`TrainSettings.params`."""

        if key == "amp_contrast":
            return [self.amp_contrast]
        if key == "normalisation":
            return [self.normalisation]
        if key == "aberrations":
            return [t for k, t in self.aberrations.items() if self.aberration_trainable(k, settings)]
        if key == "envelope":
            out = []
            if settings.envelope_flags.get("deltaE", False):
                out.append(self.delta_E)
            if settings.envelope_flags.get("Cc", False):
                out.append(self.Cc_um)
            return out
        kind, name, attr = key.split(".")
        if kind == "groups":
            g = self.groups[name]
            return [{"shift": g.shift, "rotation": g.rotation, "scale": g.scale, "contrast": g.contrast,
                     "strain_field": g.strain}[attr]]
        lp = self.layers[self.layer_index(name)]
        return [{"free_positions": lp.shifts, "intensities": lp.intensity, "occupancies": lp.logit}[attr]]

    def all_leaves(self) -> List[torch.Tensor]:
        out = [self.amp_contrast, self.delta_E, self.Cc_um, self.normalisation, *self.aberrations.values()]
        for g in self.groups.values():
            out += [g.shift, g.rotation, g.scale, g.contrast, g.strain]
        for lp in self.layers:
            out += [lp.shifts, lp.intensity, lp.logit]
        return out

    def apply_flags(self, settings: TrainSettings) -> None:
        """Set ``requires_grad`` of every leaf from the switches (Section 4)."""

        settings.ensure_for(self.experiment)
        for t in self.all_leaves():
            t.requires_grad_(False)
            t.grad = None
        for key, param in settings.params().items():
            if not param.enabled:
                continue
            for t in self.tensors(key, settings):
                t.requires_grad_(True)

    # ------------------------------------------------------------------ forward quantities
    def optics_state(self) -> OpticsState:
        """Microscope state for Part I with the bounds of Section 1."""

        coeffs = {}
        for k, v in self.aberrations.items():
            if k == "C10":
                v = torch.clamp(v, *DEFOCUS_RANGE_NM)
            elif k == "C30":
                v = torch.clamp(v, *SPHERICAL_RANGE_UM)
            coeffs[k] = v
        dE = torch.clamp(self.delta_E, *DELTA_E_RANGE_EV) if self.delta_E.requires_grad else self.delta_E
        Cc = torch.clamp(self.Cc_um, *CC_RANGE_UM) if self.Cc_um.requires_grad else self.Cc_um
        return OpticsState(coeffs, torch.clamp(self.amp_contrast, 0.0, 1.0), dE, Cc)

    def rigid(self, group: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bounded ``(shift, rotation, scale)`` of a group, Eq. (1.2)."""

        g, spec = self.groups[group], self.experiment.group(group)
        rotation, shift, scale = g.rotation, g.shift, g.scale
        if spec.max_rotation is not None:
            rotation = torch.clamp(rotation, -spec.max_rotation, spec.max_rotation)
        if spec.max_shift is not None:
            shift = torch.clamp(shift, -spec.max_shift, spec.max_shift)
        if scale.requires_grad:
            scale = torch.clamp(scale, *SCALE_RANGE)
        return shift, rotation, scale

    def rigid_positions(self, index: int) -> torch.Tensor:
        """Positions of layer ``index`` after the rigid group transformation, before the strain field."""

        lp = self.layers[index]
        shift, rotation, scale = self.rigid(self.experiment.layers[index].group)
        return transform(lp.base + lp.shifts, self.shape, shift, rotation, scale)

    def positions(self, index: int) -> torch.Tensor:
        """Effective positions :math:`\\mathbf r^{\\rm eff}` of layer ``index``, Eq. (1.2)."""

        group = self.experiment.layers[index].group
        lp = self.layers[index]
        shift, rotation, scale = self.rigid(group)
        return transform(lp.base + lp.shifts, self.shape, shift, rotation, scale, self.groups[group].strain)

    def occupancy(self, index: int) -> torch.Tensor:
        return torch.sigmoid(self.layers[index].logit)

    def mixed(self, index: int) -> bool:
        return len(self.experiment.layers[index].elements) > 1

    def weights(self, index: int) -> torch.Tensor:
        """:math:`w_{\\ell i}` of Eq. (1.2)."""

        lp = self.layers[index]
        occupancy = self.occupancy(index)
        if self.mixed(index):
            return occupancy
        return relative_weights(lp.intensity, lp.reference, occupancy)

    def atom_layer(self, index: int, active: bool = True) -> AtomLayer:
        spec = self.experiment.layers[index]
        lp = self.layers[index]
        mixed = self.mixed(index)
        return AtomLayer(
            z_nm=spec.z_nm, element=spec.element, positions=self.positions(index), weights=self.weights(index),
            elements=sorted_elements(spec.element, spec.elements) if mixed else (),
            atomic_number=lp.intensity if mixed else None,
            debye_waller_u_A=spec.debye_waller_u,
            contrast=self.groups[spec.group].contrast, active=active)

    def effective_layers(self, active_group: Optional[str] = None) -> List[AtomLayer]:
        """The atom layers of Part I.

        While a group with ``isolate_when_active`` is active only its layers
        contribute (the others are switched off, not deleted).
        """

        isolate = (active_group is not None and active_group in self.groups
                   and self.experiment.group(active_group).isolate_when_active)
        return [self.atom_layer(i, active=(not isolate) or spec.group == active_group)
                for i, spec in enumerate(self.experiment.layers)]

    # ------------------------------------------------------------------ editing the atom list
    def append_atoms(self, index: int, positions, intensity=None, logit: float = 6.0, shifts=None,
                     reference=None) -> None:
        """Add atoms at ``positions`` (initial frame, ``(m, 2)``) to layer ``index``.

        ``reference`` is :math:`I_0` (default: the atomic number of the layer element) and
        ``intensity`` is :math:`I` (default: ``reference``, i.e. relative weight 1).  The
        default logit 6 gives :math:`\\sigma \\approx 0.9975`.
        """

        lp = self.layers[index]
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        m = len(positions)
        if m == 0:
            return
        if reference is None:
            reference = atomic_number_of(self.experiment.layers[index].element)
        reference = np.broadcast_to(np.asarray(reference, dtype=np.float64), (m,))
        intensity = reference if intensity is None else np.broadcast_to(np.asarray(intensity, dtype=np.float64), (m,))
        requires = {k: getattr(lp, k).requires_grad for k in ("shifts", "intensity", "logit")}

        def cat(old, new):
            return torch.cat([old.detach(), _leaf(new, self.device)])

        lp.base = cat(lp.base, positions)
        lp.reference = cat(lp.reference, reference)
        lp.shifts = cat(lp.shifts, np.zeros((m, 2)) if shifts is None else shifts)
        lp.intensity = cat(lp.intensity, intensity)
        lp.logit = cat(lp.logit, np.full(m, float(logit)))
        for k, flag in requires.items():
            getattr(lp, k).requires_grad_(flag)

    def remove_atoms(self, index: int, keep) -> None:
        """Keep only the atoms where ``keep`` (boolean mask or index array) is true."""

        lp = self.layers[index]
        keep = torch.as_tensor(np.asarray(keep), device=self.device)
        if keep.dtype != torch.bool:
            mask = torch.zeros(len(lp), dtype=torch.bool, device=self.device)
            mask[keep.long()] = True
            keep = mask
        for k in ("base", "reference", "shifts", "intensity", "logit"):
            flag = getattr(lp, k).requires_grad
            setattr(lp, k, getattr(lp, k).detach()[keep].clone().requires_grad_(flag))

    # ------------------------------------------------------------------ export
    def to_state(self) -> ModelState:
        def np_(t):
            return t.detach().cpu().numpy().astype(np.float64)

        return ModelState(
            position_shifts=[np_(lp.shifts) for lp in self.layers],
            intensities=[np_(lp.intensity) for lp in self.layers],
            occupancies=[np_(lp.logit) for lp in self.layers],
            shifts={n: np_(g.shift) for n, g in self.groups.items()},
            rotations={n: float(g.rotation) for n, g in self.groups.items()},
            scales={n: np_(g.scale) for n, g in self.groups.items()},
            contrast_factors={n: float(g.contrast) for n, g in self.groups.items()},
            strain_control={n: np_(g.strain) for n, g in self.groups.items()},
            optics=StoredOptics(amp_contrast=float(self.amp_contrast), delta_E=float(self.delta_E),
                                Cc_um=float(self.Cc_um),
                                aberrations={k: float(v) for k, v in self.aberrations.items()}),
            normalisation=np_(self.normalisation))

    def to_structure(self) -> Structure:
        """Initial atoms as they are now (after seeding): base positions and reference numbers."""

        return Structure([lp.base.detach().cpu().numpy().astype(np.float64) for lp in self.layers],
                         [lp.reference.detach().cpu().numpy().astype(np.float64) for lp in self.layers])

    def counts(self) -> List[int]:
        return [len(lp) for lp in self.layers]
