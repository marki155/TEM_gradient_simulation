r"""Part II, Section 3: priors (regularisation).

Theory
======

The data term of Section 2 compares an image with an image.  The unknowns of
Section 1 are far more numerous than the pixels that constrain them, and some
combinations leave the image almost unchanged.  A *prior* adds the knowledge
that is not in the image: atoms are either there or not, they sit close to
their nominal sites, they do not overlap, and a crystal deforms smoothly.  In
the language of Bayes, minimising

.. math::

    \mathcal L = \mathcal L_{\rm data} + \mathcal L_{\rm prior}
    \qquad (3.1)

is the search for the maximum *a posteriori* estimate; every prior term is the
negative logarithm of a prior probability density, up to a constant.  The
weights below are therefore inverse variances in disguise and must be chosen
with the scale of the data term in mind (Section 2 standardises it, so the
weights are of order one or below).

Occupancy entropy
-----------------
With :math:`p_{\ell i} = \sigma(o_{\ell i})` the mean binary entropy

.. math::

    \mathcal L_{\rm ent} = \frac{1}{N_{\rm ent}}\sum_{\ell\in\mathcal E}
    \Big\langle -p\ln(p+\varepsilon) - (1-p)\ln(1-p+\varepsilon)\Big\rangle_i,
    \quad \varepsilon = 10^{-8},
    \qquad (3.2)

with :math:`\mathcal E` the layers that have the prior switched on, is
maximal at :math:`p = \tfrac12` and zero at :math:`p\in\{0,1\}`.  It pushes the
continuous relaxation of "atom present or absent" towards a decision; without it
the optimiser may keep many half-occupied atoms that fake a smooth density.

Position-shift norm
-------------------
.. math::

    \mathcal L_{\rm pos} = \frac{1}{N_{\rm free}}\sum_{\ell\in\mathcal F}
    \big\langle |\Delta\mathbf r_{\ell i}|\big\rangle_i
    \qquad (3.3)

(the mean absolute component), the sum over the layers whose positions are
trained.  An :math:`\ell_1` norm corresponds to a Laplace prior: most atoms
stay exactly at their site, a few may move far.

Repulsion
---------
Two occupied atoms of the same layer cannot be closer than a physical distance
:math:`d_0` (in pixels :math:`d_0 = d_{0,\text{\AA}}/(10\,\Delta x)`).  With
the effective positions :math:`\mathbf r` (before the strain field is applied,
see below) and the set :math:`P_\ell` of occupied pairs
(:math:`\sigma(o) > \tfrac12`) with distance :math:`d_{ij} < d_0`,

.. math::

    \mathcal L_{\rm rep} = \sum_{\ell\in\mathcal F}\frac{1}{n_\ell}
    \sum_{(i,j)\in P_\ell}\Big(1 - \frac{d_{ij}}{d_0}\Big)^2,
    \qquad (3.4)

with :math:`n_\ell` the number of occupied atoms.  The term is smooth (and zero
beyond :math:`d_0`).  The pair list is found with a k-d tree, without gradient;
only the distances are differentiable.  Eq. (3.4) is evaluated with
:math:`\mathbf r_0 + \Delta\mathbf r`, i.e. in the frame of the *initial*
lattice, because the repulsion belongs to the free positions
:math:`\Delta\mathbf r`.  A rigid group transformation or a smooth strain does not
change relative distances noticeably, and treating it this way keeps the term
independent of the other parameters.

Attraction and merging
----------------------
The repulsion keeps two atoms apart; a pair that is really *one* atom, seen twice, needs the
opposite.  For the occupied pairs closer than the merge distance :math:`d_m`

.. math::

    \mathcal L_{\rm att} = \sum_{\ell\in\mathcal F}\frac{1}{n_\ell}
    \sum_{(i,j):\,d_{ij}<d_m}\Big(\frac{d_{ij}}{d_m}\Big)^2 ,
    \qquad (3.4a)

which pulls the two towards each other (it is zero beyond :math:`d_m`; choose
:math:`d_m < d_0` of Eq. (3.4), else the two terms work against each other).  The attraction is
differentiable, but *fusing* two atoms into one is not: the number of atoms is an integer.  The
fusion is therefore a separate edit between two steps (``merge_close``,
:meth:`~temsim.learning.parameters.ParameterSet.merge_close`), by hand or by the training job
every ``merge_every`` steps; the attraction only brings the pairs into reach.

Elastic energy of the strain field
----------------------------------
For the field of Part I, Section 5 with strain tensor
:math:`\boldsymbol\varepsilon = \tfrac12(J + J^\top)`, :math:`J = \partial\mathbf u/\partial\mathbf r`,
the energy density of linear elasticity is Eq. (5.4),

.. math::

    w = \tfrac12\lambda\,(\operatorname{tr}\boldsymbol\varepsilon)^2
        + \mu\,\boldsymbol\varepsilon:\boldsymbol\varepsilon .

The prior is its mean over the atoms of the group,

.. math::

    \mathcal L_{\rm el} = \frac{1}{|\mathcal G_{\rm el}|}\sum_{g\in\mathcal G_{\rm el}}
    \Big\langle w\big(\mathbf r_{i}^{\rm rigid}\big)\Big\rangle_{i\in g},
    \qquad (3.5)

with the groups :math:`\mathcal G_{\rm el}` whose strain field is trained and the
atoms evaluated at their positions after the rigid transformation (the point
where the field is sampled in Eq. (5.1)).  A rigid rotation has no strain,
:math:`\boldsymbol\varepsilon = 0`, and costs nothing.  The earlier program
computed the second derivative inside the loss with a library routine that could
not be differentiated once more by the optimiser; the strain energy therefore
never had an effect.  The field of Part I is a pure tensor expression, so the
gradient exists (verified against an analytic case in the tests).

The total prior, with the weights of
:class:`~temsim.data.schema.Regularisation`, is

.. math::

    \mathcal L_{\rm prior} = a_{\rm ent}\mathcal L_{\rm ent} + a_{\rm pos}\mathcal L_{\rm pos}
    + a_{\rm rep}\mathcal L_{\rm rep} + a_{\rm att}\mathcal L_{\rm att} + a_{\rm el}\mathcal L_{\rm el}.
    \qquad (3.6)

A term with weight zero is not evaluated.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from scipy.spatial import cKDTree

from ..data.schema import TrainSettings
from ..physics import strain
from .parameters import ParameterSet


def entropy(ps: ParameterSet) -> torch.Tensor:
    """Eq. (3.2)."""

    values = []
    for i, spec in enumerate(ps.experiment.layers):
        if not spec.occ_entropy:
            continue
        p = torch.sigmoid(ps.layers[i].logit)
        values.append((-(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))).mean())
    return torch.stack(values).mean() if values else torch.zeros((), device=ps.device)


def position_norm(ps: ParameterSet, settings: TrainSettings) -> torch.Tensor:
    """Eq. (3.3)."""

    values = [ps.layers[i].shifts.abs().mean() for i, spec in enumerate(ps.experiment.layers)
              if settings.layer_enabled(spec.name, "free_positions") and len(ps.layers[i]) > 0]
    return torch.stack(values).mean() if values else torch.zeros((), device=ps.device)


def repulsion(ps: ParameterSet, settings: TrainSettings) -> torch.Tensor:
    """Eq. (3.4)."""

    d0 = float(settings.regularisation.repulsion_distance_A) / (ps.experiment.imaging.pixel_size_nm * 10.0)
    total = torch.zeros((), device=ps.device)
    for i, spec in enumerate(ps.experiment.layers):
        if not settings.layer_enabled(spec.name, "free_positions"):
            continue
        lp = ps.layers[i]
        index = torch.nonzero(torch.sigmoid(lp.logit.detach()) > 0.5).squeeze(1)
        if len(index) < 2:
            continue
        xy = (lp.base + lp.shifts)[index]
        pairs = cKDTree(xy.detach().cpu().numpy()).query_pairs(d0, output_type="ndarray")
        if len(pairs) == 0:
            continue
        pairs = torch.as_tensor(pairs, device=xy.device)
        distance = torch.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], dim=1)
        total = total + ((1.0 - distance / d0).clamp(min=0.0) ** 2).sum() / len(index)
    return total


def attraction(ps: ParameterSet, settings: TrainSettings) -> torch.Tensor:
    """Eq. (3.4a): the counterpart of the repulsion for pairs closer than the merge distance."""

    d_merge = float(settings.regularisation.merge_distance_A) / (ps.experiment.imaging.pixel_size_nm * 10.0)
    total = torch.zeros((), device=ps.device)
    for i, spec in enumerate(ps.experiment.layers):
        if not settings.layer_enabled(spec.name, "free_positions"):
            continue
        lp = ps.layers[i]
        index = torch.nonzero(torch.sigmoid(lp.logit.detach()) > 0.5).squeeze(1)
        if len(index) < 2:
            continue
        xy = (lp.base + lp.shifts)[index]
        pairs = cKDTree(xy.detach().cpu().numpy()).query_pairs(d_merge, output_type="ndarray")
        if len(pairs) == 0:
            continue
        pairs = torch.as_tensor(pairs, device=xy.device)
        distance = torch.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], dim=1)
        total = total + ((distance / d_merge) ** 2).sum() / len(index)
    return total


def elastic_energy(ps: ParameterSet, settings: TrainSettings) -> torch.Tensor:
    """Eq. (3.5)."""

    reg = settings.regularisation
    energies = []
    for group in ps.experiment.groups:
        if not settings.group_enabled(group.name, "strain_field"):
            continue
        indices = ps.experiment.layer_indices(group.name)
        points = [ps.rigid_positions(i) for i in indices if len(ps.layers[i]) > 0]
        if not points:
            continue
        points = torch.cat([p.detach() for p in points])
        J = strain.jacobian(ps.groups[group.name].strain, points, ps.shape)
        w = strain.elastic_energy_density(J, reg.strain_energy_lambda, reg.strain_energy_mu)
        energies.append(w.mean())
    return torch.stack(energies).mean() if energies else torch.zeros((), device=ps.device)


def prior_loss(ps: ParameterSet, settings: TrainSettings) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Eq. (3.6) and the value of every evaluated term."""

    reg = settings.regularisation
    total = torch.zeros((), device=ps.device)
    parts: Dict[str, float] = {}

    def add(name, weight, fn):
        nonlocal total
        if weight > 0.0:
            value = fn()
            total = total + weight * value
            parts[name] = float(value.detach())

    add("entropy", reg.occ_entropy_a, lambda: entropy(ps))
    add("position_shifts", reg.position_shifts_a, lambda: position_norm(ps, settings))
    add("repulsion", reg.repulsion_a, lambda: repulsion(ps, settings))
    add("attraction", reg.attraction_a, lambda: attraction(ps, settings))
    add("strain_energy", reg.strain_energy_a, lambda: elastic_energy(ps, settings))
    return total, parts
