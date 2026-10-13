"""Small synthetic experiments for the learning tests."""

import numpy as np

from temsim.data.schema import Experiment, Group, Imaging, Layer, Structure, Stem

SHAPE, PAD, PX = (64, 72), 12, 0.02


def make_experiment(mode="tem", n=(30, 20), mixed=False, seed=0, groups=None, **imaging_kw):
    rng = np.random.default_rng(seed)
    layers = [Layer("top", "C", z_nm=0.0, group="g1", init_occupancy=3.0),
              Layer("bottom", "S", z_nm=0.33, group="g1" if groups is None else "g2", init_occupancy=3.0,
                    debye_waller_u=0.06)]
    zs = [6.0, 16.0]
    if mixed:
        layers[0].elements = ["B", "C", "N", "O"]
        zs[0] = 6.4
    gs = groups if groups is not None else [Group("g1", control_points=[4, 4], max_shift=5.0, max_rotation=0.2)]
    positions = [rng.uniform(8, min(SHAPE) - 8, (k, 2)) for k in n]
    numbers = [np.full(k, z) for k, z in zip(n, zs)]
    imaging = Imaging(shape=SHAPE, pixel_size_nm=PX, mode=mode, voltage_kV=80.0, padding=PAD,
                      aberrations={"C10": -8.0, "C12": 1.0, "phi12": 0.2, "C30": 1.0},
                      alpha_max_mrad=50.0, stem=Stem(25.0, (60.0, 200.0), 0.5), **imaging_kw)
    return Experiment(imaging, layers, gs, Structure(positions, numbers))
