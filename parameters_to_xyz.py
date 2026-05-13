import json

import matplotlib.pyplot as plt
import numpy as np
import sys
from PySide6.QtWidgets import QApplication, QFileDialog
from scipy.spatial import ConvexHull

from mains.main import deserialize
from nn import ParticlesConfig, ParticlesTunableParameters
from parameter_frame import serialize
from potential import DifferentiablePotential
from training_parameter import TrainMethod

from ase import Atoms

from shapely.geometry import Polygon, MultiPoint, Point


def load_json(path=None):

    if path is None:
        app = QApplication(sys.argv)
        file_path, _ = QFileDialog.getOpenFileName(
                    None,
                    "Parameter laden",
                    "parameters",
                    "JSON-Dateien (*.json)"
                )

        if not file_path:
            return
        file_path = path

    else:
        file_path = path


    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


if __name__ == "__main__":
    path = "platelet_11/"
    config_file = path + "config/config.json"
    param_file = path + "best_long_free_ws2_no_occ4.json"
    params_json = load_json(param_file)


    with open(config_file, "r") as f:
        config_json = json.load(f)
        config = deserialize(config_json, {"ParticlesConfig": ParticlesConfig})


    params = ParticlesTunableParameters(**params_json)

    pixel_size_nm = config.pixel_size_nm

    train_method = TrainMethod(area_masks=[None,None], target_gr=None)
    pot_layer = DifferentiablePotential.from_config_and_parameters(config, train_method, params.potential_parameters)
    positions_occ, Zs = pot_layer.occupied_positions_A(occ_cutoff=0.5, include_graphene=False)

    atoms = Atoms(numbers=Zs, positions=positions_occ)
    atoms.write(path+'structure.xyz')

    from ase.visualize import view

    view(atoms)
