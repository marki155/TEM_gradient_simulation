import json

import matplotlib.pyplot as plt
import numpy as np
import sys
from PySide6.QtWidgets import QApplication, QFileDialog

from nn import ParticlesConfig, ParticlesTunableParameters
from potential import DifferentiablePotential
from training_parameter import TrainMethod


def find_edge_w_by_w_neighbors(W, cutoff):

    N = len(W)
    neighbor_counts = np.zeros(N, dtype=int)

    for i in range(N):
        d = np.linalg.norm(W - W[i], axis=1)
        # selbst ausschließen und nächste Nachbarn zählen
        neighbor_counts[i] = np.sum((d < cutoff) & (d > 1e-12))

    edge_indices = np.where(neighbor_counts < 6)[0]
    return edge_indices


def pairwise_distances(points, p):
    return np.linalg.norm(points - p, axis=1)

def find_neighbors(points, center, cutoff):
    d = pairwise_distances(points, center)
    return np.where(d < cutoff)[0]

def has_atom_near(points, pos, tol):
    if len(points) == 0:
        return False
    d = np.linalg.norm(points - pos, axis=1)
    return np.any(d < tol)

def get_reference_s_vectors(S_bottom, W, S_top, ws_cutoff):
    """
    Findet ein vollständiges W-Atom mit 3 S unten und 3 S oben
    und gibt die relativen Vektoren W->S zurück.
    """
    for i, w in enumerate(W):
        idx_b = find_neighbors(S_bottom, w, ws_cutoff)
        idx_t = find_neighbors(S_top, w, ws_cutoff)

        if len(idx_b) == 3 and len(idx_t) == 3:
            vecs_b = S_bottom[idx_b] - w
            vecs_t = S_top[idx_t] - w
            return vecs_b, vecs_t, i

    raise ValueError("Kein vollständiges W mit 3 unteren und 3 oberen S gefunden.")


def add_missing_sulfur(S_bottom, W, S_top, edge_w_indices,
                       ref_vecs_bottom, ref_vecs_top,
                       match_tol=0.3):
    """
    Ergänzt fehlende S-Atome an den gegebenen Rand-W-Atomen.

    Parameter
    ----------
    S_bottom, W, S_top : ndarray
        Atompositionen, Form (N,2) oder (N,3)
    edge_w_indices : array-like
        Indizes der Rand-W-Atome
    ref_vecs_bottom : (3,d) ndarray
        Referenzvektoren W -> S_bottom
    ref_vecs_top : (3,d) ndarray
        Referenzvektoren W -> S_top
    match_tol : float
        Toleranz: wenn schon ein S näher als match_tol existiert,
        wird kein neues eingefügt.

    Returns
    -------
    S_bottom_new, S_top_new : ndarray
        Erweiterte S-Listen
    added_bottom, added_top : ndarray
        Neu eingefügte S-Positionen
    """
    S_bottom_new = S_bottom.copy()
    S_top_new = S_top.copy()

    added_bottom = []
    added_top = []

    for idx in edge_w_indices:
        w = W[idx]

        # erwartete untere S-Positionen
        for v in ref_vecs_bottom:
            candidate = w + v
            if not has_atom_near(S_bottom_new, candidate, match_tol):
                S_bottom_new = np.vstack([S_bottom_new, candidate])
                added_bottom.append(candidate)

        # erwartete obere S-Positionen
        for v in ref_vecs_top:
            candidate = w + v
            if not has_atom_near(S_top_new, candidate, match_tol):
                S_top_new = np.vstack([S_top_new, candidate])
                added_top.append(candidate)

    if len(added_bottom) == 0:
        added_bottom = np.empty((0, S_bottom.shape[1]))
    else:
        added_bottom = np.array(added_bottom)

    if len(added_top) == 0:
        added_top = np.empty((0, S_top.shape[1]))
    else:
        added_top = np.array(added_top)

    return S_bottom_new, S_top_new, added_bottom, added_top

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
    path = "../platelet_11/"
    config_file = path + "particles_config.json"
    param_file = path + "best.json"
    params_json = load_json(param_file)
    config_json = load_json(config_file)

    config = ParticlesConfig(**config_json)
    params = ParticlesTunableParameters(**params_json)

    pixel_size_nm = config.pixel_size_nm

    train_method = TrainMethod(area_masks=[None,None], target_gr=None)
    pot_layer = DifferentiablePotential.from_config_and_parameters(config, train_method, params.potential_parameters)
    positions = pot_layer.occupied_positions(occ_cutoff=0.5)

    S_bottom = positions[1]
    W = positions[2]
    S_top = positions[3]

    # 1) Referenz-Umgebung bestimmen

    edge_w = find_edge_w_by_w_neighbors(W, cutoff=1.5*3.18*.1/pixel_size_nm)
    ref_b, ref_t, ref_w_index = get_reference_s_vectors(S_bottom, W, S_top, ws_cutoff=1.5*3.18*0.5*.1/pixel_size_nm)

    # 2) Fehlende S ergänzen
    S_bottom_new, S_top_new, added_b, added_t = add_missing_sulfur(
        S_bottom, W, S_top,
        edge_w_indices=edge_w,
        ref_vecs_bottom=ref_b,
        ref_vecs_top=ref_t,
        match_tol=0.35
    )

    print("Referenz-W:", ref_w_index)
    print("Neu eingefügte untere S:", len(added_b))
    print("Neu eingefügte obere S:", len(added_t))

    w_atoms_edge = W[edge_w]
    plt.scatter(w_atoms_edge[:,1], w_atoms_edge[:,0], color='red', label='Rand-W-Atome')
    plt.scatter(added_b[:,1], added_b[:,0], color='blue', label='S unten')
    plt.show()