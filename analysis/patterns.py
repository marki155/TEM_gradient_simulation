import json

import matplotlib.pyplot as plt
import numpy as np
import sys
from PySide6.QtWidgets import QApplication, QFileDialog
from scipy.spatial import ConvexHull

from nn import ParticlesConfig, ParticlesTunableParameters
from parameter_frame import serialize
from potential import DifferentiablePotential
from training_parameter import TrainMethod

from shapely.geometry import Polygon, MultiPoint, Point


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
    yes = np.any(d < tol)
    if yes:
        idx = int(np.where(d < tol)[0])
    else:
        idx = None
    return yes, idx

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


def add_missing_sulfur(W, S_pos_all, S_occ, edge_w_indices,
                       ref_vecs_bottom, tol=2.0):
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


    W_polygon = Polygon(W[edge_w_indices])


    added_S = []
    add_S_idx = []

    for idx in edge_w_indices:
        w = W[idx]

        # erwartete untere S-Positionen
        candidates_idxs = []
        candidates_dist_to_centers = []

        for v in ref_vecs_bottom:
            candidate = w + v
            yes, cand_idx = has_atom_near(S_pos_all, candidate, tol)
            if yes:
                candidates_idxs.append(cand_idx)


        for i, idx in enumerate(candidates_idxs):
            #if not W_polygon.contains(Point(S_pos_all[idx])):
            add_S_idx.append(idx)
            added_S.append(S_pos_all[idx])

    from matplotlib.path import Path
    polygon_punkte = W[edge_w_indices]

    def sortiere_nn(punkte, toleranz=0.05):
        """
        Nearest-Neighbor Sortierung.
        Bei zwei gleich weit entfernten Nachbarn (innerhalb 'toleranz' * beste Distanz)
        wird der Punkt gewählt, der weiter vom Zentroid entfernt ist.
        """
        punkte = punkte.copy()
        n = len(punkte)
        zentroid = punkte.mean(axis=0)
        besucht = np.zeros(n, dtype=bool)
        reihenfolge = [0]
        besucht[0] = True

        for _ in range(n - 1):
            aktuell = reihenfolge[-1]
            distanzen = np.linalg.norm(punkte - punkte[aktuell], axis=1)
            distanzen[besucht] = np.inf

            beste_dist = distanzen.min()
            # Kandidaten: alle Punkte innerhalb der Toleranz zur besten Distanz
            kandidaten = np.where(distanzen <= beste_dist * (1 + toleranz))[0]

            if len(kandidaten) == 1:
                naechster = kandidaten[0]
            else:
                # Tiebreaker: Kandidat der am weitesten vom Zentroid entfernt ist
                dist_zentroid = np.linalg.norm(punkte[kandidaten] - zentroid, axis=1)
                naechster = kandidaten[np.argmax(dist_zentroid)]

            reihenfolge.append(naechster)
            besucht[naechster] = True

        return punkte[reihenfolge]

    polygon_sortiert = sortiere_nn(polygon_punkte)


    alle_punkte = np.array(added_S)
    poly_path = Path(np.vstack([polygon_sortiert, polygon_sortiert[0]]))
    maske = poly_path.contains_points(alle_punkte)

    innen = alle_punkte[maske]
    aussen = alle_punkte[~maske]

    # ── Plot ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    from matplotlib.patches import Polygon as MplPolygon
    # Polygon
    polygon_patch = MplPolygon(
        polygon_sortiert,
        closed=True,
        facecolor="#3b82f620",
        edgecolor="#3b82f6",
        linewidth=1.5,
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(polygon_patch)

    # Polygon-Eckpunkte
    ax.scatter(*polygon_punkte.T, s=60, facecolors="white",
               edgecolors="#3b82f6", linewidths=2, zorder=4, label="Eckpunkte")

    # Punkte außen
    ax.scatter(*aussen.T, s=50, color="#f97316",
               edgecolors="#c2410c", linewidths=0.8, zorder=3, label="außen")

    # Punkte innen
    ax.scatter(*innen.T, s=50, color="#22c55e",
               edgecolors="#15803d", linewidths=0.8, zorder=3, label="innen")

    # Achsen & Style
    #ax.set_xlim(-0.5, 6.5)
    #ax.set_ylim(-0.5, 6.5)
    ax.set_aspect("equal")
    ax.grid(True, color="gray", alpha=0.15, linewidth=0.8)
    ax.set_facecolor("#f8f8f6")
    fig.patch.set_facecolor("#f8f8f6")
    ax.set_title("Polygon · Punkt-Filter", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("x");
    ax.set_ylabel("y")
    ax.legend(framealpha=0.9, fontsize=9)

    # Zähler
    ax.text(0.02, 0.98,
            f"innen: {maske.sum()}   außen: {(~maske).sum()}",
            transform=ax.transAxes, fontsize=9,
            va="top", color="#555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    plt.tight_layout()
    plt.show()

    add_S_idx = np.array(add_S_idx)

    for i in add_S_idx[~maske]:
        S_occ[i] = 50

    return S_occ, np.array(added_S)[~maske]

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
    param_file = path + "best_long_free_ws2_no_occ4.json"
    params_json = load_json(param_file)
    config_json = load_json(config_file)

    param_save_file = path + "best_with_added_s.json"

    config = ParticlesConfig(**config_json)
    params = ParticlesTunableParameters(**params_json)

    pixel_size_nm = config.pixel_size_nm

    train_method = TrainMethod(area_masks=[None,None], target_gr=None)
    pot_layer = DifferentiablePotential.from_config_and_parameters(config, train_method, params.potential_parameters)
    positions_occ = pot_layer.occupied_positions(occ_cutoff=0.5)
    positions_all = pot_layer.get_positions()
    occupancies = params.potential_parameters["occupancies"]

    S_bottom_all = positions_all[1].cpu().numpy()
    S_bottom_occ = occupancies[1]
    W = positions_occ[2]
    S_top_all = positions_all[3].cpu().numpy()
    S_top_occ = occupancies[3]


    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    o = sigmoid(np.array(S_bottom_occ))
    undecided = np.where((o > 0.49) & (o < 0.51))
    #plt.scatter(S_bottom_all[undecided,1], S_bottom_all[undecided,0], color='blue', label='S unten')
    #plt.show()


    #plt.hist(sigmoid(np.array(S_bottom_occ)), bins=100)
    #plt.show()

    # 1) Referenz-Umgebung bestimmen

    edge_w = find_edge_w_by_w_neighbors(W, cutoff=1.5*3.18*.1/pixel_size_nm)
    edges_w_filtered = []

    for i in edge_w:
        w = W[i]
        if w[1] < 1750:
            if not (w[0] > 1500 and w[1] > 750):
                edges_w_filtered.append(i)

    ref_b, ref_t, ref_w_index = get_reference_s_vectors(S_bottom_all, W, S_top_all, ws_cutoff=1.5*3.18*0.5*.1/pixel_size_nm)

    # 2) Fehlende S ergänzen
    #W, S_pos_all, S_occ, edge_w_indices, ref_vecs_bottom, ref_vecs_top, match_tol=0.3
    S_bottom_occ, added_b = add_missing_sulfur(
        W=W, S_pos_all=S_bottom_all, S_occ=S_bottom_occ,
        edge_w_indices=edges_w_filtered,
        ref_vecs_bottom=ref_b,
        tol=15.0
    )


    print("Referenz-W:", ref_w_index)
    print("Neu eingefügte untere S:", len(added_b))
    #print("Neu eingefügte obere S:", len(added_t))

    w_atoms_edge = W[edge_w]



    occupancies_new = [occupancies[0], S_bottom_occ, occupancies[2], S_top_occ]

    params.potential_parameters["occupancies"] = occupancies_new
    with open(param_save_file, "w", encoding="utf-8") as f:
        json.dump(serialize(params), f, indent=4)