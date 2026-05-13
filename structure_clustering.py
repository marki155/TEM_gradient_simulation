from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from umap import umap_ as umap
import numpy as np
from ase.io import read
import os

path = "platelet_11/"
atoms = read(os.path.join(path, "mod_structure.xyz"))




# 1. SOAP-Deskriptor
soap = SOAP(
    species=["W", "S"],
    r_cut=4.5,       # ~2. Nachbarschale
    n_max=9,
    l_max=9,
    periodic=False,
    sparse=False
)

features = soap.create(atoms)  # shape: (N_atoms, N_features)

# Nur W-Atome klassifizieren
w_mask = np.array([a.symbol == 'W' for a in atoms])
X = features[w_mask]

# 2. Normalisieren
X = StandardScaler().fit_transform(X)

# 3. Dimensionsreduktion
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
X_2d = reducer.fit_transform(X)

# 4. Clustering
# HDBSCAN: findet Cluster automatisch, markiert Ausreißer als -1
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
labels = clusterer.fit_predict(X_2d)


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Farben ──────────────────────────────────────────────
unique_labels = np.unique(labels)
cmap = plt.cm.get_cmap("tab10", len(unique_labels))
color_map = {l: ("#aaaaaa" if l == -1 else cmap(i))
             for i, l in enumerate(unique_labels)}
colors = np.array([color_map[l] for l in labels])

# ── 1. UMAP-Plot ─────────────────────────────────────────
ax = axes[0]
for l in unique_labels:
    mask = labels == l
    label_str = "Ausreißer" if l == -1 else f"Cluster {l}  (n={mask.sum()})"
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=[color_map[l]], s=18, alpha=0.7,
               linewidths=0, label=label_str)
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.set_title("UMAP + HDBSCAN — W-Atome")
ax.legend(fontsize=9, markerscale=1.5)

# ── 2. Räumliche Karte ────────────────────────────────────
ax2 = axes[1]
w_positions = atoms.positions[w_mask]
for l in unique_labels:
    mask = labels == l
    label_str = "Ausreißer" if l == -1 else f"Cluster {l}"
    ax2.scatter(w_positions[mask, 0], w_positions[mask, 1],
                c=[color_map[l]], s=22, alpha=0.85,
                linewidths=0, label=label_str)
ax2.set_xlabel("x (Å)")
ax2.set_ylabel("y (Å)")
ax2.set_title("Räumliche Verteilung der Cluster")
ax2.set_aspect("equal")
ax2.legend(fontsize=9, markerscale=1.5)

plt.tight_layout()
plt.savefig("ws2_clusters.png", dpi=150, bbox_inches="tight")
plt.show()
print(np.unique(labels, return_counts=True))