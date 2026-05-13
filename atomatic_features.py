import os


import numpy as np

from structure_clustering import find_hexagons, compare_s_triangles


def hexagon_features(h, atoms, ww_neighbors):
    """
    Feature-Vektor pro Hexagon aus geometrischen Eigenschaften.
    """
    pos  = atoms.positions
    w_z  = np.mean(pos[list(h['W'])][:, 2])
    center_2d = np.mean(pos[list(h['W'])][:, :2], axis=0)
    ring = set(h['W']) | set(h['S'])

    n_S  = len(h['S'])
    s_arr = np.array(h['S']) if n_S > 0 else np.array([], dtype=int)

    # ── S oben / unten ───────────────────────────────────────
    if n_S > 0:
        s_z   = pos[s_arr][:, 2]
        n_top = int(np.sum(s_z > w_z + 0.5))
        n_bot = int(np.sum(s_z < w_z - 0.5))
        top_i = s_arr[s_z > w_z + 0.5]
        bot_i = s_arr[s_z < w_z - 0.5]
    else:
        n_top = n_bot = 0
        top_i = bot_i = np.array([], dtype=int)

    # ── Radiale Abstände S zum Zentrum ───────────────────────
    if n_S > 0:
        r_s = np.linalg.norm(pos[s_arr][:, :2] - center_2d, axis=1)
        r_mean = float(np.mean(r_s))
        r_std  = float(np.std(r_s))
    else:
        r_mean = r_std = 0.0

    # ── Verdrehwinkel top vs bot Dreieck ─────────────────────
    if len(top_i) >= 2 and len(bot_i) >= 2:
        tri_info = compare_s_triangles(pos[top_i], pos[bot_i], center_2d)
        angle    = tri_info['angle']
        r_ratio  = tri_info['r_ratio']
    else:
        angle   = 0.0
        r_ratio = 1.0

    # ── W-Koordination (Randindikator) ───────────────────────
    w_ext = [len(ww_neighbors[w] - ring) for w in h['W']]
    min_w_coord = float(min(w_ext))
    mean_w_coord = float(np.mean(w_ext))

    # ── W-Dreieck-Verzerrung ─────────────────────────────────
    w_pos = pos[list(h['W'])][:, :2]
    w_dists = [np.linalg.norm(w_pos[i] - w_pos[j])
               for i in range(3) for j in range(i+1, 3)]
    w_dist_std = float(np.std(w_dists))

    return np.array([
        n_S,
        r_mean,        # mittlerer S-Abstand zum Zentrum
        r_std,         # Streuung S-Abstände
        angle,         # Verdrehwinkel top/bot Dreieck
        r_ratio,       # Radiusverhältnis top/bot
        w_dist_std,    # W-Dreieck Verzerrung
    ])


# ── Features berechnen ────────────────────────────────────────────────────────
from ase.io import read

path = "platelet_11/"
atoms = read(os.path.join(path, 'mod_structure.xyz'))
hexagons, ww_neighbors = find_hexagons(atoms)

X = np.array([hexagon_features(h, atoms, ww_neighbors) for h in hexagons])

# NaN/Inf auffangen
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import hdbscan
X_scaled = StandardScaler().fit_transform(X)

# ── HDBSCAN ───────────────────────────────────────────────────────────────────
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2)
labels    = clusterer.fit_predict(X_scaled)

for i, h in enumerate(hexagons):
    h['cluster'] = int(labels[i])
from collections import Counter
cluster_counts = Counter(labels)
print(f"Cluster gefunden: {len(set(labels) - {-1})}")
print(cluster_counts)

# ── Plot: räumlich + Feature-Scatter ─────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mplcm

unique_labels = sorted(set(labels))
n_clusters    = len([l for l in unique_labels if l >= 0])
cmap          = mplcm.get_cmap('tab10')
color_map     = {l: ('#aaaaaa' if l == -1 else cmap(l % 10))
                 for l in unique_labels}

fig, axes = plt.subplots(1, 3, figsize=(20, 7),
                         gridspec_kw={'width_ratios': [2.5, 1.5, 1]})
ax_map, ax_feat, ax_bar = axes

# ── 1. Räumliche Karte mit Dreiecken ─────────────────────────────────────────
from matplotlib.collections import PolyCollection

syms = np.array(atoms.get_chemical_symbols())
pos  = atoms.positions
ax_map.scatter(pos[syms == 'S', 0], pos[syms == 'S', 1],
               c='#D3D1C7', s=6, linewidths=0, zorder=1)
ax_map.scatter(pos[syms == 'W', 0], pos[syms == 'W', 1],
               c='#888780', s=12, linewidths=0, zorder=2)

def triangle_verts(s_group, pos):
    pts    = pos[s_group][:, :2]
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]

by_cluster = {l: [] for l in unique_labels}

for h in hexagons:
    l     = h['cluster']
    s_arr = np.array(h['S'])
    w_arr = np.array(h['W'])
    if len(s_arr) == 0:
        continue
    w_z   = np.mean(pos[list(h['W'])][:, 2])
    s_z   = pos[s_arr][:, 2]
    top_i = s_arr[s_z > w_z + 0.5]
    bot_i = s_arr[s_z < w_z - 0.5]
    by_cluster[l].append(triangle_verts(w_arr, pos))

for l in unique_labels:
    color = color_map[l]
    alpha = 0.35 if l == -1 else 0.65
    if by_cluster[l]:
        ax_map.add_collection(PolyCollection(
            by_cluster[l], facecolor=color,
            edgecolor='white', linewidth=0.3, alpha=alpha, zorder=3))

ax_map.autoscale_view()
ax_map.set_aspect('equal')
ax_map.set_xlabel('x (Å)', fontsize=11)
ax_map.set_ylabel('y (Å)', fontsize=11)
ax_map.set_title(f'Automatische Cluster  —  {len(hexagons)} Hexagone\n'
                 f'▲ obere S  - - untere S', fontsize=11)

patches = [mpatches.Patch(color=color_map[l],
           label=f"Cluster {l}  (n={cluster_counts[l]})" if l >= 0
                 else f"Ausreißer  (n={cluster_counts[l]})")
           for l in unique_labels]
ax_map.legend(handles=patches, fontsize=8, loc='upper right', framealpha=0.9)

# ── 2. Feature-Scatter: Verdrehwinkel vs. n_S ────────────────────────────────
angles = X[:, 3]   # Verdrehwinkel
n_S    = X[:, 0]   # Anzahl S

for l in unique_labels:
    mask  = labels == l
    color = color_map[l]
    label = f"Cluster {l}" if l >= 0 else "Ausreißer"
    ax_feat.scatter(n_S[mask], angles[mask],
                    c=[color], s=18, alpha=0.7,
                    linewidths=0, label=label)

ax_feat.axhline( 60, color='#1D9E75', linewidth=0.8, linestyle=':', alpha=0.7)
ax_feat.axhline(  0, color='#D85A30', linewidth=0.8, linestyle=':', alpha=0.7)
ax_feat.text(6.1,  62, 'H-Phase', fontsize=8, color='#1D9E75', va='bottom')
ax_feat.text(6.1,   2, 'T-Phase', fontsize=8, color='#D85A30', va='bottom')
ax_feat.set_xlabel('Anzahl S-Atome', fontsize=10)
ax_feat.set_ylabel('Verdrehwinkel top/bot (°)', fontsize=10)
ax_feat.set_title('Feature-Raum', fontsize=11)
ax_feat.tick_params(labelsize=9)

# ── 3. Balkendiagramm ────────────────────────────────────────────────────────
bar_labels = [f"Cluster {l}" if l >= 0 else "Ausreißer"
              for l in unique_labels]
values = [cluster_counts[l] for l in unique_labels]
colors = [color_map[l] for l in unique_labels]
bars   = ax_bar.barh(bar_labels, values, color=colors, alpha=0.9, height=0.6)
ax_bar.bar_label(bars, padding=4, fontsize=9)
ax_bar.set_xlabel('Anzahl Hexagone', fontsize=10)
ax_bar.set_title('Clustergrößen', fontsize=11)
ax_bar.invert_yaxis()
ax_bar.set_xlim(0, max(values) * 1.25)
ax_bar.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('ws2_auto_clusters.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Feature-Bedeutung ausgeben ────────────────────────────────────────────────
feature_names = ['n_S',
                 'r_mean', 'r_std', 'angle', 'r_ratio',
                 'w_dist_std']


print("\nCluster-Mittelwerte pro Feature:")
print(f"{'Feature':<16}", end='')
for l in unique_labels:
    print(f"  C{l:>2}", end='')
print()
for fi, fname in enumerate(feature_names):
    print(f"{fname:<16}", end='')
    for l in unique_labels:
        mask = labels == l
        print(f"  {X[mask, fi].mean():>5.2f}", end='')
    print()