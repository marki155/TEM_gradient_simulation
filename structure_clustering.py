import os.path

from ase.neighborlist import neighbor_list
from collections import Counter
from ase.io import read

path = "platelet_11/"
atoms = read(os.path.join(path, 'mod_structure.xyz'))


from ase.neighborlist import neighbor_list
from collections import Counter
import numpy as np
from collections import Counter
import numpy as np


def find_hexagons(atoms, r_WW=3.9):
    """
    Jedes WS2-Hexagon = 3 W-Atome die ein gleichseitiges Dreieck bilden.
    Strategie: finde alle W-Dreiecke, dann weise jedem Dreieck die 3 S-Atome zu
    die am nächsten am Dreiecksmittelpunkt liegen.
    """
    syms = np.array(atoms.get_chemical_symbols())
    pos = atoms.positions

    w_idx = np.where(syms == 'W')[0]
    s_idx = np.where(syms == 'S')[0]

    # W-W Nachbarliste
    i_list, j_list = neighbor_list('ij', atoms, {('W', 'W'): r_WW})

    ww_neighbors = {int(w): set() for w in w_idx}
    for i, j in zip(i_list, j_list):
        ww_neighbors[int(i)].add(int(j))

    # Alle W-Dreiecke finden
    triangles = []
    for w1 in w_idx:
        for w2 in ww_neighbors[w1]:
            if w2 <= w1:
                continue
            for w3 in ww_neighbors[w1] & ww_neighbors[w2]:
                if w3 <= w2:
                    continue
                triangles.append((int(w1), int(w2), int(w3)))

    print(f"W-Dreiecke gefunden: {len(triangles)}")

    # Für jedes Dreieck: die 3 nächsten S-Atome zum Schwerpunkt
    hexagons = []
    for (w1, w2, w3) in triangles:
        center = (pos[w1][:2] + pos[w2][:2] + pos[w3][:2]) / 3.0

        dists = np.linalg.norm(pos[s_idx][:,:2] - center, axis=1)
        dists_sorted = np.argsort(dists)
        index = np.argmax(dists[dists_sorted] > 3.0)
        closest = s_idx[dists_sorted[:index]]

        # Plausibilitätscheck: alle 3 S sollten nah am Zentrum sein
        if dists[np.argsort(dists)[2]] > r_WW:
            continue

        hexagons.append({
            'W': (w1, w2, w3),
            'S': tuple(closest),
            'center': center[:2],
        })

    return hexagons, ww_neighbors

def compare_s_triangles(s_top, s_bot, center_2d):


    # Schwerpunktswinkel jedes S-Atoms relativ zum W-Dreieck-Zentrum
    def triangle_angle(s_group):
        vecs   = s_group[:, :2] - center_2d
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        # mittlerer Winkel des Dreiecks (Schwerpunkt-Winkel)
        return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

    def triangle_radius(s_group):
        vecs = s_group[:, :2] - center_2d
        return np.mean(np.linalg.norm(vecs, axis=1))

    angle_top = np.degrees(triangle_angle(s_top))
    if angle_top < 0:
        angle_top += 120
    if angle_top > 120:
        angle_top -= 120
    angle_bot = np.degrees(triangle_angle(s_bot))
    if angle_bot < 0:
        angle_bot += 120
    if angle_bot > 120:
        angle_bot -= 120
    r_top     = triangle_radius(s_top)
    r_bot     = triangle_radius(s_bot)

    # Rotationswinkel zwischen den beiden Dreiecken (in Grad)
    delta = np.abs(angle_top - angle_bot)


    # Radiusverhältnis: 1.0 = symmetrisch, abweichend = verzerrt
    r_ratio = r_top / r_bot if r_bot > 0 else np.nan

    # ── Phasenzuordnung ──────────────────────────────────────
    # H-Phase:  |delta| ~ 60° (Dreiecke gegeneinander verdreht)
    # T-Phase:  |delta| ~ 0°  (Dreiecke deckungsgleich)
    # T'-Phase: dazwischen + asymmetrisches r_ratio

    is_symmetric = 0.85 < r_ratio < 1.15 if not np.isnan(r_ratio) else False

    if abs(abs(delta) - 60) < 10 and is_symmetric:
        phase = 'H'
    elif abs(delta) < 10 and is_symmetric:
        phase = 'T'
    elif abs(abs(delta) - 60) < 10 and not is_symmetric:
        phase = 'H_distorted'
    else:
        phase = "T'"   # verdreht + asymmetrisch

    return {
        'phase':   phase,
        'angle':   round(delta, 2),     # Verdrehwinkel in Grad
        'r_top':   round(r_top, 3),     # mittlerer Radius oberes Dreieck
        'r_bot':   round(r_bot, 3),     # mittlerer Radius unteres Dreieck
        'r_ratio': round(r_ratio, 3),   # Symmetriemaß
    }


def classify_hexagon(h, atoms, ww_neighbors, r_WW=3.9):
    syms = np.array(atoms.get_chemical_symbols())
    pos = atoms.positions
    ring = set(h['W']) | set(h['S'])

    n_S = len(h['S'])
    center_3d = np.mean(pos[list(h['W'])], axis=0)  # Schwerpunkt der W-Atome

    # ── Lage: Rand oder innen ────────────────────────────────
    w_ext_coords = [len(ww_neighbors[w] - ring) for w in h['W']]
    is_edge = any(c < 2 for c in w_ext_coords)

    # ── S-Lagen: oben / unten relativ zur W-Ebene ────────────
    w_z = np.mean(pos[list(h['W'])], axis=0)[2]
    if n_S > 0:
        s_i = np.array(h['S'])
        s_z = pos[s_i][:, 2]
        S_top_i = s_i[np.where(s_z > w_z + 0.5)]
        S_bottom_i = s_i[np.where(s_z < w_z - 0.5)]
        n_S_top = S_top_i.shape[0]
        n_S_bot = S_bottom_i.shape[0]
    else:
        n_S_top = n_S_bot = 0

    # ── S-Verzerrung: mittlere radiale Abweichung vom Zentrum ─
    if n_S > 0:
        triangles_info = compare_s_triangles(pos[S_top_i], pos[S_bottom_i], center_3d[:2])
    else:
        mean_deviation = np.nan
        is_distorted = False

    # ── Klassifikation ───────────────────────────────────────
    prefix = 'edge_' if is_edge else ''

    if n_S == 6:
        cls = 'pristine'

    elif n_S < 6:
        cls = f'V_{6-n_S}S'

        #if n_S_top == 2 or n_S_bot == 2:
        #    cls = 'V_S_same_side'  # beide S auf gleicher Seite
        #else:
        #    cls = 'V_S'  # je eines oben/unten
    else:
        cls = 'unknown'

    cls += "_" + triangles_info['phase']

    return prefix + cls, {
        'n_S': n_S,
        'n_S_top': n_S_top,
        'n_S_bot': n_S_bot,
        'is_edge': is_edge,
        'S_triangles_info': triangles_info
    }


# ── Ausführen ────────────────────────────────────────────────────────────────
hexagons, ww_neighbors = find_hexagons(atoms)
deltas = []
for h in hexagons:
    h['class'], h['info'] = classify_hexagon(h, atoms, ww_neighbors)
    deltas.append(h['info']['S_triangles_info']['angle'])

import matplotlib.pyplot as plt
plt.hist(deltas, bins=30, color='skyblue', edgecolor='black')
plt.show()

counts = Counter(h['class'] for h in hexagons)
print(counts)


# ── Plot ─────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection, LineCollection
import numpy as np
from collections import Counter

class_colors = {
    'pristine_H':          '#1D9E75',
    'pristine_T':          '#085041',
    "pristine_T'":         '#5DCAA5',
    'pristine_H_distorted':'#9FE1CB',
    'V_1S_H':              '#7F77DD',
    'V_1S_T':              '#3C3489',
    "V_1S_T'":             '#AFA9EC',
    'V_2S_H':              '#D85A30',
    "V_2S_T'":             '#F0997B',
    'V_3S_H':              '#D4537E',
    "V_3S_T'":             '#ED93B1',
    'V_4S_H':              '#BA7517',
    'V_5S_H':              '#EF9F27',
    'V_6S_H':              '#E24B4A',
    'edge_pristine_H':     '#85B7EB',
    'edge_pristine_T':     '#0C447C',
    "edge_pristine_T'":    '#B5D4F4',
    'edge_V_1S_H':         '#888780',
    'edge_V_2S_H':         '#B4B2A9',
    'unknown':             '#444441',
}

def triangle_verts(s_group, pos):
    """xy-Koordinaten eines S-Dreiecks, nach Winkel sortiert."""
    pts = pos[s_group][:, :2]
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                         gridspec_kw={'width_ratios': [2.5, 1]})
ax_map = axes[0]
ax_bar = axes[1]

# ── Hintergrund: Atome ────────────────────────────────────────────────────────
syms = np.array(atoms.get_chemical_symbols())
pos  = atoms.positions
ax_map.scatter(pos[syms == 'S', 0], pos[syms == 'S', 1],
               c='y', s=6, linewidths=0, zorder=1)
ax_map.scatter(pos[syms == 'W', 0], pos[syms == 'W', 1],
               c='#888780', s=12, linewidths=0, zorder=2)

# ── Dreiecke sammeln ──────────────────────────────────────────────────────────
top_verts_by_class  = {cls: [] for cls in class_colors}
bot_verts_by_class  = {cls: [] for cls in class_colors}

for h in hexagons:
    cls = h['class']
    if cls not in top_verts_by_class:
        top_verts_by_class[cls] = []
        bot_verts_by_class[cls] = []

    w_z   = np.mean(pos[list(h['W'])][:, 2])
    s_arr = np.array(h['S'])
    w_arr = np.array(h['W'])
    if len(s_arr) == 0:
        continue
    s_z   = pos[s_arr][:, 2]

    top_i = s_arr[s_z > w_z + 0.5]
    bot_i = s_arr[s_z < w_z - 0.5]


    top_verts_by_class[cls].append(triangle_verts(w_arr, pos))

# ── Dreiecke zeichnen ─────────────────────────────────────────────────────────
for cls in list(top_verts_by_class.keys()) + list(bot_verts_by_class.keys()):
    color = class_colors.get(cls, '#888780')
    alpha = 0.45 if cls.startswith('edge') else 0.65

    # oberes Dreieck: gefüllt
    if top_verts_by_class.get(cls):
        col_top = PolyCollection(top_verts_by_class[cls],
                                 facecolor=color,
                                 edgecolor='white',
                                 linewidth=0.4,
                                 alpha=alpha,
                                 zorder=3)
        ax_map.add_collection(col_top)

    # unteres Dreieck: nur Umriss, gestrichelt
    if bot_verts_by_class.get(cls):
        for verts in bot_verts_by_class[cls]:
            closed = np.vstack([verts, verts[0]])  # Dreieck schließen
            ax_map.plot(closed[:, 0], closed[:, 1],
                        color=color, linewidth=0.8,
                        linestyle='--', alpha=alpha + 0.1, zorder=4)

ax_map.autoscale_view()
ax_map.set_aspect('equal')
ax_map.set_xlabel('x (Å)', fontsize=11)
ax_map.set_ylabel('y (Å)', fontsize=11)
ax_map.set_title(f'WS₂ S-Dreiecke  —  {len(hexagons)} Hexagone  '
                 f'(▲ oben  - - unten)', fontsize=12)

# ── Legende ───────────────────────────────────────────────────────────────────
cls_order = [c for c in class_colors if c in counts]
for c in counts:
    if c not in cls_order:
        cls_order.append(c)

patches = [
    mpatches.Patch(facecolor=class_colors.get(c, '#888780'),
                   edgecolor='white', linewidth=0.5,
                   label=f"{c}  ({counts[c]})")
    for c in cls_order
]
ax_map.legend(handles=patches, fontsize=7.5, loc='upper right',
              framealpha=0.92, ncol=1)

# ── Balkendiagramm ────────────────────────────────────────────────────────────
values = [counts[c] for c in cls_order]
colors = [class_colors.get(c, '#888780') for c in cls_order]
bars   = ax_bar.barh(cls_order, values, color=colors, alpha=0.9, height=0.6)
ax_bar.bar_label(bars, padding=4, fontsize=9)
ax_bar.set_xlabel('Anzahl Hexagone', fontsize=10)
ax_bar.set_title('Klassenverteilung', fontsize=12)
ax_bar.invert_yaxis()
ax_bar.set_xlim(0, max(values) * 1.25)
ax_bar.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('ws2_triangles.png', dpi=150, bbox_inches='tight')
plt.show()