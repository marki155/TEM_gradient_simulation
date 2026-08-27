"""Gitter-Basen fuer das ImageJ-Plugin (reines Jython, kein numpy).

Die Basis wird explizit in die init.json geschrieben, damit das Overlay im
Plugin und die Simulation in Python garantiert dieselbe Zelle benutzen.
"""

import math

# Gitterkonstanten in Angstroem und Schichtabstand M-X in Angstroem
MX2_TABLE = {
    "WS2":   (3.155, 1.575),
    "MoS2":  (3.160, 1.564),
    "WSe2":  (3.282, 1.676),
    "MoSe2": (3.288, 1.670),
}

GRAPHENE_A = 2.46
HBN_A = 2.504


def graphene_basis():
    """Wabengitter: hexagonale Zelle (gamma = 120 Grad), zwei Kohlenstoffe."""

    return [("C", 0.0, 0.0, 0.0), ("C", 1.0 / 3.0, 2.0 / 3.0, 0.0)]


def hbn_basis():
    return [("B", 0.0, 0.0, 0.0), ("N", 1.0 / 3.0, 2.0 / 3.0, 0.0)]


def mx2_basis(formula):
    """2H-MX2: X unten, M in der Mitte, X oben. z in Nanometern."""

    metal = formula[0] if len(formula) < 3 else formula[:-2]
    chalcogen = formula[-2]
    a, dz = MX2_TABLE.get(formula, (3.155, 1.575))
    return a, [(chalcogen, 1.0 / 3.0, 2.0 / 3.0, 0.0),
               (metal,     2.0 / 3.0, 1.0 / 3.0, dz / 10.0),
               (chalcogen, 1.0 / 3.0, 2.0 / 3.0, 2.0 * dz / 10.0)]


def basis_for(kind, formula="WS2", custom=None):
    """(Gitterkonstante a in A, Basis) fuer einen Gittertyp."""

    if kind == "Graphen":
        return GRAPHENE_A, graphene_basis()
    if kind == "hBN":
        return HBN_A, hbn_basis()
    if kind == "2H-MX2 (WS2, MoS2 ...)":
        return mx2_basis(formula)
    if kind == "eigene Zelle":
        return (custom or {}).get("a", 3.0), (custom or {}).get("basis", [])
    return GRAPHENE_A, graphene_basis()


def lattice_vectors(a, b, gamma_deg):
    """Gittervektoren in Angstroem: ((ax, ay), (bx, by))."""

    gamma = math.radians(gamma_deg)
    return (a, 0.0), (b * math.cos(gamma), b * math.sin(gamma))


def layer_points(a, b, gamma_deg, basis, pixel_size_nm, shape, rotation_deg,
                 offset_px, margin_px=0.0, limit=None):
    """Atompositionen je Schicht in Bildkoordinaten (Zeile, Spalte).

    ``limit`` deckelt die Punktzahl je Schicht - fuers Overlay reichen ein paar
    tausend, sonst wird das Zeichnen zaeh.
    """

    pixel_A = pixel_size_nm * 10.0
    (ax, ay), (bx, by) = lattice_vectors(a, b, gamma_deg)
    ax, ay, bx, by = ax / pixel_A, ay / pixel_A, bx / pixel_A, by / pixel_A

    height, width = shape
    cx, cy = height / 2.0, width / 2.0
    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    step = min(math.hypot(ax, ay), math.hypot(bx, by))
    reach = int(math.ceil(1.6 * max(height + 2 * margin_px, width + 2 * margin_px) / max(step, 1e-6)))

    lo_x, hi_x = -margin_px, height + margin_px
    lo_y, hi_y = -margin_px, width + margin_px

    layers = {}
    for symbol, fa, fb, z in basis:
        key = round(z, 6)
        points = layers.setdefault(key, {"z_nm": z, "elements": [], "points": []})
        base_x = fa * ax + fb * bx
        base_y = fa * ay + fb * by
        for i in range(-reach, reach + 1):
            for j in range(-reach, reach + 1):
                x = i * ax + j * bx + base_x
                y = i * ay + j * by + base_y
                rx = cos_t * x - sin_t * y + cx + offset_px[0]
                ry = sin_t * x + cos_t * y + cy + offset_px[1]
                if lo_x <= rx < hi_x and lo_y <= ry < hi_y:
                    points["points"].append((rx, ry))
                    points["elements"].append(symbol)

    ordered = [layers[k] for k in sorted(layers)]
    if limit:
        for layer in ordered:
            if len(layer["points"]) > limit:
                stride = int(math.ceil(len(layer["points"]) / float(limit)))
                layer["points"] = layer["points"][::stride]
                layer["elements"] = layer["elements"][::stride]
    return ordered
