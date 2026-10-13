#@ String (label="Ergebnisdatei", value="", persist=false, required=false) result_path

"""Deckt sich das gezeichnete Overlay mit den Positionen, die simuliert werden?

Das Overlay baut seinen Pfad in Gitterkoordinaten und setzt Drehung und
Versatz in die Transformation. Dieser Test prueft, dass dabei genau dieselben
Atompositionen herauskommen wie bei layer_points() - also bei dem, was
spaeter wirklich simuliert wird.

    ImageJ-linux64 --headless --console --run imagej/test_overlay.py
"""
import os, sys, math

REPO = os.environ.get(
    "TEM_SIM_REPO", "/home/koegelmarco/remote_python/TEM_gradient_simulation")
sys.path.append(os.path.join(REPO, "imagej"))
import java.lang.System as System
from ij import IJ, ImagePlus
from ij.process import ByteProcessor
from java.awt.geom import Point2D

import TEM_Sim_Init as P
from tem_sim_lattice import layer_points, layer_points_local

H, W = 700, 900
imp = ImagePlus("t", ByteProcessor(W, H))

def check(kind, formula, rot, off, a_over, pixel):
    entry = {"name": "G", "kind": kind, "formula": formula, "a": a_over,
             "gamma_deg": 120.0, "rotation_deg": rot, "offset_px": off}
    params = {"pixel_size_nm": pixel, "lattices": [entry]}
    a, basis = P.resolve_lattice(entry)

    reference = layer_points(a, a, 120.0, basis, pixel, (H, W), rot, off)
    want = set()
    for layer in reference:
        for row, col in layer["points"]:
            want.add((round(row, 2), round(col, 2)))

    shapes, count, stride = P.lattice_shapes(entry, params, (H, W))
    roi = P.LatticeRoi(imp, shapes, 0.0)
    roi.set_placement(H / 2.0, W / 2.0, rot, off[0], off[1])

    radius = 0.5 * math.hypot(H, W) + math.hypot(off[0], off[1]) + 2.0 * a / (pixel * 10.0)
    local = layer_points_local(a, a, 120.0, basis, pixel, radius, cell_stride=stride)
    got = set()
    for layer in local:
        for x, y in layer["points"]:
            p = roi.lattice_tf.transform(Point2D.Double(y, x), None)   # (Spalte, Zeile)
            col, row = p.getX(), p.getY()
            if -0.001 <= row < H and -0.001 <= col < W:
                got.add((round(row, 2), round(col, 2)))

    missing, extra = len(want - got), len(got - want)
    status = "ok" if (missing == 0 and extra == 0) else "ABWEICHUNG"
    IJ.log("%-6s rot %7.2f off %-14s a %-6s px %.5f : %5d Atome, fehlen %d, zuviel %d  %s"
           % (kind[:6], rot, str(off), str(a_over), pixel, len(want), missing, extra, status))
    return missing == 0 and extra == 0

ok = True
ok &= check("Graphen", "WS2", 0.0, (0.0, 0.0), None, 0.012)
ok &= check("Graphen", "WS2", -25.412, (0.0, 0.0), None, 0.010614)
ok &= check("Graphen", "WS2", 17.5, (33.0, -21.0), None, 0.012)
ok &= check("Graphen", "WS2", 137.0, (-80.0, 45.0), 2.6, 0.015)
ok &= check("hBN", "WS2", -12.0, (5.0, 5.0), None, 0.013)
ok &= check("2H-MX2 (WS2, MoS2 ...)", "WS2", 30.0, (10.0, -10.0), None, 0.014)
IJ.log("GESAMT: %s" % ("alle gleich" if ok else "FEHLER"))

if result_path:
    handle = open(result_path, "w")
    handle.write("ok\n" if ok else "fehler\n")
    handle.close()
System.exit(0 if ok else 1)
