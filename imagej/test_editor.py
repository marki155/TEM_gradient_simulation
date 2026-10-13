#@ String (label="Ergebnisdatei", value="", persist=false, required=false) result_path

"""Editor im echten Fenster: Dialog bauen, Felder aendern, mit der Maus ziehen.

Braucht ein Fiji MIT Oberflaeche (GenericDialog legt echte AWT-Bauteile an)
und dass keine andere Fiji-Instanz laeuft - sonst reicht der Starter die
Argumente nur weiter.

    DISPLAY=:0 ImageJ-linux64 --console --run imagej/test_editor.py
"""
import os, sys, math

REPO = os.environ.get(
    "TEM_SIM_REPO", "/home/koegelmarco/remote_python/TEM_gradient_simulation")
sys.path.append(os.path.join(REPO, "imagej"))
import java.lang.System as System
from ij import IJ, ImagePlus
from ij.gui import GenericDialog
from ij.process import ByteProcessor
from java.awt.event import MouseEvent
import TEM_Sim_Init as P

FAILS = []
LOG = open(result_path, "w") if result_path else None
def check(name, cond, detail=""):
    line = "  %-46s %s %s" % (name, "ok" if cond else "FEHLER", detail)
    IJ.log(line)
    if LOG is not None:
        LOG.write(line + "\n"); LOG.flush()
    if not cond:
        FAILS.append(name)

H, W = 700, 900
imp = ImagePlus("editor", ByteProcessor(W, H))
imp.show()

entry = {"name": "Gr", "kind": "Graphen", "formula": "WS2", "a": None, "gamma_deg": 120.0,
         "rotation_deg": 10.0, "offset_px": (0.0, 0.0), "fft_min_radius_px": 0.0,
         "fft_max_radius_px": 0.0}
entry2 = {"name": "WS2", "kind": "2H-MX2 (WS2, MoS2 ...)", "formula": "WS2", "a": None,
          "gamma_deg": 120.0, "rotation_deg": -5.0, "offset_px": (12.0, 8.0),
          "fft_min_radius_px": 0.0, "fft_max_radius_px": 0.0}
params = {"pixel_size_nm": 0.012, "lattices": [entry, entry2]}

overlay = P.LatticeOverlay(imp, params)
check("Overlay gebaut", overlay.total() > 1000, "%d Atome" % overlay.total())
check("ein Roi je Gitter", len([r for r in overlay.rois if r is not None]) == 2)

editor = P.LatticeEditor(imp, params, overlay)
editor.dragger = P.LatticeDragger(editor)

# Dialog genau wie in show() aufbauen, aber nicht anzeigen (das wuerde blockieren)
gd = GenericDialog("Gitter ausrichten")
editor.gd = gd
gd.addMessage("Test")
gd.addChoice("Gitter", [e["name"] for e in params["lattices"]], entry["name"])
gd.addSlider("Orientierung [Grad]", -180.0, 180.0, entry["rotation_deg"], 0.05)
gd.addSlider("Versatz Zeile [px]", -H, H, 0.0, 0.5)
gd.addSlider("Versatz Spalte [px]", -W, W, 0.0, 0.5)
gd.addSlider("Gitterkonstante a [A]", 2.46 * 0.7, 2.46 * 1.3, 2.46, 0.001)
gd.addNumericField("Pixelgroesse [nm]", params["pixel_size_nm"], 6)
gd.addCheckbox("mit der Maus verschieben", True)
gd.addCheckbox("Gitter zeigen", True)
gd.addMessage(overlay.note() + " " * 40)
editor.note_label = gd.getComponent(gd.getComponentCount() - 1)

fields = gd.getNumericFields()
check("5 Zahlenfelder in der Reihenfolge", fields is not None and fields.size() == 5,
      "gefunden %s" % (fields.size() if fields else None))
check("Label fuer die Meldung", hasattr(editor.note_label, "setText"),
      editor.note_label.getClass().getSimpleName())
editor.status()

# Regler aendern -> dialogItemChanged
editor.set_field("Orientierung [Grad]", 42.5, 3)
editor.set_field("Versatz Zeile [px]", -30.0, 2)
editor.set_field("Versatz Spalte [px]", 17.0, 2)
gd.resetCounters()
editor.dialogItemChanged(gd, None)
check("Drehung uebernommen", abs(entry["rotation_deg"] - 42.5) < 1e-6, str(entry["rotation_deg"]))
check("Versatz uebernommen", entry["offset_px"] == (-30.0, 17.0), str(entry["offset_px"]))

# Gitterkonstante und Pixelgroesse loesen einen Neuaufbau aus
before = overlay.total()
editor.set_field("Gitterkonstante a [A]", 3.0, 4)
gd.resetCounters()
editor.dialogItemChanged(gd, None)
check("a uebernommen", abs(entry["a"] - 3.0) < 1e-6, str(entry.get("a")))
check("Pfad neu gebaut", overlay.total() != before, "%d -> %d" % (before, overlay.total()))

# anderes Gitter waehlen
gd.getChoices().get(0).select(1)
gd.resetCounters()
editor.dialogItemChanged(gd, None)
check("Gitter gewechselt", editor.index == 1, "Index %d" % editor.index)
check("Felder nachgeladen", abs(float(fields.get(0).getText()) + 5.0) < 1e-6,
      "Orientierung %s" % fields.get(0).getText())

# Ziehen mit der Maus
canvas = imp.getCanvas()
check("Canvas da", canvas is not None)
editor.index = 0
start = tuple(entry["offset_px"])
dragger = editor.dragger
def event(x, y, kind):
    return MouseEvent(canvas, kind, System.currentTimeMillis(), 0, x, y, 1, False)
dragger.mousePressed(event(100, 100, MouseEvent.MOUSE_PRESSED))
dragger.mouseDragged(event(140, 125, MouseEvent.MOUSE_DRAGGED))
moved = (entry["offset_px"][0] - start[0], entry["offset_px"][1] - start[1])
expect = (canvas.offScreenYD(125) - canvas.offScreenYD(100),
          canvas.offScreenXD(140) - canvas.offScreenXD(100))
check("Ziehen verschiebt das Gitter",
      abs(moved[0]-expect[0]) < 1e-6 and abs(moved[1]-expect[1]) < 1e-6,
      "%s statt %s" % (str(moved), str(expect)))
dragger.mouseReleased(event(140, 125, MouseEvent.MOUSE_RELEASED))
check("Regler nach dem Ziehen nachgefuehrt",
      abs(float(fields.get(1).getText()) - entry["offset_px"][0]) < 0.02,
      "Feld %s, Gitter %.2f" % (fields.get(1).getText(), entry["offset_px"][0]))

# ImageJs Maus-Hoerer wieder da?
n_before = len(canvas.getMouseListeners())
editor.set_drag_mode(True)
during = len(canvas.getMouseListeners())
editor.set_drag_mode(False)
n_after = len(canvas.getMouseListeners())
check("Maus-Hoerer sauber getauscht", n_after == n_before and during >= 1,
      "vorher %d, waehrend %d, nachher %d" % (n_before, during, n_after))

# Knoepfe
editor.centre()
check("zentrieren setzt den Versatz auf 0", float(fields.get(1).getText()) == 0.0)
editor.reset()
check("zuruecksetzen holt den Startwert", abs(float(fields.get(0).getText()) - 10.0) < 1e-6,
      fields.get(0).getText())

IJ.log("FEHLER: %s" % (", ".join(FAILS) if FAILS else "keine"))
if LOG is not None:
    LOG.write("fails=%d\n" % len(FAILS)); LOG.close()
System.exit(1 if FAILS else 0)
