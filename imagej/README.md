# ImageJ-Plugin: Simulation aus einem Bild initialisieren

Sammelt im aktiven ImageJ-Fenster alles, was eine Simulation braucht, und
schreibt es als `init.json` neben das Bild. `mains/imagej_main.py` baut daraus
Config, TrainMethod und Modell und startet die Trainings-GUI.

## Installation

Das Plugin ist Jython, braucht also **Fiji** (das blanke ImageJ 1.x kann kein
Python). Zwei Dateien kopieren:

```bash
cp imagej/TEM_Sim_Init.py   ~/Fiji.app/scripts/Plugins/TEM_Simulation/
cp imagej/tem_sim_lattice.py ~/Fiji.app/jars/Lib/
```

Danach steht es in Fiji unter **Plugins > TEM Simulation > TEM Sim Init**.
Auf dieser Maschine ist beides schon installiert (`~/Fiji.app`).

## Ablauf

1. Bild in Fiji oeffnen und zurechtschneiden.
2. Bereiche als ROI markieren und in den ROI-Manager legen. Der **Name des ROI
   wird zum Namen der Gruppe** - so bekommt jede Struktur ihre eigene Maske.
   Ohne ROI laeuft der Loss ueber das ganze Bild.
3. Plugin starten. Es arbeitet auf dem aktiven Fenster.
4. Erster Dialog: Bild, Abbildung, Ausgabe - und **wie viele Gitter**.
5. Danach ein Dialog je Gitter: Typ, Gitterkonstante, Orientierung, Versatz,
   z-Versatz, Suchbereich, Debye-Waller, Elementmischung. Der Name des Gitters
   wird zum Namen seiner Gruppe.
   **Orientierung aus FFT** sucht den staerksten Gitterreflex (gemessen an
   synthetischen Bildern: Winkel auf 0.33 Grad, Pixelgroesse auf 0.4 % genau).
   Bei mehreren Gittern grenzt man mit den Suchradien den richtigen Ring ein.
6. Werte pruefen, Overlay kontrollieren (ein Farbton je Gitter), exportieren -
   wahlweise startet die Simulation gleich mit.

### Mehrere unabhaengige Gitter

Jedes Gitter bekommt eigene Orientierung, eigenen Versatz, eigenen z-Versatz,
eigenen Suchbereich und eine eigene Gruppe. Graphen mit einem WS2-Flake
darueber sieht dann so aus:

| Gitter | Typ | Orientierung | z-Versatz | Gruppe | Schichten |
|---|---|---|---|---|---|
| Gr | Graphen | aus FFT | 0.0 nm | Gr | Gr |
| WS2 | 2H-MX2 | eigener Wert | 0.33 nm | WS2 | S1, W, S2 |

Die Schichtnamen entstehen automatisch: ein Gitter mit einer Schicht heisst wie
das Gitter, sonst Element plus Nummer. Debye-Waller 0 heisst "Tabellenwert je
Element" - S und W schwingen unterschiedlich stark.

## Was eingestellt wird

| Bereich | Felder |
|---|---|
| Bild | Pixelgroesse, Hochspannung, Padding, Anzahl Gitter |
| Gitter (je Stueck) | Typ (Graphen, hBN, 2H-MX2, eigene Zelle, Strukturdatei), Gitterkonstante, Orientierung, Versatz, z-Versatz |
| Schichten | Name, Element, z, Gruppe, Start-Occupancy, Elementmischung (Dotierung), Debye-Waller |
| Gruppen | Suchbereich fuer Verschiebung und Drehung, Maske, Isolation |
| Abbildung | TEM (Multislice + CTF) oder STEM-HAADF, Aberrationen, Aperturwinkel, Amplitudenkontrast, Envelope |
| STEM | eigene Sonden-Aberrationen, Konvergenzwinkel, Detektorring innen/aussen, Quellgroesse |
| Training | aktive Gruppe, Schalter, Lernraten, Loss-Gewichte |

## Ohne Dialog (Skripte und Tests)

Alle Werte lassen sich auch aus einer Datei lesen:

```bash
~/Fiji.app/ImageJ-linux64 --headless --console \
  --run ~/Fiji.app/scripts/Plugins/TEM_Simulation/TEM_Sim_Init.py \
  "params_path='/pfad/zu/params.json'"
```

Die Parameterdatei enthaelt dieselben Werte wie der Dialog, dazu `image_path`
und optional `fft_estimate: true` sowie `rois: [{"name": "Gr", "rect": [x, y, w, h]}]`.

## Test

```bash
python3 imagej/test_imagej_chain.py
```

Erzeugt ein Bild aus einem bekannten Gitter, laesst es vom Plugin
initialisieren und prueft, ob das Modell daraus dasselbe Bild liefert:

```
TEM exakt   : -12.000 Grad (wahr -12.0), 0.015000 nm, NCC 1.0000
TEM aus FFT : -11.668 Grad (wahr -12.0), 0.015030 nm, NCC 0.9217
STEM aus FFT: -11.668 Grad (wahr -12.0), 0.015030 nm, NCC 0.8790
zwei Gitter : ['Gr','S1','W','S2'] in ['Gr','WS2'], z = [0.0, 0.33, 0.4875, 0.645]
```

## Weiterverarbeiten

```bash
python3 mains/imagej_main.py /pfad/zu/init.json          # GUI
python3 mains/imagej_main.py /pfad/zu/init.json --no-gui # nur aufbauen und pruefen
```

Config und TrainMethod landen reproduzierbar in `config_imagej/` neben der
`init.json`.

## Grenzen

- Die FFT-Schaetzung braucht sichtbare Gitterreflexe. Defokus und Astigmatismus
  liessen sich aus den Thon-Ringen schaetzen - dafuer braucht es aber amorphen
  Untergrund im **TEM**-Bild; an einem STEM-Bild gibt es diese Ringe nicht.
- Zugeschnitten wird vorher in ImageJ, das Plugin nimmt das aktive Bild wie es ist.
- Strukturdateien (.cif/.xyz) bekommen kein Overlay - die Schichten legt erst
  die Python-Seite an, weil das Plugin kein ase hat.
