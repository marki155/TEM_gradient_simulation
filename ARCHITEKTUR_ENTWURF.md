# Architekturentwurf: Neubau des Simulationskerns

Status: Entwurf zur Freigabe. Es wurde noch kein Code umgebaut.

## 1. Leitgedanken

1. Der Quelltext ist die Arbeit. Jedes Modul beginnt mit einem Theorieabschnitt (Annahmen, Gleichungen, Herleitung in Kurzform). Der Code darunter setzt die nummerierten Gleichungen um und verweist auf sie (`# Gl. 2.7`).
2. Drei fachliche Teile mit je eigenem Theorieteil, dazu ein Dienst und die GUI als Client:
   - Teil I Physik: was ein Bild ist.
   - Teil II Lernen: wie Parameter und Atome aus einem Bild gefunden werden.
   - Teil III Daten: wie Modelle, Einstellungen und Datensätze beschrieben, gespeichert und migriert werden.
3. Abhängigkeiten laufen nur in eine Richtung. Das wird per Test erzwungen (Importprüfung).
4. Die GUI läuft in einem eigenen Prozess und kennt weder torch noch die Physik. Sie spricht nur das Protokoll.

## 2. Paketstruktur

```
temsim/
  data/            Teil III  (nur numpy/json, KEIN torch)
    schema.py        Datenklassen: Experiment, Imaging, Layer, Group, Lattice, TrainSettings, Regularisation
    store.py         Lesen/Schreiben: config.json, train_method.json, npz/npy/tif, xyz
    dataset.py       Datenordner (target, Masken, init.json) -> Experiment
    legacy.py        alle Migrationen alter Formate an EINER Stelle, versioniert (schema_version)
    testset.py       Erzeugung synthetischer Testsätze mit Wahrheit
  physics/         Teil I    (torch, kennt data.schema, sonst nichts)
    constants.py     CODATA-Konstanten, einmalig
    electron.py      relativistische Wellenlänge, Wechselwirkungsparameter sigma
    scattering.py    Atomformfaktoren: Kirkland, Lobato (auswählbar), projizierte Potentiale
    structure.py     Gitter (Graphen, MX2, xyz), Schichten, Gruppen, Atome -> Dichte
    strain.py        Verschiebungsfeld (Splines), Dehnungstensor, Elastizitätsenergie
    multislice.py    Transmission, Fresnel-Propagator (bandbegrenzt), Schichtstapel
    optics.py        Aberrationen, Kontrastübertragung, Sonde, Einhüllende
    imaging.py       TEM (kohärent) und STEM/HAADF (inkohärent, linear in den Atomen)
    window.py        lokale Neuberechnung (Fenster) für lineare Abbildung
  learning/        Teil II   (torch, kennt physics + data)
    parameters.py    trainierbare Parameter, Packen/Entpacken, Maskierung eingefrorener Teile
    objective.py     Verlustterme (MSE, NCC, SSIM, lokal ...), Normierung, Maskierung
    priors.py        Regularisierung: Repulsion, Entropie, Strain-Energie, Positionsshift
    optimiser.py     Lernratengruppen, Schedule
    trainer.py       Trainingsschritt, Zustand pausieren/fortsetzen, Gradientenvektoren
    seeding/
      store.py         AtomStore (einfügen/verschieben/entfernen mit exaktem Rücksetzen)
      collision.py     Berührungs-/Verschmelzungsregel
      proposals.py     Zufall, Gradientenlandschaft, Nachbar, Relax
      seeder.py        Akzeptanz über lokalen Loss
      minima.py        Minima der Differenz, Filterkriterien (alle optional)
  service/         Brücke    (kennt alles, stellt Prozessgrenze her)
    protocol.py      Nachrichten (Befehle, Ereignisse, Arrays binär)
    session.py       Fassade: ein Experiment laden, trainieren, seeden, editieren, speichern
    server.py        Prozess mit Arbeits-Thread; Pause/Parken wie bisher, aber im Kern
    cli.py           Kopfloser Betrieb (ersetzt mains/main_random_atoms.py u.ä.)
temsim_gui/        Client    (PySide6 + numpy + matplotlib; importiert nur data.schema + service.protocol)
    app.py             Start, Verbindung
    client.py          Gegenstelle des Protokolls, Qt-Signale
    state.py           Spiegel des Kernzustands (Bilder, Parameter, Atome)
    views/             CanvasGrid (Kacheln), Zoom, Gradienten, MinimaOverlay
    panels/            Training, Parameter, TrainMethod-Editor, Seeding, Minima, Speichern
    windows/           MainWindow, TestsetBuilder
docs/              Erzeugte Fassung der Arbeit (HTML aus den Modulen)
archive/legacy_core/   der heutige Kern (nn.py, potential.py, ...), unverändert, nur Referenz für Vergleiche
imagej/            Jython-Plugin bleibt, schreibt weiter init.json (Format bleibt lesbar)
```

Abhängigkeitsregel: `data` ← `physics` ← `learning` ← `service` ← (Prozessgrenze) → `temsim_gui`. Die GUI importiert `data.schema` und `service.protocol` (beide ohne torch), mehr nicht.

## 3. Gliederung der Arbeit

**Teil I – Physik**
1. Elektron-Probe-Wechselwirkung: relativistische Wellenlänge, sigma, Phasenobjekt-Näherung.
2. Atompotentiale: Kirkland/Lobato, Projektion, Faltung mit der Atomdichte, Debye-Waller, Abtastung (Supersampling).
3. Strukturmodell: Gitter, Schichten, Gruppen, Besetzung und Gewicht eines Atoms, Verschiebungsfeld und Dehnung.
4. Multislice: Propagator, Bandbegrenzung, Abtasttheorem, Konvergenz.
5. Abbildung: Kontrastübertragung, Aberrationen, Einhüllende; TEM kohärent; STEM/HAADF inkohärent und linear in den Atomen.
6. Lokale Neuberechnung: Linearität als Grundlage der Fenster-Simulation, Fehlerabschätzung der Kernbeschneidung.
7. Verifikation: Analytik (ebene Welle, Energieerhaltung, Einzelatom), Gradientenprüfung.

**Teil II – Lernen**
1. Inverses Problem und Zielfunktion: Normierung, Verlustterme, Maskierung, Eigenschaften (Skalen- und Offsetinvarianz).
2. Parametrisierung: Position, Intensität, Besetzung (sigmoid), Aberrationen, Strain; was trainiert wird und warum.
3. Regularisierung: Repulsion, Entropie, Positionsshift, Elastizität; Wirkung auf die Landschaft.
4. Optimierung: Adam, Lernraten je Parametergruppe, Schedule, Einfrieren.
5. Struktursuche (Seeding): Akzeptanzkriterium, Kollisionsregel, Vorschlagsverteilungen (Zufall, Gradient, Nachbar), Konvergenzverhalten.
6. Minima der Differenz: Definition, Filterkriterien als einzeln schaltbare Prädikate.
7. Verifikation: Gradienten gegen Differenzenquotient, lokal gleich voll, synthetische Testsätze mit Wahrheit.

**Teil III – Datenmanagement**
1. Modell der Daten: Experiment, Abbildungssetup, Schichten/Gruppen, Trainingseinstellungen.
2. Dateiformate und Versionierung (`schema_version`); Datenordner-Konvention.
3. Migration: alle Altformate lesbar, ein Eintrittspunkt, ein Test pro Altformat.
4. Reproduzierbarkeit: Seeds, gespeicherte Zustände, Wiederaufnahme.

**Teil IV – Dienst und Oberfläche** (kurz, technisch): Protokoll, Zustandsmodell, Threading, Fehlerverhalten.

## 4. Prozessgrenze

- Kern als eigener Prozess mit einem Arbeits-Thread (Training/Seeding), GUI als Client. Kein geteilter Speicher, kein geteiltes Modell.
- Nachrichten: kleiner JSON-Kopf plus binäre Arrays (dtype/shape im Kopf). Die GUI schickt Befehle (`load`, `start`, `pause`, `edit_parameter`, `edit_train`, `seed`, `insert_minima`, `save` ...) und erhält Ereignisse (`frame` mit Simulation/Ziel/Differenz/Potential/Atome/Gradienten, `loss`, `log`, `state`).
- Bisher teilen sich Worker und GUI-Thread dasselbe Modell und synchronisieren über `parked`/`pause`. Danach gibt es nur noch eine Besitzerin des Modells (der Arbeits-Thread im Kernprozess); Edits laufen als Befehle in dessen Warteschlange und werden zwischen zwei Schritten angewendet. Die Race-Condition (Atomzahl ändert sich mitten im Schritt) entfällt damit konstruktiv.
- Auch auf einem entfernten Rechner nutzbar (SSH-Tunnel auf den Port).

## 5. Physikalische Änderungen, die ich vorschlage (nach Freigabe je Punkt)

1. Konstanten einmalig (CODATA) statt drei verschiedener Wellenlängenformeln.
2. Propagator mit Bandbegrenzung (2/3-Regel) und Test auf Aliasing.
3. Lokaler Simulator ohne eigene Physikkopie: dieselben Kerne wie die Vollsimulation, Abweichung nur die Fensterung.
4. Lobato als wählbares Modell (heute importiert, aber ungenutzt).
5. Lokaler Simulator nicht mehr auf eine Schicht/ein Element beschränkt, soweit die Linearität es erlaubt.

## 6. Abgrenzung (nicht Teil des Neubaus)

- Tot bzw. veraltet, wird nicht übernommen: `simulation.py`, `tesster.py`, `testing.py`, `kirkland.py`, `strain.py`, `losses.py`, `NN_free_potential`, Tk-Builder, `mains/main*.py` (außer als Referenz).
- Analysewerkzeuge `structure_clustering.py`, `atomatic_features.py`: bleiben als eigenständige Skripte unter `analysis/`.

## 7. Vorgehen

0. Freigabe dieses Entwurfs.
1. Gerüst, Importtest, `data` (Schema, Store, Legacy) mit Test: alle bisherigen `config.json`/`train_method.json` laden.
2. `physics` mit Theorietext; Vergleich gegen `archive/legacy_core` (Bild, Gradient) und analytische Tests.
3. `learning` (Loss, Regularisierung, Trainer), dann Seeding/Minima; Vergleich Loss/Entscheidungen gegen den Altkern.
4. `service` (Protokoll, Server, CLI).
5. `temsim_gui`: Aufteilung von `gui_qt.py` in Klassen, Anbindung an das Protokoll; bestehende GUI-Tests auf den Client umstellen.
6. Altkern nach `archive/legacy_core`, Doku erzeugen.

Nach jedem Schritt Rückmeldung mit Teststand, bevor der nächste beginnt.
