# Core und GUI auf getrennten Rechnern

Core (Simulation, PyTorch, GPU) und GUI (Qt) sind getrennte Prozesse und sprechen nur
über einen ZeroMQ-Socket (Protokoll: `temsim/service/protocol.py`, Doku Teil IV).
Der Core rechnet z. B. auf dem GPU-Server, die GUI läuft auf dem Laptop.

```
Laptop                                   GPU-Server
python -m temsim_gui --endpoint ...  <-- SSH-Tunnel -->  python -m temsim.service serve ...
(Qt, matplotlib, pyzmq, numpy)                            (torch, pyzmq, numpy, ...)
```

## 1. Voraussetzungen

| Rechner | benötigt |
|---|---|
| Server (Core) | Repository, Python mit `torch` (CUDA), `numpy`, `scipy`, `pyzmq` |
| Laptop (GUI)  | Repository, Python mit `PySide6`, `matplotlib`, `numpy`, `pyzmq` – **kein torch** |

Beide Seiten brauchen denselben Stand des Repositorys (die GUI importiert `temsim.data`
und `temsim.service.client/protocol`). Die Protokollversion wird beim Verbinden geprüft.

## 2. Core auf dem Server starten

```bash
cd TEM_gradient_simulation
python -m temsim.service serve --endpoint tcp://127.0.0.1:5555 --device cuda
# -> "temsim core on tcp://127.0.0.1:5555 (device cuda)"
```

Optionen: `--device auto|cpu|cuda`, `--frame-interval 1.5` (Sekunden zwischen Bildern an die
GUI), `--save-interval 120` (Autosave in `<Run-Ordner>/autosave`).

Damit der Core eine SSH-Sitzung überlebt, in `tmux`/`screen` starten:

```bash
tmux new -s temsim
python -m temsim.service serve --endpoint tcp://127.0.0.1:5555 --device cuda
# Ctrl-b d  zum Abhängen,  tmux attach -t temsim  zum Zurückkehren
```

Der Core bindet bewusst nur an `127.0.0.1`. Es gibt **keine Authentifizierung**: Wer den Port
erreicht, kann Pfade auf dem Server öffnen und beschreiben. Den Port daher nicht ins Netz
öffnen (kein `0.0.0.0`), sondern per SSH-Tunnel erreichen.

## 3. Tunnel vom Laptop aufbauen

```bash
ssh -N -L 5555:127.0.0.1:5555 benutzer@gpu-server
# hinter einem Jump-Host:  ssh -N -J benutzer@jumphost -L 5555:127.0.0.1:5555 benutzer@gpu-server
```

Das Fenster offen lassen (oder `-f` für den Hintergrund). Lokaler Port und Server-Port müssen
nicht gleich sein, z. B. `-L 6000:127.0.0.1:5555` und dann `tcp://127.0.0.1:6000`.

Test: `python -m temsim.service ping --endpoint tcp://127.0.0.1:5555` gibt Status und
Beschreibung des Cores aus (braucht auf dem Laptop nur `pyzmq`, kein torch).

## 4. GUI auf dem Laptop starten

```bash
python -m temsim_gui --endpoint tcp://127.0.0.1:5555 --remote
```

`--remote` sagt der GUI, dass der Core auf einem anderen Rechner läuft (bei einem Tunnel sieht
sie nur `127.0.0.1` und kann das nicht selbst erkennen). Ohne `--remote` zeigt sie lokale
Datei-Dialoge. Ohne `--endpoint` startet die GUI einen eigenen lokalen Core (`--device auto|cpu|cuda`).

Daten öffnen: `--run <Ordner>` (gespeicherter Run) bzw. `--dataset <Datei/Ordner>`.
**Alle Pfade gelten auf dem Server**, nicht auf dem Laptop:

```bash
python -m temsim_gui --endpoint tcp://127.0.0.1:5555 --remote --run /data/ws2/run_07
```

Dasselbe gilt für "Parameter laden/speichern unter": bei einem entfernten Core fragt die GUI
den Pfad als Text ab (statt eines lokalen Datei-Dialogs). Eingabebilder, Masken und
`init.json` müssen also vorher auf den Server, z. B.

```bash
rsync -av platelet_11/ benutzer@gpu-server:/data/platelet_11/
```

Bei `--remote` (oder einer Endpoint-Adresse, die nicht `localhost` ist) fragt die GUI Pfade als
Text ab; sonst durchsucht der Datei-Dialog den lokalen Rechner.

## 5. Verhalten beim Beenden und Trennen

- GUI schließen: laufender Job wird gestoppt und (Haken "Beim Schließen speichern") der Run
  auf dem Server gespeichert. Der Core läuft weiter und kann wieder verbunden werden.
- Tunnel bricht ab: der Job auf dem Server läuft weiter und speichert per Autosave. Tunnel neu
  aufbauen, GUI neu starten, `--endpoint` wie zuvor; ein bereits geöffneter Run ist dann
  weiterhin offen (Status wird beim Verbinden gelesen).
- Core beenden: `Ctrl-C` im tmux-Fenster.
- Mehrere GUIs können sich am selben Core anmelden; sie teilen sich dann denselben Zustand.

## 6. Bandbreite

Die GUI bekommt etwa alle `--frame-interval` Sekunden ein Bild (Simulation, Target, Differenz
und Atomlisten als float32). Für große Bilder über langsame Leitungen: Intervall erhöhen
(`--frame-interval 5`). Potentiale und Gradientenpfeile werden nur übertragen, wenn sie in der
GUI eingeschaltet sind.

## 7. Fehlersuche

- *GUI hängt beim Start*: Tunnel prüfen (`ping`), Port stimmt? Core läuft noch?
- *"unknown command"/Protokollfehler*: unterschiedlicher Repository-Stand auf den Rechnern.
- *Pfad nicht gefunden*: Pfad existiert auf dem Server nicht (siehe Abschnitt 4).
- Fehler im Core erscheinen in der Statusleiste der GUI; der vollständige Traceback steht
  im Terminal der GUI und im Log des Core.
