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

Optionen: `--workspace <Ordner>` (siehe unten), `--device auto|cpu|cuda`, `--frame-interval 1.5` (Sekunden zwischen den vollständigen Bildern an die
GUI), `--frame-steps 10` (Training: Bild alle N Schritte), `--save-interval 120` (Autosave in `<Run-Ordner>/autosave`).

**Workspace.** Mit `--workspace ~/daten` hat der Core einen festen Ordner für Laden und Speichern. Die GUI
(Knopf „Laden ...“) zeigt daraus eine Liste der Läufe und Datenordner (bis 3 Ebenen tief, neueste zuerst);
Autosave-Ordner werden nicht aufgelistet, ein Datenordner mit Autosave macht dort weiter (Haken „neu
beginnen“ ignoriert ihn). „Speichern unter ...“ fragt nur noch einen Namen und schreibt nach
`<Workspace>/<Name>` (ein vorhandener Datenordner wird nie überschrieben). Relative Pfade in `--open`
und in den Befehlen gelten ebenfalls relativ zum Workspace. Ohne `--workspace` fragt die GUI wie bisher
nach einem Pfad auf dem Core-Rechner.

**Speichern und Fortsetzen.** Der Core schreibt den Stand nach `<Ausgabeordner>/autosave`
(Seeding: `<Datenordner>/config_random/autosave`, Run: `<Run-Ordner>/autosave`) bei Pause, Stop,
Jobende, Fehler, Edits (Minima einfügen, Atome entfernen ...), vor dem Öffnen von etwas anderem,
beim Beenden und alle `--save-interval` Sekunden. Wird derselbe Datenordner erneut geöffnet
(`--open <Ordner>` bzw. `--dataset`), macht der Core automatisch dort weiter ("continued from ..." in
den Notizen); `--fresh` ignoriert den Autosave. Ein Autosave ist ein normaler Run und lässt sich auch
mit `--run <Ordner>/autosave` öffnen.

Schließt man die GUI, die an einen fremden Core (`--endpoint`) angeschlossen ist, läuft dessen Job
weiter und speichert weiter; nur ein von der GUI selbst gestarteter Core wird mit ihr beendet.
"Beim Schliessen speichern" schreibt ebenfalls nach `<Ausgabeordner>/autosave`.

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

### Gemeinsames venv für den Core (mehrere Benutzer)

Torch liegt derzeit im Benutzerpaketordner `~/.local` eines einzelnen Benutzers; andere sehen es nicht.
Ein venv **außerhalb** des Projektordners (der Pfad ist in dessen Skripten fest eingetragen) löst das:

```bash
python3 -m venv /home/remote_python/envs/temsim
/home/remote_python/envs/temsim/bin/pip install -r requirements-core.txt
sudo chgrp -R remote_python /home/remote_python/envs && sudo chmod -R g+rX /home/remote_python/envs
# Core starten (ohne activate, der Pfad genügt):
/home/remote_python/envs/temsim/bin/python -m temsim.service serve --endpoint tcp://127.0.0.1:9097 --device cuda --workspace ...
```

### Komfortabel: `~/.ssh/config` auf dem Laptop

```
Host emdata
    HostName <voller-name-von-emdata>
    User koegelmarco

Host hillier
    HostName hillier
    User koegelmarco
    ProxyJump emdata
    # Tunnel gleich mitnehmen: GUI-Endpoint tcp://127.0.0.1:9097
    LocalForward 9097 127.0.0.1:9097
    ServerAliveInterval 30
```

Dann gibt `ssh hillier` direkt eine Konsole auf dem GPU-Server (und baut nebenbei den Tunnel auf);
`ssh -t hillier tmux new -A -s temsim` hängt sich direkt an die tmux-Sitzung des Cores,
`ssh -N hillier` baut nur den Tunnel.

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

Beim Training schickt der Core alle `--frame-steps 10` Schritte die Bilder **aus dem Trainingsschritt
selbst** (Simulation, Differenz und Bilder x/y als float16, Atome als float32; Target und Maske nur
einmal, danach nur wenn sie sich ändern; grob 17–21 MB bei 2048² statt 50 MB).
Aufbereitung und Versand laufen in einem eigenen Thread, das Training wartet nicht darauf; ein
neueres Bild ersetzt ein noch nicht verschicktes. Die GUI bestätigt jedes gezeichnete Bild; der Core
schickt höchstens zwei unbestätigte, eine langsame Leitung oder ein langsamer Laptop bremst also nie
das Training. Optik, Dehnung, Potentiale und Gradientenpfeile brauchen eine eigene Simulation und
kommen weiterhin nur alle `--frame-interval` Sekunden (und nur, wenn die GUI sie anfordert); dazwischen
behält die GUI die letzten. `--frame-steps 0` schaltet das schnelle Verfahren ab. Beim Seeding gilt
weiter `--frame-interval`.

## 7. Fehlersuche

- *GUI hängt beim Start*: Tunnel prüfen (`ping`), Port stimmt? Core läuft noch?
- *"unknown command"/Protokollfehler*: unterschiedlicher Repository-Stand auf den Rechnern.
- *Pfad nicht gefunden*: Pfad existiert auf dem Server nicht (siehe Abschnitt 4).
- Fehler im Core erscheinen in der Statusleiste der GUI; der vollständige Traceback steht
  im Terminal der GUI und im Log des Core.
