#!/usr/bin/env bash
# Core in einer tmux-Sitzung (laeuft unabhaengig weiter), GUI als eigener Prozess dagegen.
# Benutzung:  ./start_temsim.sh [workspace] [port]      Core beenden:  tmux kill-session -t temsim-<port>
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HERE/venv/bin/python}"
WORKSPACE="${1:-$HOME/data/TEM_gradient_simulation}"
PORT="${2:-9098}"
ENDPOINT="tcp://127.0.0.1:$PORT"
SESSION="temsim-$PORT"

if ! "$PY" -m temsim.service ping --endpoint "$ENDPOINT" >/dev/null 2>&1; then
    tmux new -d -s "$SESSION" -c "$HERE" \
        "nice -n 5 '$PY' -m temsim.service serve --endpoint $ENDPOINT --device cuda --workspace '$WORKSPACE'; read -p 'Core beendet, Enter'"
    for _ in $(seq 60); do
        "$PY" -m temsim.service ping --endpoint "$ENDPOINT" >/dev/null 2>&1 && break
        sleep 1
    done
fi

cd "$HERE"
exec "$PY" -m temsim_gui --endpoint "$ENDPOINT"
