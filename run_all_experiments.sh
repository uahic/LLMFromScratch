#!/usr/bin/env bash
set -euo pipefail

DONE_DIR=".experiment_done"
mkdir -p "$DONE_DIR"

shopt -s nullglob
configs=(experiments/*.json)
if [[ ${#configs[@]} -eq 0 ]]; then
    echo "No experiment configs found in experiments/"
    exit 0
fi

for cfg in "${configs[@]}"; do
    name=$(basename "$cfg" .json)
    marker="$DONE_DIR/$name.done"

    if [[ -f "$marker" ]]; then
        echo "[SKIP] $cfg already completed"
        continue
    fi

    echo "[RUN] $cfg"
    if python -u train.py "$cfg"; then
        touch "$marker"
        echo "[DONE] $cfg"
    else
        echo "[FAIL] $cfg — will retry on next run"
    fi
done
