#!/usr/bin/env bash
# AURA — one-shot launcher (macOS/Linux).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
echo "[AURA] installing dependencies (first run may take a minute) ..."
$PY -m pip install -q -r requirements.txt
echo "[AURA] training models + running quantum-vs-classical benchmark ..."
$PY -m aura_cli train
$PY -m aura_cli bench
echo "[AURA] starting gateway + dashboard on http://127.0.0.1:8000"
$PY -m aura_cli serve
