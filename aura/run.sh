#!/usr/bin/env bash
# AURA — one-shot launcher (macOS/Linux).
set -e
# The repository root, not aura/, is the import root: every module is addressed as
# aura.<...>, so `python -m` has to be invoked from one level up.
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
echo "[AURA] installing dependencies (first run may take a minute) ..."
$PY -m pip install -q -r aura/requirements.txt
echo "[AURA] training models + running quantum-vs-classical benchmark ..."
$PY -m aura.aura_cli train
$PY -m aura.aura_cli bench
echo "[AURA] starting gateway + dashboard on http://127.0.0.1:8000"
$PY -m aura.aura_cli serve
