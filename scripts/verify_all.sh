#!/usr/bin/env bash
set -euo pipefail

# Single-command verification for review/readiness.
# Offline-first by default (no HF/WandB logins/downloads).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TITAN_OFFLINE="${TITAN_OFFLINE:-1}"
export TITAN_WANDB="${TITAN_WANDB:-0}"

if [[ -n "${TITAN_PYTHON:-}" ]]; then
  if [[ ! -x "${TITAN_PYTHON}" ]]; then
    echo "[verify] TITAN_PYTHON is set but not executable: ${TITAN_PYTHON}" >&2
    exit 2
  fi
  PY="${TITAN_PYTHON}"
else
  if [[ ! -x ".titan-venv/bin/python" ]]; then
    echo "[verify] .titan-venv missing; bootstrapping local Python venv ..."
    bash scripts/bootstrap_venv.sh
  fi
  PY=".titan-venv/bin/python"
fi

echo "[verify] Python: $("$PY" -V)"
echo "[verify] TITAN_OFFLINE=$TITAN_OFFLINE"

echo "[verify] Secret scan ..."
"$PY" scripts/secret_scan.py

echo "[verify] Pytest ..."
"$PY" -m pytest -q

echo "[verify] Preflight (offline) ..."
"$PY" scripts/titan_preflight.py

echo "[verify] Operator mode gate (safe, offline) ..."
"$PY" scripts/operator_mode_gate.py --no-pytest --overfit-dataset datasets/validation.jsonl

echo "[verify] OK"
