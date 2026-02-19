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

echo "[verify] Closure 57 matrix gate (strict in-scope pending) ..."
"$PY" scripts/check_57_matrix.py --require-no-pending

echo "[verify] Tokenizer sync gate ..."
"$PY" scripts/check_tokenizer_sync.py

echo "[verify] Translation pointer policy gate ..."
"$PY" scripts/check_translation_pointer_policy.py

echo "[verify] Documentation claim consistency gate ..."
"$PY" scripts/check_doc_claim_consistency.py

echo "[verify] OK"
