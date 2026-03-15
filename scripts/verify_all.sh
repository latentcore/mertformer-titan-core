#!/usr/bin/env bash
set -euo pipefail

# Single-command verification for review/readiness.
# Offline-first by default (no HF/WandB logins/downloads).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TITAN_OFFLINE="${TITAN_OFFLINE:-1}"
export TITAN_WANDB="${TITAN_WANDB:-0}"
if [[ "${TITAN_OFFLINE}" == "1" ]]; then
  export TITAN_PREFLIGHT_ALLOW_MISSING_STAGE_JSONL="${TITAN_PREFLIGHT_ALLOW_MISSING_STAGE_JSONL:-1}"
fi

if [[ -n "${TITAN_PYTHON:-}" ]]; then
  if [[ -x "${TITAN_PYTHON}" ]]; then
    PY="${TITAN_PYTHON}"
  elif command -v "${TITAN_PYTHON}" >/dev/null 2>&1; then
    PY="$(command -v "${TITAN_PYTHON}")"
  else
    echo "[verify] TITAN_PYTHON is set but not executable/in PATH: ${TITAN_PYTHON}" >&2
    exit 2
  fi
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
mkdir -p logs/verify
"$PY" scripts/check_57_matrix.py \
  --require-no-pending \
  --out logs/verify/closure_57_matrix.verify.json \
  --md-out logs/verify/closure_57_matrix.verify.md \
  --md-tr-out logs/verify/closure_57_matrix.verify_TR.md

echo "[verify] Tokenizer sync gate ..."
"$PY" scripts/check_tokenizer_sync.py

echo "[verify] Translation pointer policy gate ..."
"$PY" scripts/check_translation_pointer_policy.py

echo "[verify] Documentation claim consistency gate ..."
"$PY" scripts/check_doc_claim_consistency.py



echo "[verify] Unicode path guard gate ..."
"$PY" scripts/unicode_path_guard.py --root . --out reports/unicode_path_guard_report.json --fail-on-hit

echo "[verify] Duplicate zip guard gate ..."
"$PY" scripts/duplicate_zip_guard.py --root packages --root artifacts --out reports/duplicate_zip_guard_report.json

echo "[verify] Manifest sync gate ..."
"$PY" scripts/sync_manifest.py --root . --manifest reports/release_manifest.json --structure docs/PROJECT_STRUCTURE.md --matrix reports/file_sync_matrix.json --sync-report reports/project_structure_sync_report.json --policy-report reports/policy_sync_report.json

echo "[verify] OK"
