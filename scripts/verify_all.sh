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
pytest_output="$("$PY" -m pytest -q 2>&1)" || {
  printf '%s\n' "$pytest_output"
  exit 1
}
printf '%s\n' "$pytest_output"
export MERTFORMER_EXPECTED_TEST_STAT="$(printf '%s\n' "$pytest_output" | grep -oE '[0-9]+ passed, [0-9]+ skipped' | tail -n 1)"

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

echo "[verify] Markdown integrity gate ..."
"$PY" scripts/md_integrity_check.py --root .

echo "[verify] Master closure matrix refresh ..."
"$PY" scripts/build_master_closure_matrix.py

echo "[verify] Train readiness contract refresh ..."
"$PY" scripts/build_train_readiness_contract.py --allow-not-ready

echo "[verify] Final orchestrator contract refresh ..."
"$PY" scripts/final_orchestrator.py --plan-only

echo "[verify] Code-truth delta audit refresh ..."
"$PY" scripts/build_code_truth_audit.py

echo "[verify] Workspace hygiene manifest refresh ..."
"$PY" scripts/build_workspace_hygiene_manifest.py

echo "[verify] Scoped external intake audit ..."
"$PY" scripts/build_scoped_external_intake_matrix.py --sync-mode audit

echo "[verify] Chess GUI onefile sync check ..."
"$PY" scripts/sync_chess_gui_onefile.py --check-only

echo "[verify] Chess teaching contract smoke report ..."
"$PY" scripts/build_chess_teaching_contract_report.py

echo "[verify] Chess onefile extension report ..."
"$PY" scripts/build_chess_onefile_extension_report.py

echo "[verify] Chess training readiness report ..."
"$PY" scripts/build_chess_training_readiness_report.py

echo "[verify] Closure governance pack refresh ..."
"$PY" scripts/build_closure_governance_pack.py

echo "[verify] Start gate operator decision refresh ..."
"$PY" scripts/start_gate.py --skip-verify-all --allow-not-ready

echo "[verify] Target machine handoff bundle refresh ..."
"$PY" scripts/build_target_machine_handoff_bundle.py

echo "[verify] Max closure handoff refresh ..."
"$PY" scripts/build_max_closure_handoff.py



echo "[verify] Unicode path guard gate ..."
"$PY" scripts/unicode_path_guard.py --root . --out reports/unicode_path_guard_report.json --fail-on-hit

echo "[verify] Duplicate zip guard gate ..."
"$PY" scripts/duplicate_zip_guard.py --root packages --root artifacts --out reports/duplicate_zip_guard_report.json

echo "[verify] Manifest sync gate ..."
"$PY" scripts/sync_manifest.py --root . --manifest reports/release_manifest.json --structure docs/PROJECT_STRUCTURE.md --matrix reports/file_sync_matrix.json --sync-report reports/project_structure_sync_report.json --policy-report reports/policy_sync_report.json

echo "[verify] OK"
