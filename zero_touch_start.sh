#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "--offline-4060-demo" || "${1:-}" == "--chess-5080-poc" || "${1:-}" == "--chess-5080-delivery-export" || "${1:-}" == "--kaggle-onefile" ]]; then
  exec bash "$ROOT_DIR/run.sh" "$@"
fi

PY="${TITAN_PYTHON:-}"
if [[ -n "$PY" ]]; then
  if [[ ! -x "$PY" ]] && ! command -v "$PY" >/dev/null 2>&1; then
    echo "[zero-touch] TITAN_PYTHON is set but not executable/in PATH: $PY" >&2
    exit 2
  fi
else
  if [[ ! -x ".titan-venv/bin/python" ]]; then
    if [[ -f "scripts/bootstrap_venv.sh" ]]; then
      echo "[zero-touch] bootstrapping .titan-venv ..."
      bash scripts/bootstrap_venv.sh
    fi
  fi

  if [[ -x ".titan-venv/bin/python" ]]; then
    PY=".titan-venv/bin/python"
  else
    PY="python3"
  fi
fi

should_run_phase0=1
for arg in "$@"; do
  case "$arg" in
    --plan-only|--dry-run|--check-only|--post-only|--bench-only|--export-only|--demo-only|--readme-update-only)
      should_run_phase0=0
      ;;
  esac
done

LOGITS_DIR="${TITAN_LOGITS_PATH:-./datasets/logits}"
SKIP_PHASE0="${TITAN_SKIP_PHASE0:-0}"

if [[ "$should_run_phase0" == "0" ]]; then
  echo "[phase-0] skipping optional precompute for non-training invocation."
elif [[ "$SKIP_PHASE0" == "1" ]]; then
  echo "[phase-0] TITAN_SKIP_PHASE0=1, optional precompute skipped."
else
  phase0_base=( "$PY" scripts/precompute_logits_topk.py --all-stages --logits-dir "$LOGITS_DIR" )

  echo "[phase-0] checking dataset availability for offline teacher logits ..."
  if "${phase0_base[@]}" --dry-run >/dev/null 2>&1; then
    if "${phase0_base[@]}" --check-complete >/dev/null 2>&1; then
      echo "[phase-0] offline teacher logits already complete."
      # [B2] Verify teacher-logit shards align to the student stream (identity).
      # Advisory here; preflight/has_precomputed_logits enforce it authoritatively.
      if "$PY" scripts/validate_logit_alignment.py --all-stages --logits-dir "$LOGITS_DIR" >/dev/null 2>&1; then
        echo "[phase-0] logit alignment verified."
      else
        echo "[phase-0] WARNING: logit alignment check did not pass; readiness gate will block until shards are realigned (re-run precompute)."
      fi
    elif [[ -z "${HF_TOKEN:-}" ]]; then
      echo "[phase-0] HF_TOKEN missing; optional offline teacher precompute skipped."
      echo "[phase-0] final orchestrator / readiness policy will block canonical offline_clean until logits exist or Phase-0 becomes actionable."
    else
      echo "[phase-0] launching offline teacher logits precompute ..."
      precompute_cmd=( "${phase0_base[@]}" )
      if [[ -n "${TITAN_TOP_K:-}" ]]; then
        precompute_cmd+=( --top-k "$TITAN_TOP_K" )
      fi
      if [[ -n "${TITAN_PRECOMPUTE_BATCH:-}" ]]; then
        precompute_cmd+=( --batch-size "$TITAN_PRECOMPUTE_BATCH" )
      fi

      if "${precompute_cmd[@]}"; then
        echo "[phase-0] offline teacher logits precompute finished."
      else
        phase0_rc=$?
        if [[ "$phase0_rc" -eq 130 ]]; then
          echo "[phase-0] interrupted by user; resume is preserved."
          exit 130
        fi
        echo "[phase-0] precompute failed with rc=$phase0_rc; continuing to canonical orchestrator."
      fi
    fi
  else
    echo "[phase-0] stage datasets are missing or unreadable; optional precompute skipped."
  fi
fi

exec "$PY" scripts/final_orchestrator.py "$@"
