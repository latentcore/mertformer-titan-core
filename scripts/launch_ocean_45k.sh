#!/usr/bin/env bash
# =============================================================================
# MertFormer Titan — Ocean / 45K remote_bootstrap training launch (operator runbook)
# -----------------------------------------------------------------------------
# Companion to scripts/launch_8xb300.sh. That wrapper drives the OFFLINE_CLEAN lane
# (TITAN_OFFLINE=1, precomputed Top-K logits). THIS wrapper drives the README-recommended
# REMOTE_BOOTSTRAP lane (TITAN_OFFLINE=0, online gated teacher, Phase-0 skipped) and adds
# operator observability: a redacted env snapshot, cuda.lock, an 8x/sm_100/bf16 GPU assert,
# targeted pretests, verify_all, a readiness check, live nvidia-smi telemetry, and a
# post-run output bundle + presence check.
#
# This NEVER auto-trains. Default mode is preview. Training starts only with --go, on a real
# 8x Blackwell box, with HF_TOKEN supplied IN THE ENVIRONMENT (this file stores no secrets).
#
# Usage:
#   HF_TOKEN=hf_xxx WANDB_API_KEY=... bash scripts/launch_ocean_45k.sh --check-only  # readiness gate, no GPU spend
#   HF_TOKEN=hf_xxx                   bash scripts/launch_ocean_45k.sh --dry-run      # preview resolved plan/command
#   HF_TOKEN=hf_xxx WANDB_API_KEY=... bash scripts/launch_ocean_45k.sh --go           # ACTUALLY launch the 45K run
#   (extra args such as `--resume auto` are passed through to zero_touch_start.sh)
# =============================================================================
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "$ROOT_DIR/zero_touch_start.sh" ]]; then
  echo "[ocean] FATAL: not a repo root (zero_touch_start.sh missing): $ROOT_DIR"
  exit 2
fi

MODE="preview"
PASSTHRU=()
for a in "$@"; do
  case "$a" in
    --go)         MODE="go" ;;
    --check-only) MODE="check" ;;
    --dry-run)    MODE="dry" ;;
    *)            PASSTHRU+=("$a") ;;
  esac
done

PY="${TITAN_PYTHON:-.titan-venv/bin/python}"

# --- Secrets: read from the environment only; NEVER hard-coded in this file. ---
# Required for --check-only and --go (the gated teacher needs them); preview/dry do not.
if [[ "$MODE" == "go" || "$MODE" == "check" ]]; then
  : "${HF_TOKEN:?[ocean] FATAL: export HF_TOKEN (gated meta-llama/Llama-3.3-70B-Instruct) before --go/--check-only}"
fi

# --- remote_bootstrap / Ocean 45K canonical environment (online gated teacher). ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export ACCELERATE_CONFIG_FILE="$ROOT_DIR/repro/accelerate_8xgpu.yaml"
export TITAN_OFFLINE=0
export TITAN_INSTALL="${TITAN_INSTALL:-1}"
export TITAN_PROFILE="${TITAN_PROFILE:-stable}"
# [2026-07-09] removed no-op: `export TITAN_OCEAN_45K_LAUNCH=1` was read by no .py (dead env).
export TITAN_TEACHER_MODEL_ID="${TITAN_TEACHER_MODEL_ID:-meta-llama/Llama-3.3-70B-Instruct}"
export TITAN_REQUIRE_GATED_TEACHER=1
export TITAN_USE_PRECOMPUTED_LOGITS=0
export TITAN_SKIP_PHASE0=1
export TITAN_TOKEN_BUDGET_MODE=fixed_steps
export TITAN_MAX_STEPS="${TITAN_MAX_STEPS:-45000}"
export TITAN_TARGET_TOKENS_MIN="${TITAN_TARGET_TOKENS_MIN:-23592960000}"   # 45000 * 128 * 4096 = 23.59B ≈ target
export TITAN_BATCH_SIZE="${TITAN_BATCH_SIZE:-128}"
export TITAN_BATCH_SIZE_FALLBACKS="${TITAN_BATCH_SIZE_FALLBACKS:-128}"
# [2026-07-12] BACKLOG #32: opt-in guard, off by default (config/config.py), hard-fails a
# >5% planned-token overshoot instead of silently proceeding -- catches e.g. an inflated
# TITAN_BATCH_SIZE_FALLBACKS or a doubled TITAN_MAX_STEPS before it burns real GPU-hours.
export TITAN_STRICT_TOKEN_BUDGET=1
export TITAN_DATALOADER_PIN=1
export TITAN_DATALOADER_NONBLOCKING=1
export TITAN_FFN_PACK=1
export TITAN_MOE_PACK=1
export TITAN_MLA_KV_PACK=1
export TITAN_MOE_DISPATCH=parallel
export TITAN_LIQUID_FAST_PATH=0
export TITAN_LIQUID_TRAIN_IMPL=packed_pair
export MERTFORMER_LOWBIT_KERNEL=0
export MERTFORMER_FUSED_BACKWARD=0
export TITAN_LOG_INTERVAL="${TITAN_LOG_INTERVAL:-1}"
export TITAN_VAL_CHECK_INTERVAL="${TITAN_VAL_CHECK_INTERVAL:-1000}"
export TITAN_SAVE_INTERVAL="${TITAN_SAVE_INTERVAL:-1000}"
export TITAN_WANDB="${TITAN_WANDB:-0}"
export TITAN_PREFLIGHT_REQUIRE_SECRETS=1
export TITAN_PREFLIGHT_WRITE_CUDA_LOCK=1
export TITAN_PREFLIGHT_STRICT_CUDA_LOCK=1
export TITAN_PREFLIGHT_REQUIRE_STAGE_JSONL=0
export TITAN_PREFLIGHT_MIN_DISK_GB="${TITAN_PREFLIGHT_MIN_DISK_GB:-500}"
export TITAN_AUTO_RESUME="${TITAN_AUTO_RESUME:-1}"
export TITAN_RESUME_ALLOW_PARTIAL=0

mkdir -p logs/launch
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"

# --- preview / dry: no GPU, no secrets needed; show the plan and exit. ---
if [[ "$MODE" == "preview" ]]; then
  echo "[ocean] PREVIEW ONLY (remote_bootstrap lane). Re-run with --go on an 8xB300 box to launch the real 45K run."
  echo "[ocean] HF_TOKEN must be exported in the environment (this script stores no secrets)."
  exec bash zero_touch_start.sh --plan-only
fi
if [[ "$MODE" == "dry" ]]; then
  echo "[ocean] Dry-run: resolve plan + command, no training ..."
  exec bash zero_touch_start.sh --dry-run
fi

# --- check / go: bootstrap venv if missing. ---
if [[ ! -x "$PY" ]]; then
  bash scripts/bootstrap_venv.sh 2>&1 | tee "logs/launch/bootstrap_${RUN_ID}.log"
fi
if [[ ! -x "$PY" ]]; then
  echo "[ocean] FATAL: venv python not found: $PY"
  exit 2
fi

# --- redacted env snapshot (secrets masked). ---
env | sort \
  | grep -E "^(TITAN_|MERTFORMER_|ACCELERATE_CONFIG_FILE|CUDA_DEVICE_ORDER|NCCL_|TORCH_NCCL_|HF_TOKEN|WANDB_API_KEY)=" \
  | sed -E 's/^(HF_TOKEN|WANDB_API_KEY)=.*/\1=<REDACTED>/' \
  > "logs/launch/env_snapshot_${RUN_ID}.txt"

# --- cuda.lock (hardware identity pin). ---
"$PY" scripts/write_cuda_lock.py 2>&1 | tee "logs/launch/cuda_lock_${RUN_ID}.log"

# --- 8x Blackwell (sm_100) / bf16 GPU assert. ---
"$PY" - 2>&1 <<'PYEOF' | tee "logs/launch/gpu_assert_${RUN_ID}.log"
import sys
try:
    import torch
except Exception as e:  # noqa: BLE001
    print(f"[ocean] FATAL: torch import failed: {e}"); sys.exit(2)
if not torch.cuda.is_available():
    print("[ocean] FATAL: CUDA not available (this lane targets a rented 8xB300 box)."); sys.exit(2)
n = torch.cuda.device_count()
if n != 8:
    print(f"[ocean] FATAL: need 8 GPUs, saw {n}."); sys.exit(2)
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    gib = p.total_memory / (1024 ** 3)
    maj, _ = torch.cuda.get_device_capability(i)
    print(f"[ocean] gpu{i} {p.name} sm_{maj}x {gib:.1f}GiB")
    if maj < 10:
        print(f"[ocean] FATAL: gpu{i} sm_{maj}x < sm_100 (B300/Blackwell required)."); sys.exit(2)
    if gib < 250:
        print(f"[ocean] FATAL: gpu{i} {gib:.1f}GiB looks under B300-class."); sys.exit(2)
if not torch.cuda.is_bf16_supported():
    print("[ocean] FATAL: bf16 not supported."); sys.exit(2)
print("[ocean] OK: 8x Blackwell bf16")
PYEOF

# --- targeted pretests (fast; fail before burning GPU-hours). ---
"$PY" -m pytest -q \
  tests/test_final_orchestrator_cli.py \
  tests/test_packed_projection_equivalence.py \
  tests/test_liquid_safeguard.py \
  tests/test_build_training_outputs_bundle.py \
  2>&1 | tee "logs/launch/pretests_${RUN_ID}.log"

# --- offline structural gate (teacher mocked). ---
env TITAN_OFFLINE=1 TITAN_WANDB=0 bash scripts/verify_all.sh \
  2>&1 | tee "logs/launch/verify_all_${RUN_ID}.log"

# --- readiness gate on THIS machine (no GPU spend). ---
bash zero_touch_start.sh --check-only \
  2>&1 | tee "logs/launch/check_only_${RUN_ID}.log"

if [[ "$MODE" == "check" ]]; then
  echo "[ocean] --check-only complete. Re-run with --go to launch the real 45K run (real GPU-hours)."
  exit 0
fi

# --- MODE == go: the real run, with live GPU telemetry. ---
echo "[ocean] LAUNCHING the real 45K run (remote_bootstrap lane) on 8xB300 ..."
nvidia-smi dmon -s pucvmet -d 10 -o TD > "logs/launch/nvidia_smi_dmon_${RUN_ID}.log" 2>&1 &
DMON_PID=$!
trap 'kill "$DMON_PID" 2>/dev/null || true' EXIT

TRAIN_RC=0
bash zero_touch_start.sh ${PASSTHRU[@]+"${PASSTHRU[@]}"} \
  2>&1 | tee "logs/launch/zero_touch_${RUN_ID}.log" || TRAIN_RC=$?

kill "$DMON_PID" 2>/dev/null || true
trap - EXIT

# --- post-run: bundle outputs + presence check. ---
"$PY" scripts/build_training_outputs_bundle.py 2>&1 | tee "logs/launch/bundle_${RUN_ID}.log" || true
if [[ -f artifacts/mertformer_training_outputs_bundle.zip ]]; then
  unzip -l artifacts/mertformer_training_outputs_bundle.zip \
    | grep -E "env_snapshot_|repro/cuda.lock|checkpoints|logs" \
    2>&1 | tee "logs/launch/bundle_presence_${RUN_ID}.log" || true
fi

exit "$TRAIN_RC"
