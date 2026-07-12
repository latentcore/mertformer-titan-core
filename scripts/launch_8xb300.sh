#!/usr/bin/env bash
# =============================================================================
# MertFormer Titan — 8x NVIDIA B300 final training launch (operator wrapper)
# -----------------------------------------------------------------------------
# Adds the one piece the repo lacks — a sm_100 (Blackwell) / cu128 / 8-GPU / bf16
# hardware assert — then sets the canonical 45K environment and DELEGATES to the
# existing pipeline:
#     zero_touch_start.sh -> (Phase-0 Top-K precompute) -> start_gate
#                         -> accelerate launch train/train.py -> eval -> bundle
#
# This NEVER auto-trains. Default mode is preview. Training starts only with --go,
# and only on a real 8x Blackwell box with HF_TOKEN for the gated Llama-3.3-70B teacher.
# Storage is the sharp edge: Top-K=256 teacher logits over ~23.6B tokens is TENS OF TB.
#
# Usage:
#   HF_TOKEN=hf_xxx bash scripts/launch_8xb300.sh --check-only   # readiness gate, no GPU spend
#   HF_TOKEN=hf_xxx bash scripts/launch_8xb300.sh --dry-run      # preview resolved plan/command
#   HF_TOKEN=hf_xxx bash scripts/launch_8xb300.sh --go           # ACTUALLY launch the 45K run
#   (extra args such as `--resume auto` are passed through to zero_touch_start.sh)
# =============================================================================
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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

# (1) HARD HARDWARE ASSERTS — the only genuinely new logic; the repo has no sm_100/cu128 check.
echo "[b300] Hardware assert (8x Blackwell sm_100, cu128, bf16) ..."
"$PY" - <<'PYEOF'
import sys
try:
    import torch
except Exception as e:  # noqa: BLE001
    print(f"[b300] FATAL: torch import failed: {e}")
    sys.exit(2)
if not torch.cuda.is_available():
    print("[b300] FATAL: CUDA not available (this launcher targets a rented 8xB300 box).")
    sys.exit(2)
n = torch.cuda.device_count()
if n != 8:
    print(f"[b300] FATAL: need 8 GPUs, saw {n}.")
    sys.exit(2)
maj = 0
for i in range(n):
    maj, _ = torch.cuda.get_device_capability(i)
    if maj < 10:
        print(f"[b300] FATAL: GPU{i} reports sm_{maj}x < sm_100 (B300/Blackwell required).")
        sys.exit(2)
cu = torch.version.cuda or ""
if not cu.startswith("12.8"):
    print(f"[b300] WARN: torch CUDA build is {cu!r}, expected cu128 (continuing — adjust knowingly).")
if not torch.cuda.is_bf16_supported():
    print("[b300] FATAL: bf16 not supported on this device.")
    sys.exit(2)
print(f"[b300] OK: 8x {torch.cuda.get_device_name(0)} sm_{maj}x cuda{cu} bf16")
PYEOF

# (2) HF_TOKEN required (gated meta-llama/Llama-3.3-70B-Instruct teacher).
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[b300] FATAL: HF_TOKEN is required (gated meta-llama/Llama-3.3-70B-Instruct teacher)."
  exit 2
fi

# (3) DISK REALITY — Top-K teacher logits over the full budget are enormous.
echo "[b300] NOTE: Top-K=${TITAN_TOP_K:-256} teacher logits over ~23.6B tokens is TENS OF TB"
echo "[b300]       (≈36 TB @ top_k=256 … ≈4.5 TB @ top_k=32). Lower TITAN_TOP_K or pre-stage shards;"
echo "[b300]       the disk pre-flight gate in scripts/precompute_logits_topk.py enforces this."

# (4) CANONICAL 8xB300 / 45K ENVIRONMENT (config defaults, set explicitly to be self-documenting).
export ACCELERATE_CONFIG_FILE="repro/accelerate_8xgpu.yaml"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TITAN_OFFLINE=1
export TITAN_REQUIRE_GATED_TEACHER=1
export TITAN_USE_PRECOMPUTED_LOGITS=1
export TITAN_USE_TR_TOKENIZER=0          # MUST be 0 — TR trips TOKENIZER_IDENTITY_DRIFT for precomputed KD
export TITAN_LOGITS_PATH="${TITAN_LOGITS_PATH:-./datasets/logits}"
export TITAN_BATCH_SIZE="${TITAN_BATCH_SIZE:-128}"
export TITAN_MAX_STEPS="${TITAN_MAX_STEPS:-45000}"
export TITAN_TARGET_TOKENS_MIN="${TITAN_TARGET_TOKENS_MIN:-23600000000}"
export TITAN_TOKEN_BUDGET_MODE=fixed_steps    # 45000 * 128 * 4096 = 23.59B tokens ≈ target
# [2026-07-12] BACKLOG #32: opt-in guard, off by default (config/config.py), hard-fails a
# >5% planned-token overshoot instead of silently proceeding -- catches e.g. an inflated
# TITAN_BATCH_SIZE_FALLBACKS or a doubled TITAN_MAX_STEPS before it burns real GPU-hours.
export TITAN_STRICT_TOKEN_BUDGET=1
export TITAN_PRECOMPUTE_GPUS="${TITAN_PRECOMPUTE_GPUS:-8}"
export TITAN_WANDB="${TITAN_WANDB:-0}"

# (5) DELEGATE to the existing pipeline — never re-implement training here.
case "$MODE" in
  check)
    echo "[b300] Readiness gate (no GPU spend) ..."
    exec bash zero_touch_start.sh --check-only
    ;;
  dry)
    echo "[b300] Dry-run: resolve plan + command, no training ..."
    exec bash zero_touch_start.sh --dry-run
    ;;
  preview)
    echo "[b300] PREVIEW ONLY. Re-run with --go to launch the real 45K run on 8xB300 (real GPU-hours)."
    exec bash zero_touch_start.sh --plan-only
    ;;
  go)
    echo "[b300] LAUNCHING the real 45K run on 8xB300 ..."
    exec bash zero_touch_start.sh ${PASSTHRU[@]+"${PASSTHRU[@]}"}
    ;;
esac
