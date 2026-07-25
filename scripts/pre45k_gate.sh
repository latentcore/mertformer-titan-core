#!/usr/bin/env bash
# =============================================================================
# scripts/pre45k_gate.sh -- cheap, pre-spend readiness gate for the real 45K launch.
#
# Chains three checks, all BEFORE any real GPU training budget is spent:
#   1. Offline preflight   (scripts/titan_preflight.py, zero GPU spend)
#   2. Dry-run preview     (zero_touch_start.sh --dry-run, zero GPU spend)
#   3. DDP smoke test      (scripts/ddp_smoke.py; real but short on genuine 2-GPU hosts,
#                           a clean non-blocking skip everywhere else)
#
# Rationale: today's only DDP-rank-sync assertion (train/train.py's "[Gate 3]") fires
# INSIDE the real training run, at the step-10000-class Liquid-unfreeze event -- i.e.
# after real budget has already been spent getting there. This script exists so a broken
# DDP path (including a B8-class silent no-op) fails cheaply first. See BACKLOG.md, B8.
#
# Usage:
#   bash scripts/pre45k_gate.sh                  # report-only; DDP result is informational
#   bash scripts/pre45k_gate.sh --strict-ddp     # also exit 1 if 2+ GPUs present but DDP unconfirmed
#   MERTFORMER_DDP_SMOKE_SECONDS=120 bash scripts/pre45k_gate.sh   # override the smoke budget
#
# Writes reports/pre45k_gate_report.json and reports/pre45k_gate_report.md.
# This NEVER launches real training -- it only decides whether the real launch scripts
# (scripts/launch_8xb300.sh, scripts/launch_ocean_45k.sh) should be trusted to run --go.
# =============================================================================
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${TITAN_PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x ".titan-venv/bin/python" ]]; then
    PY=".titan-venv/bin/python"
  else
    PY="python3"
  fi
fi

exec "$PY" -m scripts.pre45k_gate --python "$PY" "$@"
