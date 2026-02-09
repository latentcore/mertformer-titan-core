# Scripts Catalog

All scripts are designed to run from the repo root.

Conventions:
- Prefer `.titan-venv/bin/python scripts/<name>.py` (see `scripts/bootstrap_venv.sh`).
- Offline-first defaults: `TITAN_OFFLINE=1` (no HF/WandB logins or dataset downloads unless explicitly enabled).

If you are unsure, run the single-command verification first: `bash scripts/verify_all.sh`.

## Core Pipelines
- `smart_runner.py` — Master orchestrator: data → distill → train.
- `data_pipeline.py` — Dataset preparation (5-stage curriculum).
- `titan_preflight.py` — End-to-end preflight verification.
- `operator_mode_gate.py` — Single-entry ops gate (safety + sanity checks).
- `overfit_gate.py` — 1MB overfit gate (safe/full modes).
- `train_smoke.py` — Tiny offline training sanity loop (CPU/MPS).

## Review-Ready Tooling
- `bootstrap_venv.sh` — Creates `.titan-venv` (Python 3.11 baseline). Use `--demo` to install `pygame`.
- `verify_all.sh` — Offline-first verify-all: secret scan → pytest → preflight → operator gate (safe).
- `secret_scan.py` — Scans tracked files for common secret patterns (CI gate).

## Evaluation & Benchmarks
- `golden_eval.py` — Golden sample evaluator (50 prompts).
- `benchmarks_internal.py` — HumanEval / MBPP output generator (SKIP if checkpoint/datasets are unavailable).
- `bitnet_kernel_benchmark_standalone.py` — Single-file standalone BitNet ternary kernel benchmark (kernel + quantization + benchmark harness in one file).
- `eval.py` — GSM8K eval wrapper (legacy; see `eval/gsm8k.py`).

## Export & ONNX
- `mobile_export.py` — ONNX export for mobile/edge.
- `test_onnx_export.py` — ONNX export test.
- `verify_onnx_local.py` — Local ONNX verification.
- `titan_onnx_stress_test.py` — ONNX stress test.

## Dataset Compliance / Provenance
- `extract_dataset_refs.py` — Extracts dataset IDs referenced by code into `datasets/inventory*` (offline by default).
- `verify_datasets.py` — Online dataset access sanity checks (opt-in HF login via `--login`).

## Ops Drills
- `nan_kill_test.py` — Synthetic NaN kill-switch drill.
- `failure_budget_drill.py` — Failure budget drill.
- `checkpoint_restore_drill.py` — Checkpoint restore drill.

## Artifacts & Reports
- `mini_titan_poc.py` — Forensic PoC logger (hash-chained logs).
- `scaling_audit_math.py` — Scaling audit math.
- `update_system_hardware.py` — Updates `reports/system_hardware*.md` (run.sh skips this in `--test` mode).
- `write_cuda_lock.py` — Writes `repro/cuda.lock` from the current system.

## Utilities
- `chat.py` — Interactive chat interface.
- `xray.py` — Project auditor (structure dump).
- `mac_simulation.py` — Mac simulation run (CPU/MPS).
- `train_tpu_turbo.py` — TPU training launcher (experimental).
- `download_tr_tokenizer.py` — Turkish tokenizer download (opt-in).
- `logbook_build.py` — Unified logbook builder (writes under `logs/` as a gitignored artifact).
- `version_checker.py` — Version consistency checker (fails on deprecated markers).

## Assets
- `build_investor_deck.py` — Generates the PPTX investor deck.
- `auto_demo_video.py` — Optional demo video automation (ffmpeg required).

## Folders
- `scripts/reports/` — Script-generated report artifacts.
- `scripts/runs/` — Per-run outputs.

---

Tip: `run.sh` covers the primary automated path (env + preflight + training). For review, prefer `scripts/verify_all.sh`.
