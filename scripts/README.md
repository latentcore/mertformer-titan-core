# Scripts Catalog

All scripts are designed to run from the repo root using `python3 scripts/<name>.py`.
Some scripts require GPU, HF_TOKEN, or WANDB_API_KEY. When in doubt, run `run.sh` first.

## Core Pipelines
- `smart_runner.py` — Master orchestrator: data → distill → train.
- `data_pipeline.py` — Dataset preparation (5-stage curriculum).
- `titan_preflight.py` — End-to-end preflight verification.
- `operator_mode_gate.py` — Single-entry ops gate (safety + sanity checks).
- `overfit_gate.py` — 1MB overfit gate (safe/full modes).

## Evaluation & Benchmarks
- `golden_eval.py` — Golden sample evaluator (50 prompts).
- `benchmarks_internal.py` — HumanEval / MBPP output generator.
- `eval.py` — GSM8K evaluator stub (integration placeholder).

## Export & ONNX
- `mobile_export.py` — ONNX export for mobile/edge.
- `test_onnx_export.py` — ONNX export test.
- `verify_onnx_local.py` — Local ONNX verification.
- `titan_onnx_stress_test.py` — ONNX stress test.

## Ops Drills
- `nan_kill_test.py` — Synthetic NaN kill-switch drill.
- `failure_budget_drill.py` — Failure budget drill.
- `checkpoint_restore_drill.py` — Checkpoint restore drill.

## Artifacts & Reports
- `mini_titan_poc.py` — Forensic PoC logger (hash-chained logs).
- `scaling_audit_math.py` — Scaling audit math.
- `update_system_hardware.py` — Refreshes `reports/system_hardware*.md`.
- `write_cuda_lock.py` — Writes `repro/cuda.lock` from the current system.
- `verify_datasets.py` — Dataset sanity checks.

## Utilities
- `chat.py` — Interactive chat interface.
- `xray.py` — Project auditor (structure dump).
- `mac_simulation.py` — Mac simulation run (CPU/MPS).
- `train_tpu_turbo.py` — TPU training launcher (experimental).
- `download_tr_tokenizer.py` — Turkish tokenizer download (opt-in).

## Assets
- `build_investor_deck.py` — Generates the PPTX investor deck.
- `auto_demo_video.py` — Optional demo video automation (ffmpeg required).

## Folders
- `scripts/reports/` — Script-generated report artifacts.
- `scripts/runs/` — Per-run outputs.

---

Tip: `run.sh` covers the primary automated path (env + preflight + training).
