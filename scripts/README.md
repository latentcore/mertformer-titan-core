# Scripts Catalog

All scripts are designed to run from the repo root.

Conventions:
- Prefer `.titan-venv/bin/python scripts/<name>.py` (see `scripts/bootstrap_venv.sh`).
- Script-level verification remains offline-first (`TITAN_OFFLINE=1`).
- `run.sh` training contract is online-by-default and supports readiness-only mode (`bash run.sh --train-ready`).
- Profile contract: `TITAN_PROFILE=stable` (default) or `TITAN_PROFILE=max_arch` for full advanced overlay.

If you are unsure, run the single-command verification first: `bash scripts/verify_all.sh`.

## Canonical One-File Path
- Official/canonical one-file script: `scripts/kaggle_onefile_demo_build30.py`
- Colab fastproof companion one-file script: `scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py`
- Text-understanding PoC one-file script: `scripts/kaggle_onefile_demo_build30_text_understanding.py`
- Windows RTX 5080 chess PoC one-file script: `scripts/chess_5080_onefile.py`
- Chess onefile supported modes: `train`, `verify`, `benchmark`, `package`, `resume`, `arena`
- `--mode arena` provides an interactive human-vs-model terminal surface; use `--resume-from <checkpoint>` for meaningful play.
- The chess onefile now mirrors the canonical Build30 trunk families in one file: BitLinear, MLA, CfC Liquid, MoE/LiquidRouter, QINN, cognitive extensions, and world-model hooks.
- Mirror anti-drift evidence is written to `reports/mirror_parity_report.json` during chess runs.
- Chess runtime observability is contract-backed: `logs/run_log.jsonl`, `reports/logging_contract.json`, and `reports/observability_report.json`.
- Fatal runtime failures are expected to appear both in `logs/run_log.jsonl` (`fatal_exception`) and in the Desktop-side `*_FAILED_*.json` artifact.
- The Windows builder/export flow no longer embeds `MERTFORMER_CHESS_ARCHIVE_PASSWORD` into the compiled launcher; provide it on the target machine before running the final EXE when encrypted output is required.
- Windows RTX 5080 share/export builder: `scripts/export_chess_5080_share.py`
- Repo-external copies are unsupported and treated as drift sources.

## Core Pipelines
- `smart_runner.py` — Master orchestrator: data → distill → train.
- `data_pipeline.py` — Dataset preparation (5-stage curriculum).
- `titan_preflight.py` — End-to-end preflight verification.
- `operator_mode_gate.py` — Single-entry ops gate (safety + sanity checks).
- `overfit_gate.py` — 1MB overfit gate (safe/full modes).
- `train_smoke.py` — Tiny offline training sanity loop (CPU/MPS).
- `cfc_moe_tolerance_check.py` — CfC/MoE loss tolerance check (<=1% diff).

## Review-Ready Tooling
- `bootstrap_venv.sh` — Creates `.titan-venv` (Python 3.11 baseline). Use `--demo` to install `pygame`.
- `verify_all.sh` — Offline-first verify-all: secret scan → pytest → preflight → operator gate (safe).
- `secret_scan.py` — Scans tracked files for common secret patterns (CI gate).
- `check_tokenizer_sync.py` — Enforces canonical tokenizer spec sync (`interfaces/tokenizer_spec.json` -> `tokenizer/tokenizer.json`).
- `check_translation_pointer_policy.py` — Enforces pointer policy for translated deep-audit counterparts.
- `check_doc_claim_consistency.py` — Checks claim/evidence consistency in key docs.
- `build_code_truth_audit.py` — Emits the code-truth delta audit with maturity labels, four-column evidence requirements, and marker scan output.
- `build_workspace_hygiene_manifest.py` — Builds the quarantine-first workspace hygiene manifest; `--apply-quarantine` is opt-in and should only be used after reviewing the generated manifest.
- `clean_runtime_artifacts.sh` — Cleans runtime artifacts (including root `kaggle_onefile_build30_*.jsonl`).
- `run_and_clean_pycache.py` — Runs any command and then guarantees post-run cache cleanup (`--full-clean` includes `.DS_Store`, `.cache`, `.ipynb_checkpoints`, `.tox`, `.nox`, `.hypothesis`, `.vs`; add `--include-venv-caches` to also clean venv caches).
- `zip_denylist_audit.py` — Audits release zip against denylisted paths and secret patterns.
- `build_scoped_external_intake_matrix.py` — Hashes and classifies scoped Desktop/Documents/Downloads/Applications project artifacts into a closure intake matrix.
- `cleanup_scoped_closure_junk.py` — Removes scoped closure junk (`__pycache__`, `.pyc`, duplicate stale zips) from repo + scoped external directories.

## SOP Outputs
- `reports/one_command_full_sop_summary.md` — Consolidated single-document summary for the full one-command SOP run.
- `reports/one_command_full_sop.log` — Raw full log for the same run.
- Both artifacts are refreshed/overwritten on each full SOP run.

## Evaluation & Benchmarks
- `plot_training_log.py` — Training JSONL → dashboard plotter (`reports/training_dashboard.png`).
- `golden_eval.py` — Golden sample evaluator (50 prompts).
- `golden_score.py` — Assertion-based golden scorer (`reports/benchmarks/golden_summary.json`).
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
- `mathfp_interactive_chat.py` — Math-fastproof interactive Q/A loop (type `q` to quit).
- `xray.py` — Project auditor (structure dump).
- `mac_simulation.py` — Mac simulation run (CPU/MPS).
- `train_tpu_turbo.py` — TPU training launcher (experimental).
- `download_tr_tokenizer.py` — Turkish tokenizer download (opt-in).
- `logbook_build.py` — Unified logbook builder (writes under `logs/` as a gitignored artifact).
- `version_checker.py` — Version consistency checker (fails on deprecated markers).
- `resume_compat_check.py` — Verifies resume compatibility and writes `reports/resume_compat_report.json`.
- `tools/claim_number_audit.py` — Audits `*.md` parameter-size formats (writes `reports/claim_number_audit.json`).
- `tools/denylist_scan_zip.py` — Pre-checks release zip against denylist (writes `reports/artifacts_zip_denylist_audit.json`).

## Assets
- `build_investor_deck.py` — Generates the PPTX investor deck.
- `update_investor_deck.py` — Updates investor deck PPTX to Build 30 V2 (auto V2 slide + text replacements).

## Folders
- `scripts/reports/` — Script-generated report artifacts.
- `scripts/runs/` — Per-run outputs.

---

Tip: `run.sh` covers the primary automated path (install + strict preflight + training). For review, prefer `scripts/verify_all.sh`.

## Build30 Colab Math Fastproof V2 (V1 Closure)

`scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py` now includes guarded full-spectrum hooks for closure-grade PoC packaging.

- Strict config schema v2 (`validate_run_config_schema`) with unknown-key reject/fail-fast.
- Runtime fingerprint and ownership bundle (`runtime_fingerprint`, `ownership_proof`, `env_snapshot_redacted`, `reproduce_command`).
- Compile/CUDAGraph stall guards (`compile_policy=off` default, timeout fallback, guard snapshot telemetry).
- Zero-shot unseen math split (`eval_unseen_*`) and compare payload v2 (`exact_match_unseen`).
- Interpretability outputs (`gradient_flow_heatmap.png`, `moe_expert_bar_proxy.png`) behind flags.
- Feature coverage contract (`feature_coverage_matrix`) with completeness percent in payload.
