# Scripts Catalog

All scripts are designed to run from the repo root.

Conventions:
- Prefer `.titan-venv/bin/python scripts/<name>.py` (see `scripts/bootstrap_venv.sh`).
- Script-level verification remains offline-first (`TITAN_OFFLINE=1`).
- `run.sh` training contract is online-by-default and supports readiness-only mode (`bash run.sh --train-ready`).
- Profile contract: `TITAN_PROFILE=stable` (default) or `TITAN_PROFILE=max_arch` for full advanced overlay.

If you are unsure, run the single-command verification first: `bash scripts/verify_all.sh`.

## Canonical One-File Path
- Official/canonical Kaggle closure script (terminal-first): `scripts/kaggle_onefile_closure_build30.py`
- Official/canonical Kaggle one-cell script (single-T4 copy/paste lane): `scripts/kaggle_onecell_t4_build30.py`
- Legacy/reference one-file trainer: `scripts/kaggle_onefile_demo_build30.py`
- Legacy/reference fastproof companion one-file script: `scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py`
- Text-understanding PoC one-file script: `scripts/kaggle_onefile_demo_build30_text_understanding.py`
- Windows RTX 5080 chess PoC one-file script: `scripts/chess_5080_onefile.py`
- Canonical general 5080 final one-file script: `scripts/mertformer_5080_final_onefile.py`
- General 5080 delivery builder: `scripts/build_mertformer_5080_final_delivery.py`
- General 5080 decrypt helper: `scripts/decrypt_mertformer_result_package.py`
- Canonical Kaggle entry command: `bash zero_touch_start.sh --kaggle-onefile --mode train-end`
- Canonical Kaggle verify command: `bash zero_touch_start.sh --kaggle-onefile --mode verify`
- Canonical Kaggle package command: `bash zero_touch_start.sh --kaggle-onefile --mode package-only`
- One-click macOS launcher: `launch_mertformer_kaggle_closure.command`
- Canonical Kaggle profiles: `auto`, `onecell_t4_sweetspot`, `p100_safe`, `t4x2_dist`, `sweetspot`, `mini300m_probe`, `fastproof_math`
- Single-T4 support matrix: `onecell_t4_sweetspot` is the preferred auto-selected lane on a single Tesla T4; `t4x2_dist` remains the preferred dual-T4 lane.
- The one-cell lane is repo-import-free at runtime and is designed for direct Kaggle cell paste plus a single `Run` click.
- One-cell observability contract: `config_snapshot.json`, `runtime_preflight_report.json`, `layer_parity_manifest.json`, `event_manifest.json`, `final_summary.json`, `sha256_manifest.txt`, and `fatal_report.json` on hard failures.
- Runtime accelerator reality: Kaggle GPU type and quota are floating/account-dependent; the canonical script detects runtime hardware instead of hard-coding a fixed entitlement claim.
- Chess onefile supported modes: `train`, `verify`, `benchmark`, `package`, `resume`, `arena`
- Chess onefile now supports named feature bundles through `--feature-bundle` plus explicit `--enable-features` / `--disable-features` overrides.
- Recommended advanced bundle names: `routing_stack`, `liquid_stack`, `memory_attention_stack`, `cognitive_stack`, `objective_stack`, `postrun_analysis_stack`, `all_stable_extensions`, `all_on_experimental`.
- Canonical 24h RTX 4060 profile: `strength_4060_24h` (`baseline_supported`, release-candidate eligible).
- Supported portable baseline profile: `production_5080` (`supported_portable_baseline`).
- Research-only 24h RTX 4060 profiles: `strength_4060_24h_all_on_experimental` (`experimental`) and `strength_4060_24h_omni_max` (`experimental_high_risk`).
- `--mode arena` provides an interactive human-vs-model terminal surface; use `--resume-from <checkpoint>` for meaningful play.
- The chess onefile now mirrors the canonical Build30 trunk families in one file: BitLinear, MLA, CfC Liquid, MoE/LiquidRouter, QINN, cognitive extensions, and world-model hooks.
- Auxiliary chess heads can now be toggled inside the onefile: `phase_head`, `wdl_head`, and `legality_head`.
- Post-run chess analysis surfaces can now be toggled: `selfplay_eval_enabled`, `tournament_eval_enabled`, and `replay_buffer_enabled`.
- Mirror anti-drift evidence is written to `reports/mirror_parity_report.json` during chess runs.
- Feature-bundle evidence is written to `reports/feature_flag_report.json` and `reports/feature_flag_report.md` during chess runs.
- Closure manifests are written to `reports/run_status_manifest.json`, `reports/postrun_analysis_manifest.json`, and `reports/artifact_truth_matrix.json`.
- Release/evidence registry surfaces are written to `reports/run_contract.json`, `reports/release_snapshot.json`, `reports/evidence_pack_stub.json`, and `reports/final_truth_registry.json`.
- Additional release-truth artifacts are written to `reports/claim_registry.json`, `reports/known_limits.json`, `reports/support_matrix.json`, and `reports/release_gate_summary.json`.
- Handoff/release stub artifacts are written to `reports/rc_stub.json`, `reports/golden_stub.json`, `reports/handoff_pack_manifest.json`, and `reports/operator_handoff_summary.json`.
- External closure stubs are written to `reports/external_repro_stub.json`, `reports/pilot_stub.json`, `reports/security_stub.json`, and `reports/legal_stub.json`.
- Operator/DR closure stubs are written to `reports/operator_handbook_stub.json`, `reports/dr_evidence_stub.json`, `reports/backup_retention_stub.json`, and `reports/blind_handoff_stub.json`.
- Release-governance artifacts are written to `reports/release_notes_stub.json`, `reports/freeze_manifest_stub.json`, `reports/changelog_snapshot.json`, and `reports/maintenance_policy_stub.json`.
- Device/export/packaging closure artifacts are written to `reports/export_truth_stub.json`, `reports/device_validation_stub.json`, `reports/packaging_closure_stub.json`, and `reports/installer_validation_stub.json`.
- Benchmark-closure artifacts are written to `reports/benchmark_raw_outputs_stub.json`, `reports/benchmark_compare_report_stub.json`, `reports/benchmark_summary_stub.json`, and `reports/benchmark_manifest_stub.json`.
- Training/accounting closure artifacts are written to `reports/training_report_stub.json`, `reports/token_accounting_stub.json`, `reports/compute_accounting_stub.json`, and `reports/cost_report_stub.json`.
- Trained-artifact-truth artifacts are written to `reports/final_weights_truth_stub.json`, `reports/best_checkpoint_truth_stub.json`, `reports/latest_checkpoint_truth_stub.json`, and `reports/trained_artifact_registry_stub.json`.
- Management-closure artifacts are written to `reports/core_complete_decision_stub.json`, `reports/research_continues_stub.json`, `reports/product_maintenance_only_stub.json`, and `reports/closure_decision_record_stub.json`.
- Repo-truth summary artifacts are written to `reports/master_closure_table.json`, `reports/remaining_core_blockers.json`, `reports/repo_side_completion_summary.json`, and `reports/readiness_snapshot.json`.
- Aggregated truth artifacts are written to `reports/aggregated_master_table.json`, `reports/real_remaining_core_work.json`, `reports/repo_truth_inventory.json`, and `reports/closure_gap_summary.json`.
- Project-truth and docs-alignment artifacts are written to `reports/project_master_truth_reference.json`, `reports/project_remaining_real_blockers.json`, `reports/truth_docs_index.json`, and `reports/truth_docs_drift_report.json`.
- Consistency/action artifacts are written to `reports/project_blocker_action_plan.json`, `reports/project_blocker_dependency_graph.json`, `reports/project_execution_sequence.json`, `reports/project_lane_status_board.json`, `reports/project_closure_phase_plan.json`, `reports/project_phase_readiness_scoreboard.json`, `reports/project_owner_accountability_matrix.json`, `reports/project_owner_work_queue.json`, `reports/project_critical_path_report.json`, `reports/project_owner_next_actions_summary.json`, `reports/project_ready_now_board.json`, `reports/project_unlock_impact_report.json`, `reports/project_parallel_workset_report.json`, `reports/project_phase_exit_criteria_report.json`, `reports/project_execution_wave_report.json`, `reports/project_evidence_backlog_report.json`, `reports/project_dependency_bottleneck_report.json`, `reports/project_owner_phase_frontier_report.json`, `reports/project_evidence_criticality_report.json`, `reports/project_phase_transition_matrix.json`, `reports/project_owner_load_report.json`, `reports/project_phase_dependency_pressure_report.json`, `reports/project_owner_bottleneck_alignment_report.json`, `reports/project_evidence_phase_heatmap_report.json`, `reports/project_blocker_risk_register_report.json`, `reports/project_release_prereq_matrix_report.json`, `reports/project_foundation_run_dependency_report.json`, `reports/project_release_path_report.json`, `reports/project_external_closure_cluster_report.json`, `reports/project_owner_evidence_gap_report.json`, `reports/project_release_gate_dependency_report.json`, `reports/project_external_signoff_queue_report.json`, `reports/project_release_evidence_bridge_report.json`, `reports/project_training_run_readiness_report.json`, `reports/project_benchmark_closure_dependency_report.json`, `reports/project_release_decision_queue_report.json`, `reports/project_external_validation_readiness_report.json`, `reports/project_artifact_lock_readiness_report.json`, `reports/project_final_release_cutover_report.json`, `reports/project_real_run_execution_queue_report.json`, `reports/project_benchmark_evidence_lock_report.json`, `reports/project_final_signoff_cutset_report.json`, `reports/generated_truth_consistency_report.json`, and `reports/generated_truth_crosscheck_matrix.json`.
- Canonical runbook/checklist support for the frozen 24h 4060 path now lives under `runbooks/chess_4060_24h.md` and `checklists/chess_4060_24h.md`.
- Experimental 24h 4060 runbooks remain under `runbooks/chess_4060_24h_all_on_experimental.md` and `checklists/chess_4060_24h_all_on_experimental.md`.
- Canonical repo-side contract surfaces now also include `configs/`, `releases/`, `knowledge/`, and `evidence/`.
- Chess runtime observability is contract-backed: `logs/run_log.jsonl`, `reports/logging_contract.json`, and `reports/observability_report.json`.
- Fatal runtime failures are expected to appear both in `logs/run_log.jsonl` (`fatal_exception`) and in the Desktop-side `*_FAILED_*.json` artifact.
- The Windows builder/export flow no longer embeds `MERTFORMER_CHESS_ARCHIVE_PASSWORD` into the compiled launcher; provide it on the target machine before running the final EXE when encrypted output is required.
- Windows RTX 5080 share/export builder: `scripts/export_chess_5080_share.py`
- General 5080 final truth boundary: `docs/MERTFORMER_5080_FINAL_ONEFILE_TRUTH.md`
- Repo-external copies are unsupported and treated as drift sources.

### Chess Onefile Bundle Examples
```bash
# Canonical 24h RTX 4060 train-start command
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h

# Stable-ish advanced stack on top of the portable baseline
python3 scripts/chess_5080_onefile.py --mode train --profile production_5080 --feature-bundle all_stable_extensions

# 24h RTX 4060 research-only profile with every major onefile extension enabled
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h_all_on_experimental

# More aggressive 24h RTX 4060 omni-max research variant with every feature flag forced on
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h_omni_max

# Start from the all-on profile but explicitly disable a risky surface
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h_all_on_experimental --disable-features use_qinn,use_world_model_head

# Keep the trunk profile but only opt into post-run self-play/tournament/replay artifacts
python3 scripts/chess_5080_onefile.py --mode train --profile production_5080 --feature-bundle postrun_analysis_stack
```

## Core Pipelines
- `smart_runner.py` — Master orchestrator: data → distill → train.
- `data_pipeline.py` — Dataset preparation (5-stage curriculum).
- `precompute_logits_topk.py` — Phase-0 offline teacher Top-K logit shard builder for the strict precomputed-KD lane.
- `precompute_logits_parallel.py` — Multi-GPU data-parallel orchestrator for the above (block-cyclic; same shards, faster wall-clock). Opt-in via `TITAN_PRECOMPUTE_GPUS`. See ADR-0005.
- `titan_preflight.py` — End-to-end preflight verification.
- `operator_mode_gate.py` — Single-entry ops gate (safety + sanity checks).
- `overfit_gate.py` — 1MB overfit gate (safe/full modes).
- `train_smoke.py` — Tiny offline training sanity loop (CPU/MPS).
- `cfc_moe_tolerance_check.py` — CfC/MoE loss tolerance check (<=1% diff).
- `liquid_train_impl_benchmark.py` — Optional Liquid train implementation microbenchmark; results are local evidence only until tied to a target-machine run artifact.
- `build_training_outputs_bundle.py` — Builds the canonical downloadable training outputs bundle zip + SHA256 + manifests.

## Review-Ready Tooling
- `bootstrap_venv.sh` — Creates `.titan-venv` (Python 3.11 baseline). Use `--demo` to install `pygame`.
- `verify_all.sh` — Offline-first verify-all: secret scan → pytest → preflight → operator gate (safe).
- `tests/test_packed_projection_equivalence.py` — Optional speed-flag equivalence coverage for `TITAN_FFN_PACK`, `TITAN_MOE_PACK`, and `TITAN_MLA_KV_PACK`.
- `tests/test_liquid_safeguard.py` — Liquid train implementation safeguard coverage for `TITAN_LIQUID_TRAIN_IMPL`.
- `secret_scan.py` — Scans tracked files for common secret patterns (CI gate).
- `check_tokenizer_sync.py` — Enforces canonical tokenizer spec sync (`interfaces/tokenizer_spec.json` -> `tokenizer/tokenizer.json`).
- `check_translation_pointer_policy.py` — Enforces pointer policy for translated deep-audit counterparts.
- `check_doc_claim_consistency.py` — Checks claim/evidence consistency in key docs.
- `build_code_truth_audit.py` — Emits the code-truth delta audit with maturity labels, four-column evidence requirements, and marker scan output.
- `build_workspace_hygiene_manifest.py` — Builds the quarantine-first workspace hygiene manifest; `--apply-quarantine` is opt-in and should only be used after reviewing the generated manifest.
- `clean_runtime_artifacts.sh` — Cleans runtime artifacts (including root `kaggle_onefile_build30_*.jsonl`).
- `macos_keepawake.sh` — Process-scoped macOS `caffeinate` wrapper for long verify/train/upload commands without changing global power settings.
- `run_and_clean_pycache.py` — Runs any command and then guarantees post-run cache cleanup (`--full-clean` includes `.DS_Store`, `.cache`, `.ipynb_checkpoints`, `.tox`, `.nox`, `.hypothesis`, `.vs`; add `--include-venv-caches` to also clean venv caches).
- `zip_denylist_audit.py` — Audits release zip against denylisted paths and secret patterns.
- `build_scoped_external_intake_matrix.py` — Hashes and classifies scoped Desktop/Documents/Downloads/Applications project artifacts into a closure intake matrix.
- `cleanup_scoped_closure_junk.py` — Removes scoped closure junk (`__pycache__`, `.pyc`, duplicate stale zips) from repo + scoped external directories.

## Optional 45K Speed-Control Surface

These controls are documented because they are wired into code, but they remain claim-safe operator knobs:
- `TITAN_BATCH_SIZE`, `TITAN_LOG_INTERVAL`, `TITAN_VAL_CHECK_INTERVAL`, `TITAN_SAVE_INTERVAL`
- `TITAN_DATALOADER_PIN`, `TITAN_DATALOADER_NONBLOCKING`
- `TITAN_FFN_PACK`, `TITAN_MOE_PACK`, `TITAN_MLA_KV_PACK`
- `TITAN_LIQUID_TRAIN_IMPL`
- `ACCELERATE_CONFIG_FILE=repro/accelerate_8xgpu.yaml`

Before enabling the optional packed/Liquid flags on a target run:

```bash
python3 -m pytest -q tests/test_packed_projection_equivalence.py tests/test_liquid_safeguard.py
```

`repro/accelerate_8xgpu.yaml` belongs under `repro/` because it records a reproducible run launch profile, not a stable model/config contract under `configs/`.

## SOP (Standard Operating Procedure) Outputs
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

## Canonical Kaggle Closure Lane

The repo now keeps a single terminal-first Kaggle closure lane:

```bash
bash zero_touch_start.sh --kaggle-onefile --mode train-end --profile auto
```

What it always tries to produce:
- first-100-step loss snapshot
- canonical `latest.pt` / `best.pt` / `manifest.json` checkpoint contract
- tiny compare report
- text-understanding mini eval
- canonical artifact index
- sha256 manifest
- canonical evidence bundle zip

Helpful local launch wrappers:

```bash
bash scripts/macos_keepawake.sh --assert-seconds 10800 -- \
  bash zero_touch_start.sh --kaggle-onefile --mode verify
```

For a rented 8x NVIDIA B300 box, use the dedicated wrapper (hardware asserts + canonical 45K env, delegates to `zero_touch_start.sh`; previews by default, trains only with `--go`):

```bash
HF_TOKEN=... bash scripts/launch_8xb300.sh --check-only   # or --dry-run, or --go
```

Two lanes (both preview by default, train only with `--go`; secrets come from the environment, never the file):
- `scripts/launch_8xb300.sh` — **offline_clean** lane (`TITAN_OFFLINE=1`, precomputed Top-K logits).
- `scripts/launch_ocean_45k.sh` — **remote_bootstrap** lane (`TITAN_OFFLINE=0`, online gated teacher, Phase-0 skipped) + operator observability (redacted env snapshot, cuda.lock, 8x/bf16 GPU assert, pretests, verify_all, live `nvidia-smi` telemetry, post-run bundle):

```bash
HF_TOKEN=... bash scripts/launch_ocean_45k.sh --check-only   # or --dry-run, or --go
```

Or, on macOS, double-click:

```bash
launch_mertformer_kaggle_closure.command
```

## Canonical Kaggle One-Cell Lane

For a direct Kaggle notebook cell paste on a single T4 runtime, use:

```python
# copy/paste the contents of scripts/kaggle_onecell_t4_build30.py into a Kaggle cell
# default profile: t4_onecell_sweetspot
# interaction: none, just press Run
```

## Kaggle Batch Runner (unattended multi-job lane)

`kaggle_batch_runner.py` — sequential, unattended runner for 5 side-experiments/re-verification
jobs (Nutrition5k N3/N4 ablations, 36M/171M LM re-verify, chess PoC) inside one Kaggle "Save & Run
All (Commit)" session. Ships as a standalone Kaggle Dataset (repo `git archive` snapshot +
`orchestrator/kaggle_batch_runner.py`), not invoked as part of any canonical 45K launch path. Real
run on 2026-07-25 (Kaggle T4×2) produced the N3/N4/36M/171M results recorded in `BACKLOG.md` and
`evidence/2026-07-25-*/`. Time-boxed per job (SIGTERM→grace→SIGKILL), exclusive-lock guarded
against duplicate concurrent invocation, and includes a real 2-GPU DDP smoke test (polls actual
GPU utilization while the subprocess is alive) before attempting DDP on the two LM jobs. See
`tests/test_kaggle_batch_runner.py` for the safety-critical behaviors verified in isolation.

## Pre-45K Gate (cheap, pre-spend launch-readiness check)

`pre45k_gate.sh` / `pre45k_gate.py` — chains three checks, all before any real training spend:
(1) the existing offline preflight (`titan_preflight.py`'s default profile); (2) the existing
dry-run preview (`zero_touch_start.sh --dry-run`); (3) a new short, real 2-GPU DDP smoke test
(`ddp_smoke.py::run_ddp_smoke_test()` — polls actual GPU utilization while the subprocess is
alive, an independent implementation of the same design as `kaggle_batch_runner.py`'s own DDP
smoke test, deliberately not sharing code with it). Writes `reports/pre45k_gate_report.json`/`.md`
with a combined verdict. Exists because today's only DDP-rank-sync assertion (`train/train.py`'s
"[Gate 3]") only fires *inside* the real training run, at the step-10000-class Liquid-unfreeze
event — after real budget is already spent. See `BACKLOG.md`, item B8, for the full design
rationale and the honest claim boundary (the DDP-confirmed path itself has never been exercised
on real 2-GPU hardware from this repo — only its skip-path and decision logic are verified here).

```bash
bash scripts/pre45k_gate.sh                # report-only; DDP result is informational
bash scripts/pre45k_gate.sh --strict-ddp   # also exit 1 if 2+ GPUs present but DDP unconfirmed
```

See `tests/test_ddp_smoke.py` and `tests/test_pre45k_gate.py`.

## Build30 Colab Math Fastproof V2 (V1 Closure)

`scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py` now includes guarded full-spectrum hooks for closure-grade PoC packaging.

- Strict config schema v2 (`validate_run_config_schema`) with unknown-key reject/fail-fast.
- Runtime fingerprint and ownership bundle (`runtime_fingerprint`, `ownership_proof`, `env_snapshot_redacted`, `reproduce_command`).
- Compile/CUDAGraph stall guards (`compile_policy=off` default, timeout fallback, guard snapshot telemetry).
- Zero-shot unseen math split (`eval_unseen_*`) and compare payload v2 (`exact_match_unseen`).
- Interpretability outputs (`gradient_flow_heatmap.png`, `moe_expert_bar_proxy.png`) behind flags.
- Feature coverage contract (`feature_coverage_matrix`) with completeness percent in payload.
