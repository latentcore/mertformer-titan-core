# Task: Specialized Edge Coding Launch (Operator Mode - Locked and Sealed)

## Phase -1: Safety and Failure Budget
- [x] Auto-Kill NaN Injection (kill switch drill): `scripts/nan_kill_test.py`
- [x] Failure Budget Define and Pivot Trigger: `orchestrator/failure_budget.py`
- [x] Checkpoint Restore Drill (kill/resume integrity): `scripts/checkpoint_restore_drill.py`

## Phase 0: Infrastructure and Reality Contact Gates
- [x] Reproducibility Stamp (git, config, seed, datasets): `scripts/operator_mode_gate.py` + `utils/logger.py`
- [x] Overfit Gate on 1MB code (gate runner + fast safe mode): `scripts/overfit_gate.py`
- [x] Observability Layer (grad norms, router entropy hook, VRAM snapshot): `orchestrator/telemetry.py`
- [x] Golden Sample Eval (50 prompts): `datasets/golden_samples.jsonl` + `scripts/golden_eval.py`

## Phase 1: Telemetry-Driven Execution
- [x] Expected vs Actual Tracking scaffolding: `orchestrator/telemetry.py`
- [x] Master Training run path prepared (execution on training hardware required)
- [x] Internal Truth Benchmarking runner (HumanEval/MBPP): `scripts/benchmarks_internal.py`

## Phase 2: Minimal Prototype Stack (Assets)
- [x] Investor Deck / One-Pager / Technical Snapshot: `TECHNICAL_REPORT.md` (tracked). Commercial items (`one_pager.md`, `technical_snapshot.md`, `PITCH.md`) moved to `private/commercial/` — untracked.
- [x] Microsoft Founders Hub Application Draft: `private/commercial/founders_hub_application.md` (moved from `reports/`; untracked). Submitted 2026-05-31.

## Phase 3: Future Horizons (Backlog)
- [x] White Paper and Defense Licensing (post-validation): `WHITE_PAPER_LIQUIDROUTER.md`

## Verification Plan
- [x] Sanity Drill: Kill/Resume integrity and Failure Budget alerts
- [x] Single-entry gate runner: `scripts/operator_mode_gate.py`

## Delivery Notes
- Single-entry test run: `python scripts/operator_mode_gate.py`
- Safe mode is default; use `--full` for full gates on training hardware.
