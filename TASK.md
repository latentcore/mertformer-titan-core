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
- [x] Demo Video Script (offline MacBook Air): `reports/demo_video_script.md`
- [x] Investor Deck / One-Pager / Technical Snapshot: `reports/one_pager.md`, `reports/technical_snapshot.md`, `PITCH.md`, `TECHNICAL_REPORT.md`
- [x] Microsoft Founders Hub Application Draft: `reports/founders_hub_application.md`

## Phase 3: Future Horizons (v29 Backlog)
- [x] White Paper and Defense Licensing (post-validation): `WHITE_PAPER_LIQUIDROUTER.md`

## Verification Plan
- [x] Sanity Drill: Kill/Resume integrity and Failure Budget alerts
- [x] Single-entry gate runner: `scripts/operator_mode_gate.py`

## Delivery Notes
- Single-entry test run: `python scripts/operator_mode_gate.py`
- Safe mode is default; use `--full` for full gates on training hardware.
