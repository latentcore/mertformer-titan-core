# Implementation Plan: High-Performance Edge-Native Intelligence

## Strategic Pillar
Boring stability and learning speed. Success is operational discipline measured through system health, not just loss.

## Phase -1: Safety and Failure Budget
- Kill Switch: Enforce non-finite detection and termination.
- Failure Budget: Pivot when learning signal or expert balance is absent for 72 hours.
- Checkpoint Drill: Save/restore integrity test before any master run.

## Phase 0: Reproducibility and Sanity Gates
- Reproducibility stamp logs git hash, config, seed, and dataset manifest.
- Overfit gate verifies memorization on 1MB of high-quality code.
- Observability layer collects grad norm, router entropy, and VRAM snapshots.
- Golden samples provide a stable logic-check set of 50 coding prompts.

## Phase 1: Telemetry-Driven Execution
- Expected vs Actual tracking for tokens/sec, loss slope, and GPU utilization.
- Master training run is driven by telemetry thresholds and failure budget.
- Internal truth benchmarking uses HumanEval and MBPP outputs for scoring.

## Phase 2: Asset Production
- One-pager and technical snapshot for rapid investor review.
- Founders Hub application: submitted 2026-05-31 (draft archived under `private/commercial/`).

## Phase 3: Future Horizons
- White paper and defense licensing after validation.

## Run Order (Operator Mode)
1. `scripts/operator_mode_gate.py` in safe mode on local machine.
2. `scripts/operator_mode_gate.py --full` on training hardware.
3. Master run (2.64B design target) with telemetry and failure budget active.
4. Internal benchmarks and asset updates after major checkpoints.

## Acceptance Criteria
- Kill switch verified by synthetic NaN injection.
- Failure budget pivot triggers under no-learning conditions.
- Checkpoint restore drill passes on tiny config.
- Overfit gate reaches target loss or 80% loss improvement.
- Golden sample set contains exactly 50 prompts.
- Asset stack present and internally consistent.

## Emergency Finalization Protocol (v1.0 (Build 30) Closeout)
- Do not change training/teacher path unless explicitly required.
- Keep all kernel paths opt-in and experimental by default.
- Ensure README/README_TR are aligned (Docs Index + Project Structure).
- Run tests and clean caches before final delivery.

## QAT Plan (When/How)
- When: After a stable baseline checkpoint exists and metrics are logged.
- How:
  - Phase A (Pilot, 1-2 days): enable QAT on a 1-5% subset with fixed seed.
  - Phase B (Compare, 1 day): compare loss/throughput vs baseline checkpoint.
  - Phase C (Scale, 2-3 days): expand to full curriculum if Phase B is neutral or positive.
- Goal: Improve low-bit inference quality without destabilizing training.
- Exit criteria: no regression on validation loss; no instability spikes.

## Turkish Tokenizer POC (Risk-Controlled)
- Default remains teacher tokenizer.
- Opt-in flag loads Turkish tokenizer cache from `tokenizer/tr`.
- Pilot steps:
  - Phase A (30-60 minutes): tokenize 500-1,000 samples, compare avg token length.
  - Phase B (1-2 hours): run 200-step mini-train on CPU/MPS, compare loss trend.
  - Phase C (same day): if loss degrades, auto-revert to teacher tokenizer.
- Exit criteria: no >5% token-length inflation and stable loss curve.

## Kernel Experimental + Tensor Core Opt-in
- Experimental low-bit kernels remain opt-in (CUDA + Triton required).
- Tensor-core path is opt-in (`MERTFORMER_TENSORCORE=1`) and correctness-first.
- Performance claims require real device profiling.
