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
- Demo video script for an offline MacBook Air showcase.
- One-pager and technical snapshot for rapid investor review.
- Founders Hub application draft ready for submission.

## Phase 3: Future Horizons
- White paper and defense licensing after validation.

## Run Order (Operator Mode)
1. `scripts/operator_mode_gate.py` in safe mode on local machine.
2. `scripts/operator_mode_gate.py --full` on training hardware.
3. Master run (2.6B) with telemetry and failure budget active.
4. Internal benchmarks and asset updates after major checkpoints.

## Acceptance Criteria
- Kill switch verified by synthetic NaN injection.
- Failure budget pivot triggers under no-learning conditions.
- Checkpoint restore drill passes on tiny config.
- Overfit gate reaches target loss or 80% loss improvement.
- Golden sample set contains exactly 50 prompts.
- Asset stack present and internally consistent.

## Emergency Finalization Protocol (v27 Closeout)
- Do not change training/teacher path unless explicitly required.
- Keep all kernel paths opt-in and experimental by default.
- Ensure README/README_TR are aligned (Docs Index + Project Structure).
- Run tests and clean caches before final delivery.

## QAT Plan (When/How)
- When: After a stable baseline checkpoint exists and metrics are logged.
- How: Enable QAT on a smaller subset first, compare with baseline, then scale.
- Goal: Improve low-bit inference quality without destabilizing training.

## Turkish Tokenizer POC (Risk-Controlled)
- Default remains teacher tokenizer.
- Opt-in flag loads Turkish tokenizer cache from `tokenizer/tr`.
- Run a small validation set to compare tokenization length and quality.
- If distillation quality drops, revert to teacher tokenizer.

## Kernel Experimental + Tensor Core Opt-in
- Experimental low-bit kernels remain opt-in (CUDA + Triton required).
- Tensor-core path is opt-in (`MERTFORMER_TENSORCORE=1`) and correctness-first.
- Performance claims require real device profiling.
