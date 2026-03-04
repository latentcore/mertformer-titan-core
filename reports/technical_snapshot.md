# Technical Snapshot

## Architecture
- Model size: 2.64B parameters (design target); latest measured runtime total: ~3.70B.
- Quantization: BitNet 1.58-bit for weights.
- Routing: LiquidRouter MoE with temporal dynamics.
- Attention: Multi-Head Latent Attention with long-context readiness.

## Training Pipeline
- Distillation pipeline with offline logits support.
- Curriculum stages and safety checks.
- Checkpointing with smart retention.

## Safety and Gates
- NaN kill switch drill: `scripts/nan_kill_test.py`.
- Failure budget monitor: `orchestrator/failure_budget.py`.
- Checkpoint restore drill: `scripts/checkpoint_restore_drill.py`.
- Overfit gate on 1MB dataset: `scripts/overfit_gate.py`.

## Observability
- Grad norm logging in training loop.
- Router entropy utility: `orchestrator/telemetry.py`.
- VRAM and system snapshots: `orchestrator/telemetry.py`.

## Reproducibility
- Run manifest with git hash, config, seed, and dataset hashes: `utils/logger.py` and `scripts/operator_mode_gate.py`.

## Evaluation
- Golden samples (50 prompts) for logic checks: `datasets/golden_samples.jsonl`.
- Internal benchmarks: HumanEval and MBPP output generation: `scripts/benchmarks_internal.py`.

## Deployment
- ONNX export and mobile pipeline scripts in `scripts/`.
- Offline demo flow documented in `reports/demo_video_script.md`.
