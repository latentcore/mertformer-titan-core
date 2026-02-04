# Report Accuracy Audit (v27.0)

This audit maps the external report claims to actual repo evidence. Each claim is labeled:
- TRUE
- PARTIAL / Evidence missing
- INCORRECT / Overstated

## A) Executive Summary / Status
- "Core engineering, not a hobby project." TRUE
  - Evidence: `scripts/titan_preflight.py`, `scripts/operator_mode_gate.py`, `orchestrator/failure_budget.py`
- "GPT-3.5 class on-device (S25/M4)." INCORRECT / Evidence missing
  - No measured device benchmarks.
- "Production-grade / locked & sealed." PARTIAL
  - Safety gates exist, but no real training run results.

## B) BitNet b1.58 / Energy
- "Ternary weights implemented." TRUE
  - Evidence: `layers/bitlinear.py`
- "0.65 GB weights / 93.75% memory savings." PARTIAL
  - Theoretical; no measured size output.
- "Multiply -> add, ~70x energy." PARTIAL
  - Low-bit kernel path exists, but energy/TOPS requires hardware measurement.

## C) NPU / Kernel Compatibility
- "Needs custom kernel for ternary." PARTIAL
  - Bitpack hooks exist, but vendor kernel is still required.
  - Evidence: `scripts/mobile_export.py` (bitpack metadata)

## D) LiquidRouter / MoE Stability
- "CfC-based router exists." TRUE
  - Evidence: `layers/liquid.py`
- "Auto jitter/entropy correction." PARTIAL
  - Monitoring exists, auto-correction is not implemented.

## E) Offline Distillation
- "Offline logits path exists." TRUE
  - Evidence: `orchestrator/distillation_manager.py`, `train/train.py`
- "75% cost reduction." INCORRECT / Evidence missing
  - No measured profiling.

## F) Roadmap / Assets
- "Founders Hub application ready." TRUE
  - Evidence: `reports/founders_hub_application.md`
- "Demo video script ready." TRUE
  - Evidence: `reports/demo_video_script.md`

## Summary
The architecture and safety tooling are real. Performance and device-level claims are not yet measured and should be treated as estimates until a full training run + device profiling are completed.
