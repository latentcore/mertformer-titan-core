# Report Accuracy Audit (v1.0 (Build 30))

This audit maps external report claims to repo evidence. Each claim is labeled:
- TRUE (implemented and evidenced)
- TARGET / ESTIMATE (explicitly framed as target; pending measurement)
- PENDING VALIDATION (requires training/device measurements)
- REMOVED (claim removed or downgraded in docs)

## A) Executive Summary / Status
- "Core engineering, not a hobby project." TRUE
  - Evidence: `scripts/titan_preflight.py`, `scripts/operator_mode_gate.py`, `orchestrator/failure_budget.py`
- "GPT-3.5 class on-device (S25/M4)." TARGET / ESTIMATE
  - Framed as a post-training target; no device benchmarks yet.
- "Production-grade / locked & sealed." REMOVED
  - Docs now state pre-training / unverified status.

## B) BitNet b1.58 / Energy
- "Ternary weights implemented." TRUE
  - Evidence: `layers/bitlinear.py`
- "0.65 GB weights / 93.75% memory savings." TARGET / ESTIMATE
  - Theoretical; needs measured size output.
- "Multiply -> add, ~70x energy." PENDING VALIDATION
  - Low-bit kernel path exists, but energy/TOPS needs hardware measurement.

## C) NPU / Kernel Compatibility
- "Needs custom kernel for ternary." PENDING VALIDATION
  - Bitpack hooks exist, vendor kernel still required.
  - Evidence: `scripts/mobile_export.py` (bitpack metadata)

## D) LiquidRouter / MoE Stability
- "CfC-based router exists." TRUE
  - Evidence: `layers/liquid.py`
- "Auto jitter/entropy correction." PENDING VALIDATION
  - Monitoring exists; auto-correction is not yet validated.

## E) Offline Distillation
- "Offline logits path exists." TRUE
  - Evidence: `orchestrator/distillation_manager.py`, `train/train.py`
- "75% cost reduction." REMOVED
  - Not claimed without measured profiling.

## F) Roadmap / Assets
- "Founders Hub application ready." TRUE
  - Evidence: `reports/founders_hub_application.md`
- "Demo video script ready." TRUE
  - Evidence: `reports/demo_video_script.md`

## Summary
Architecture and safety tooling are real. Performance and device-level claims are framed as targets and require full training + device profiling to validate.
