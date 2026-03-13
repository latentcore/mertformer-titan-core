# Model Card — MertFormer Titan (v1.0 (Build 30 V2))

## Overview
MertFormer Titan is a 2.64B-parameter, mobile-first language model built around
BitNet 1.58-bit layers, LiquidRouter MoE, and MLA-labeled GQA attention (current implementation). This card reflects the
**pre-training** stage and documents intended use, constraints, and known gaps.

## V2 Updates
- Cross-dataset deduplication enabled in data pipeline.
- MoE dispatch supports parallel gather/scatter mode.
- LiquidMixer fast path available behind `liquid_fast_path`.

## Intended Use
- On-device/edge inference research and prototyping
- Mobile/embedded deployment experiments
- Research on efficient routing and low-bit inference

## Out of Scope / Non-Goals
- Safety-critical deployment without independent validation
- Medical, legal, or defense decisions without human review
- Any use that violates dataset licenses or privacy regulations

## Training Data (Current Inventory)
Training datasets are referenced in `datasets/SOURCES.md`. Final snapshots,
hashes, and license verification are required **before** production training.

## Evaluation Status
- Benchmarks: **Not yet completed** (pre-training).
- Planned: HumanEval / MBPP / GSM8K after a stable baseline run.

## Limitations
- Performance metrics are currently **targets/estimates**.
- Low-bit kernel path is **experimental** and opt-in.
- No certified safety evaluation yet; use with caution.

## Responsible Use
Use is governed by `USE_POLICY.md`. Always follow local laws, privacy rules,
and internal governance requirements.

## Contact
For research inquiries or collaboration: see README Contact section.
