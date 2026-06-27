# Model Card — MertFormer Titan (Build 30 V2)

## Overview
MertFormer Titan is a pre-training-stage, offline-first, edge-native language model stack built around BitNet-style low-bit layers, Liquid routing dynamics, and sparse MoE execution.

## Official Positioning
Turkey-serving, offline-first, edge-native, locally integrable intelligence infrastructure.

## Truth Labels
- `measured`: backed by a concrete artifact, benchmark, manifest, or log
- `target`: intended or planned behavior not yet validated
- `vision`: long-horizon direction outside current claim scope
- `verified`, `hypothesis`, and `creative_or_folklore` are separate output modes

## Current State
- Repository state: pre-training / claim-unverified
- Parameter count: measured (~3.67B) in repo artifacts; runtime/throughput: unverified (no trained checkpoint or runtime measurement yet)
- Benchmark eligibility: `NOT ELIGIBLE FOR CLAIM` without a trained checkpoint
- 45K run: first serious architecture validation run, not the final capability ceiling
- Readiness posture: repo-side start gate is green on `remote_bootstrap` (`TRAIN_ALLOWED` / `READY_REMOTE_BOOTSTRAP`)
- Strict local boundary: `offline_clean` remains the strict precomputed-KD lane and stays blocked without local logits or actionable local Phase-0
- Remaining non-winning blockers: `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`, `online_teacher:MISSING_HF_TOKEN`

## Intended Use
- Offline-first and edge-native experimentation
- Auditable local deployment research
- Human-supervised decision-support systems

## Out of Scope
- Production or certified safety claims without training evidence
- High-risk operation without human review
- Covert surveillance or harmful autonomy framing

## Training and Data
Training data, licenses, hashes, and stage composition are governed by `datasets/` source-of-truth files and the closure-generated provenance reports.

## Responsible Use
Use is governed by `USE_POLICY.md` and `SECURITY.md`.
