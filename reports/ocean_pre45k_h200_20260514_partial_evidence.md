# Ocean Pre-45K H200 Partial Evidence - 2026-05-14

## Status
- Claim mode: `measured`
- Status: `partial_operational_evidence_only`
- Public Gist pointer: `https://gist.github.com/latentcore/dac0aa0c56b12177e4a0e8e8f684bccf`
- Machine-readable summary: `reports/ocean_pre45k_h200_20260514_clean_summary.json`

## Run Identity
- Run name: `final_math_h200`
- Proof version: `pre45k_h200_final_evidence_proof_v2_3_fast_proof`
- Runtime family: Ocean / OnCompute-style job wrapper
- Docker image: `ghcr.io/latentcore/mertformer-oncompute-enc`
- Docker tag: `build30-7c393c7`
- Training profile: reduced pre-45K synthetic arithmetic proof

## Architecture Under Test
This was a reduced synthetic arithmetic proof configuration, not the full 45K
or full ~3.67B runtime path.

- BitNet-style layers enabled.
- MoE: `8` experts, top-2 routing (`8x2`).
- Liquid layers: `[0, 2, 4, 6]`.
- Hidden size: `768`.
- Layers: `8`.
- Attention heads: `8`.
- KV heads: `4`.
- Sequence length: `512`.
- DDP world size: `2`.

## What This Supports
- 2x H200/H100-class DDP job startup and worker boot were captured.
- Package materialization, package inventory, kernel hotpatch, code module smoke, fixed eval case creation, and deterministic case-bank creation were captured as successful.
- Training step events were captured from step `1` through step `1880`.
- The captured window reached approximately `89.29` minutes, `360,960` samples, and `45,564,022` supervised tokens.
- The captured tail event reported approximately `67.376` samples/sec and `8,504.91` supervised tokens/sec.
- The curriculum opened `multi_op` at step `1020`, approximately progress `0.100887`, which is consistent with the configured progress-based boundary.
- No traceback, OOM, or NCCL failure marker was captured in the cleaned evidence window.

## What This Does Not Support
- It is not a completed `45K` run.
- It is not a completed proof artifact.
- It is not a benchmark-verified result.
- It is not a trained capability claim.
- It is not a production-readiness, mobile-readiness, or security claim.
- It is not a public checkpoint release.
- Final eval, final checkpoint, and final archive were not recovered from this run.
- Training loss alone is not treated as proof of capability. The success
  condition remains checkpoint-bound held-out exact accuracy plus reproducible
  artifacts.

## Cleaned Evidence Summary
| Field | Value |
| --- | --- |
| raw log visibility | private, not committed |
| raw log size | `19,478,098` bytes |
| raw log lines | `48,405` |
| JSON events extracted | `47,467` |
| unique events kept | `298` |
| exact duplicate events removed | `47,169` |
| unique step events | `193` |
| unique heartbeat events | `89` |
| last captured step | `1880` |
| last elapsed minutes | `89.29` |
| last progress | `0.186020` |
| last loss | `0.041666` |
| last weighted CE loss | `0.001362` |
| last grad norm | `0.088019` |

## Curriculum Transition Detail
The run used a progress/time-based curriculum, not a direct loss-threshold
trigger. The captured window showed `multi_op` first appearing at step `1020`,
progress `0.100887`, with weighted CE loss `0.459345` and grad norm `4.947806`.

This is consistent with a healthy new-difficulty transition: loss was low before
the boundary, the harder bucket opened, loss/grad norm spiked, and the later
captured steps returned to low loss. This remains a training-dynamics signal
only; it does not prove final arithmetic capability.

## Previous Partial Run Context
An earlier partial H200 run reached later wall-clock time and included quick eval
events, but quick held-out eval stayed low:

- step `1000`: `1/64` exact = `1.56%`
- step `2000`: `1/64` exact = `1.56%`

That earlier run is useful as pipeline history, not as capability proof.

## Operational Lesson
The cleaned evidence does not show a model-training exception. The weak point was
provider-side job/status/result reliability: status polling, visible logs,
result download, and job visibility became unreliable before final artifact
collection.

The next proof attempt therefore needs more than raw GPU time: persistent output
storage, reliable job identity, terminal log access, checkpoint retention, and an
artifact retrieval path that still works if a web UI polling layer fails.

## Next Compute Ask
- Preferred next window: `8-10 hours of 2x H100/H200 or equivalent`.
- Minimum useful window: `2-4 hours` only for launch and early-curriculum validation.
- Required operational properties: persistent output storage, reliable artifact retrieval, and terminal log access.

Preferred runtime controls:

- SSH or equivalent terminal access.
- `tmux`/`screen` or managed job logs.
- Explicit output directory.
- Restart/resume plan.
- Final artifact sync command.

## Share Policy
Safe to share publicly:

- sanitized summary
- selected step metrics
- startup event summary
- claim boundary
- machine-readable clean summary JSON

Do not share publicly without additional review:

- raw terminal dump
- private repo source
- private decrypt keys
- `.env` files
- access tokens
- model checkpoints/weights
- secret-bearing logs
- full provider account/job pages

## Boundary
This document is a repo-local pointer to cleaned partial operational evidence. It deliberately does not include the raw private terminal log, credentials, checkpoints, provider screenshots, or secret-bearing artifacts.
