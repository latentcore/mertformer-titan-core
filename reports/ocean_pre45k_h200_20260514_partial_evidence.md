# Ocean Pre-45K H200 Partial Evidence - 2026-05-14

## Status
- Claim mode: `measured`
- Status: `partial_operational_evidence_only`
- Public Gist pointer: `https://gist.github.com/latentcore/dac0aa0c56b12177e4a0e8e8f684bccf`
- Machine-readable summary: `reports/ocean_pre45k_h200_20260514_clean_summary.json`

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

## Next Compute Ask
- Preferred next window: `8-10 hours of 2x H100/H200 or equivalent`.
- Minimum useful window: `2-4 hours` only for launch and early-curriculum validation.
- Required operational properties: persistent output storage, reliable artifact retrieval, and terminal log access.

## Boundary
This document is a repo-local pointer to cleaned partial operational evidence. It deliberately does not include the raw private terminal log, credentials, checkpoints, provider screenshots, or secret-bearing artifacts.
