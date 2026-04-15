# Anthropic Application Strategy

## Current Decision
- `Apply now`: yes.
- `Interview bar`: stretch but credible.
- `Best framing`: systems-focused research engineer who cares about turning compute into trustworthy signal.
- `Primary role`: `Research Engineer, Science of Scaling`.
- `Secondary role`: `Tokens`, only after the primary packet is coherent.

## Lead With These Signals
- Model work plus training infrastructure plus operational rigor across `train/train.py`, `model/transformers.py`, `layers/moe.py`, and `layers/liquid.py`.
- Systems/debugging discipline across `layers/bitlinear.py`, `mertformer_sdk/kernels/dispatcher.py`, `scripts/final_orchestrator.py`, and `scripts/verify_all.sh`.
- Honesty and evidence discipline across `reports/final_truth_matrix.md`, `reports/known_limits_v1.md`, and `applications/anthropic/measured_evidence_summary.md`.
- Verification and closure behavior that keeps readiness, checkpoints, benchmarks, and release truth tied together instead of narrated loosely.

## Gaps To Say Out Loud
- No real owned long-run checkpoint evidence yet; the remaining post-run class is still open.
- The offline-clean dataset lane is contract-safe and repo-local, but it is not evidence of large-scale ETL ownership.
- Some evaluation surfaces are deterministic mini harnesses and should not be narrated as frontier-grade capability benchmarks.
- Supporting documents with strategic or forecast-heavy tone should not be used as primary evidence.

## Primary Evidence To Show
- `START_HERE.md`
- `docs/PROJECT_MASTER_TRUTH.md`
- `reports/final_truth_matrix.md`
- `reports/known_limits_v1.md`
- `reports/systems_performance_case_study.md`
- `applications/anthropic/project_summary.md`
- `applications/anthropic/measured_evidence_summary.md`

## Evidence To Keep Secondary
- `TECHNICAL_REPORT.md`
- forecast or architecture-rationale reports that are explicitly non-benchmark
- any surface that sounds larger than the checkpoint-bound evidence currently available

## Submission Guidance
1. Submit `Research Engineer, Science of Scaling` first.
2. Use the same evidence packet for `Tokens`, but shift the narrative toward training signal, bottlenecks, and experiment discipline.
3. Do not position yourself as an already-proven frontier scaling scientist.
4. Do position yourself as someone who builds serious ML systems, debugs them honestly, and understands how evidence quality affects research velocity.
