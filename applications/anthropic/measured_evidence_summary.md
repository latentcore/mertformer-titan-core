# Measured Evidence Summary

## Repo-Wide Evidence
- `bash scripts/verify_all.sh` is the canonical repo verification gate.
- Train readiness reports `TRAIN_ALLOWED` with reason `READY_OFFLINE_CLEAN`.
- Claim-critical docs explicitly separate measured, target, and vision surfaces.
- Closure scripts regenerate reports, hashes, and package artifacts instead of relying on narration.

## Systems and Scaling Evidence
- Low-bit runtime surfaces are implemented with explicit backend routing and fallback discipline.
- Kernel and dispatcher behavior is covered by targeted tests rather than presented as an unmeasured speed story.
- Repo-side benchmark artifacts exist for smoke/reference surfaces while trained-checkpoint claims remain open.
- The repo keeps compute, checkpoint, benchmark, and release evidence tied to manifests and exact artifacts.

## Product and Assistant Evidence
- Offline-first is the repo-default operating direction.
- Local retrieval, tool governance, telemetry, and memory are present as explicit system surfaces.
- The assistant lane is intentionally described as foundation work, not as a finished end-user product.

## Evaluation-Discipline Evidence
- The chess lane keeps teaching contracts, proxy benchmarking, replay output, and strength claims separate.
- The repo treats benchmark honesty as an engineering feature rather than a presentation problem.

## Evidence Still Missing By Design
- real owned training run outputs
- trained final weights
- best/latest checkpoint proof
- checkpoint-bound benchmark outputs
- trained demo bundle
- trained export and device measurements
