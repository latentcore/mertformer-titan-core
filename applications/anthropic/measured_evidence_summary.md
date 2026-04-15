# Measured Evidence Summary

## Repo-Wide Evidence
- `bash scripts/verify_all.sh` is the canonical repo verification gate.
- Train readiness reports `TRAIN_ALLOWED` with reason `READY_OFFLINE_CLEAN`.
- Claim-critical docs explicitly separate measured, target, and vision surfaces.
- Closure scripts regenerate reports, hashes, and package artifacts instead of relying on narration.

## Current Validation Snapshot
- `python3 -m pytest --collect-only -q` currently collects `205` tests.
- `pytest -q tests/test_titan_preflight_contract.py tests/test_orchestrator_swarm_runtime.py tests/test_agi_cognitive.py tests/test_triad_omega_api.py` now passes from the repo root after the pytest bootstrap fix.
- `pytest -q tests/test_architecture_integrity.py` covers the former MPS stress regression path with `cfg.max_seq_len` aligned to the exercised sequence length.
- `.lint-venv/bin/ruff check model layers train orchestrator utils eval scripts tests` passes.

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

## Gaps To Disclose Directly
- The current offline-clean dataset lane is contract-safe and seed-level, not evidence of large-scale ETL ownership.
- Some evaluation surfaces are deterministic mini harnesses and should not be framed as frontier-grade capability benchmarks.
- Forecast-heavy or strategic reports should remain secondary to the canonical truth and evidence surfaces.
