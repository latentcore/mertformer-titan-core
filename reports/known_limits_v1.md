# Known Limits v1

## Current Measured Truth
- Repo-side verification, truth-sync, and governance surfaces are active.
- Chess onefile delivery, runtime containment, and Stockfish auto-fetch are implemented.
- Repo-side training readiness is currently `TRAIN_ALLOWED` with blockers `online_teacher:MISSING_HF_TOKEN`.

## Not Yet Measured
- trained final weights
- best/latest checkpoint proof from the real main run
- claim-grade benchmark outputs tied to trained checkpoints
- final evidence pack tied to the real run
- trained-model export or edge/mobile measurement

## Chess-Specific Limit
- Internal proxy strength and readiness surfaces exist, but real strength claims still require post-run benchmark evidence.

## Research-Lane Limit
- `3000+ Elo`, `20 ms/move`, `10000x speedup`, and similar moonshots remain non-release research claims unless separately measured.
