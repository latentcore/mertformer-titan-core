# Known Limits v1

## Current Measured Truth
- Repo-side verification, truth-sync, and governance surfaces are active.
- Chess onefile delivery, runtime containment, and Stockfish auto-fetch are implemented.
- Repo-side training readiness is currently `TRAIN_ALLOWED` with blockers `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE, online_teacher:MISSING_HF_TOKEN`.
- Exact `45K` remains the preferred main-run target, but application readiness is gated by a real owned training run plus checkpoint-bound evidence rather than the exact step count alone.
- Costly large-scale compute is not a personal-funding requirement; truthful verified evidence is the actual gate.

## Not Yet Measured
- trained final weights from a real owned training run
- best/latest checkpoint proof from the real owned training run
- claim-grade benchmark outputs tied to trained checkpoints
- final checkpoint-bound evidence pack
- trained-model export or edge/mobile measurement (strong plus, not a hard blocker)

## Chess-Specific Limit
- Internal proxy strength and readiness surfaces exist, but real strength claims still require post-run benchmark evidence.

## Research-Lane Limit
- `3000+ Elo`, `20 ms/move`, `10000x speedup`, and similar moonshots remain non-release research claims unless separately measured.
