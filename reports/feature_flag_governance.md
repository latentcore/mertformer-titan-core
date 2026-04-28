# Feature Flag Governance

- generated_utc: `2026-04-28T00:13:06Z`

## Canonical Main Path

- `zero_touch_start.sh` -> `scripts/final_orchestrator.py`
- Recommended training lane for this pass: `offline_clean`
- `TITAN_REQUIRE_GATED_TEACHER=1`, `TITAN_USE_PRECOMPUTED_LOGITS=1`, and `TITAN_USE_TR_TOKENIZER=1` define the strict offline-clean lane.

## Non-Canonical / Deferred

- TPU/XLA, multimodal, TurboQuant, and scale-up lanes remain phase-2 or external.
- `run.sh` remains helper-only and must not replace the canonical launcher.
