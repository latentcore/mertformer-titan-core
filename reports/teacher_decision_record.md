# Teacher Decision Record

- generated_utc: `2026-04-28T00:13:06Z`
- canonical_training_lane: `offline_clean`
- alternate_lane: `online_teacher`
- readiness_final_status: `NOT_ALLOWED`
- readiness_reason_code: `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE__online_teacher:MISSING_HF_TOKEN`

## Decision

- The current repo-side recommended path is the offline-clean lane.
- The online teacher lane remains available only when `HF_TOKEN` and gated access are intentionally supplied.
- The canonical offline-clean launcher is now strict precomputed KD: completed logits shards or actionable Phase-0 precompute are required before start.

## Policy Boundary

- No consumer-AI scraping is allowed as teacher data.
- No gated-teacher claim is allowed without a valid credential and approved access.
- Trained-model evidence remains post-run only.

## Current Lane Status

- offline_clean: `FAIL` / `PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`
- online_teacher: `FAIL` / `MISSING_HF_TOKEN`
