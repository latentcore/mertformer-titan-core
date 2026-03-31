# Teacher Decision Record

- generated_utc: `2026-03-31T20:59:49Z`
- canonical_training_lane: `offline_clean`
- alternate_lane: `online_teacher`
- readiness_final_status: `TRAIN_ALLOWED`
- readiness_reason_code: `READY_OFFLINE_CLEAN`

## Decision

- The current repo-side recommended path is the offline-clean lane.
- The online teacher lane remains available only when `HF_TOKEN` and gated access are intentionally supplied.
- The canonical launcher may proceed teacher-free when the offline-clean path is selected and precomputed logits are unavailable.

## Policy Boundary

- No consumer-AI scraping is allowed as teacher data.
- No gated-teacher claim is allowed without a valid credential and approved access.
- Trained-model evidence remains post-run only.

## Current Lane Status

- offline_clean: `PASS` / `READY`
- online_teacher: `FAIL` / `MISSING_HF_TOKEN`
