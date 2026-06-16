# Teacher Decision Record

- generated_utc: `2026-06-16T19:01:27Z`
- canonical_training_lane: `offline_clean`
- remote_handoff_lane: `remote_bootstrap`
- alternate_lane: `online_teacher`
- readiness_final_status: `TRAIN_ALLOWED`
- readiness_reason_code: `READY_REMOTE_BOOTSTRAP`

## Decision

- The current repo-side recommended path is the remote-bootstrap lane.
- The online teacher lane remains available only when `HF_TOKEN` and gated access are intentionally supplied.
- The canonical offline-clean launcher is now strict precomputed KD: completed logits shards or actionable Phase-0 precompute are required before start.
- The remote-bootstrap lane is allowed when the target machine will inject `HF_TOKEN` and run dataset/bootstrap steps there.

## Policy Boundary

- No consumer-AI scraping is allowed as teacher data.
- No gated-teacher claim is allowed without a valid credential and approved access.
- Trained-model evidence remains post-run only.

## Current Lane Status

- offline_clean: `FAIL` / `PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`
- remote_bootstrap: `PASS` / `READY_RUNTIME_INJECTED_BOOTSTRAP`
- online_teacher: `FAIL` / `MISSING_HF_TOKEN`
