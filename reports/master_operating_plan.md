# Master Operating Plan

- canonical verification: `bash scripts/verify_all.sh`
- canonical closure refresh: `bash scripts/final_one_shot.sh`
- canonical 45K start gate: `bash zero_touch_start.sh --check-only`
- canonical 45K launcher: `bash zero_touch_start.sh`

## Definitions
- done: code path wired, report exists, gate passes, doc points to the same truth.
- shipped: closure artifacts and package artifacts refreshed without stale claim drift.
- trusted: measured claims stay backed by current artifacts.
- investable: trained evidence, benchmark, demo, and GTM package all exist together.
- customer-ready: trained evidence plus install/support/legal package exist together.
