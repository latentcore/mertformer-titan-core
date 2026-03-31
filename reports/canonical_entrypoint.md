# Canonical Entrypoint

Current command ladder for the working tree.

| Command | Role | Current Status | Notes |
| --- | --- | --- | --- |
| `bash zero_touch_start.sh --check-only` | canonical start gate and exact readiness verdict | `active` | Canonical preflight -> train/resume -> post-train closeout launcher with run lock. |
| `bash zero_touch_start.sh` | canonical train-end 45K launcher | `active` | Uses the final orchestrator plus post-train state machine. |
| `bash scripts/verify_all.sh` | canonical verification gate | active | Offline-first verification and report refresh. |
| `bash scripts/one_command_full_sop.sh` | one-command closure validation flow | active | Builds verification, packaging, and report artifacts. |
| `bash scripts/final_one_shot.sh` | max closeout and release refresh | active | Runs one-command SOP first, then release-side extras. |
| `python3 scripts/build_train_readiness_contract.py --allow-not-ready` | exact readiness decision | active | Emits current blocker reason codes. |
