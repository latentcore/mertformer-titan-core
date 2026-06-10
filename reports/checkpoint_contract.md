# Checkpoint Contract

- generated_utc: `2026-06-10T22:25:16Z`
- save_dir: `checkpoints/mertformer_titan_prod`
- naming: `<model_name>_step_<n>.pt`, `<model_name>_latest.pt`, `<model_name>_best.pt`
- retention_policy: keep latest plus bounded recent step checkpoints and the best checkpoint

## Current Evidence

- resume_compat_status: `PASS`
- final_orchestrator_status: `planned`

## Boundary

- The naming and retention contract exists in code today.
- Real checkpoint hashes and trained checkpoint proof remain post-run evidence.
