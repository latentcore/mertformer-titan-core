# Final Repo Audit

- generated_utc: `2026-07-31T10:35:56Z`
- git_branch: `main`
- git_commit_short: `bd3b776c`
- readiness_final_status: `TRAIN_ALLOWED`
- readiness_reason_code: `READY_REMOTE_BOOTSTRAP`
- recommended_path: `remote_bootstrap`

## Working Tree

- tracked_files: `967`
- modified_entries: `185`
- untracked_entries: `0`

## Canonical Gates

- `bash scripts/verify_all.sh`
- `bash zero_touch_start.sh --check-only`
- `bash scripts/one_command_full_sop.sh`
- `bash scripts/final_one_shot.sh`

## Current Closure Boundary

- Zero-touch orchestration is implemented.
- Current repo-side recommended lane: `remote_bootstrap`.
- Offline-clean remains the strict local path; remote-bootstrap remains the rented-machine runtime-injected path.
- Online teacher remains an alternate lane with external credential dependency when explicitly requested.
- Real 45K outputs remain post-run evidence, not current fact.

## Artifacts

- `artifacts/mertformer_release.zip` (27871003 bytes)
- `artifacts/mertformer_training_outputs_bundle.zip` (44307513 bytes)
- `artifacts/target_machine_handoff_bundle.zip` (155103 bytes)
- `packages/MertFormer_Titan_OnyxStorm_v2.0_B30_Release.zip` (13916126 bytes)
