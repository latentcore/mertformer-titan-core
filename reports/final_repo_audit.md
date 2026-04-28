# Final Repo Audit

- generated_utc: `2026-04-28T00:13:06Z`
- git_branch: `main`
- git_commit_short: `34d787c`
- readiness_final_status: `NOT_ALLOWED`
- readiness_reason_code: `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE__online_teacher:MISSING_HF_TOKEN`
- recommended_path: `none`

## Working Tree

- tracked_files: `783`
- modified_entries: `117`
- untracked_entries: `5`

## Canonical Gates

- `bash scripts/verify_all.sh`
- `bash zero_touch_start.sh --check-only`
- `bash scripts/one_command_full_sop.sh`
- `bash scripts/final_one_shot.sh`

## Current Closure Boundary

- Zero-touch orchestration is implemented.
- Offline-clean readiness is green.
- Online teacher remains an alternate lane with external credential dependency when explicitly requested.
- Real 45K outputs remain post-run evidence, not current fact.

## Artifacts

- `artifacts/mertformer_release.zip` (107268328 bytes)
- `artifacts/mertformer_training_outputs_bundle.zip` (203639211 bytes)
- `artifacts/target_machine_handoff_bundle.zip` (29628 bytes)
- `packages/MertFormer_5080_Final_Delivery_20260421_213941.zip` (122233 bytes)
- `packages/MertFormer_5080_Final_Delivery_20260421_221701.zip` (122904 bytes)
- `packages/MertFormer_5080_Final_Delivery_20260421_221919.zip` (362147 bytes)
- `packages/MertFormer_5080_Final_Delivery_20260421_222608.zip` (362057 bytes)
- `packages/MertFormer_Titan_OnyxStorm_v2.0_B30_Release.zip` (52343392 bytes)
