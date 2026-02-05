# Postmortem — Example 001 (Dry-Run)

- **Incident ID**: PM-001
- **Date/Time**: 2026-02-05 07:40
- **Impact**: Training run aborted at step 120 due to NaN spike.
- **Root Cause**: Gradient explosion after LR warmup; grad clip threshold too high.
- **Detection**: `nan_kill_test.py` triggered safety gate; log flagged NaN.
- **Resolution**: Lowered `grad_clip` and added extra loss checks.
- **Prevention**: Add early warning on grad norm trend; enforce cap on LR.

Status: **Dry-run template filled for process readiness.**
