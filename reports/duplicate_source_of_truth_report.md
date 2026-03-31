# Duplicate Source-of-Truth Report

Canonical source-of-truth order is enforced by `AGENTS.md` and `reports/source_of_truth_map.md`.

## Canonical Command Ladder
- `bash zero_touch_start.sh --check-only`
- `bash zero_touch_start.sh`
- `bash scripts/verify_all.sh`
- `bash scripts/one_command_full_sop.sh`
- `bash scripts/final_one_shot.sh`

## Supporting Only
- `run.sh` is intentionally retained for helper flows and is not a duplicate canonical launcher.

No conflicting second source-of-truth file was promoted in this pass.
