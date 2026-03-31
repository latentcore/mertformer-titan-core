# Train Readiness Decision

- final_status: `TRAIN_ALLOWED`
- decision_reason_code: `READY_OFFLINE_CLEAN`
- recommended_path: `offline_clean`
- guardrail: `At least one readiness path must pass cleanly before TRAIN_ALLOWED is granted.`

## Paths

### offline_clean
- profile: `strict_offline_training_readiness`
- status: `PASS`
- reason_code: `READY`
- exit_code: `0`

### online_teacher
- profile: `strict_online_training_readiness`
- status: `FAIL`
- reason_code: `MISSING_HF_TOKEN`
- exit_code: `1`

## Blockers

- `online_teacher:MISSING_HF_TOKEN`
