# Train Readiness Decision

- final_status: `TRAIN_ALLOWED`
- decision_reason_code: `READY_REMOTE_BOOTSTRAP`
- recommended_path: `remote_bootstrap`
- guardrail: `At least one readiness path must pass cleanly before TRAIN_ALLOWED is granted; offline_clean stays strict precomputed KD, while remote_bootstrap is valid only when the rented-machine bootstrap flow can inject credentials and fetch datasets at runtime.`

## Paths

### offline_clean
- profile: `strict_offline_training_readiness`
- status: `FAIL`
- reason_code: `PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`
- exit_code: `1`

### remote_bootstrap
- profile: `runtime_injected_training_readiness`
- status: `PASS`
- reason_code: `READY_RUNTIME_INJECTED_BOOTSTRAP`
- exit_code: `0`

### online_teacher
- profile: `strict_online_training_readiness`
- status: `FAIL`
- reason_code: `MISSING_HF_TOKEN`
- exit_code: `1`

## Blockers

- `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`
- `online_teacher:MISSING_HF_TOKEN`
