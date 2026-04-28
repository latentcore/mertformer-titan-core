# Train Readiness Decision

- final_status: `NOT_ALLOWED`
- decision_reason_code: `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE__online_teacher:MISSING_HF_TOKEN`
- recommended_path: `none`
- guardrail: `At least one readiness path must pass cleanly before TRAIN_ALLOWED is granted; offline_clean is canonical only when strict precomputed KD prerequisites are satisfied.`

## Paths

### offline_clean
- profile: `strict_offline_training_readiness`
- status: `FAIL`
- reason_code: `PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`
- exit_code: `1`

### online_teacher
- profile: `strict_online_training_readiness`
- status: `FAIL`
- reason_code: `MISSING_HF_TOKEN`
- exit_code: `1`

## Blockers

- `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`
- `online_teacher:MISSING_HF_TOKEN`
