# Final Orchestrator Status

- run_id: `zero_touch_20260731T095957Z`
- mode: `dry-run`
- status: `dry-run`
- generated_utc: `2026-07-31T09:59:57.387724+00:00`
- train_readiness_status: `none`
- decision_reason_code: `none`
- training_lane: `none`
- resume_policy: `auto`
- train_command: `<REPO_ROOT>\.titan-venv\bin\python -m accelerate.commands.launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --main_process_port 29501 train/train.py`
- post_mode: `full`

## Steps

| Step | Status | Return Code | Notes |
| --- | --- | --- | --- |
| `contract_outputs` | `completed` | `0` | Run contract, artifact list, exit code standard, and run manifest schema refreshed. |
| `post_train_plan_refresh` | `completed` | `0` | {"status": "planned", "mode": "plan-only", "checkpoint": null}
 |
| `start_gate` | `planned` | `0` | Refresh verify/readiness gate and exact blocker report. |
| `training` | `planned` | `0` | Launch accelerate training path with resume policy applied. |
| `post_train` | `planned` | `0` | Run post-train state machine in full mode. |
