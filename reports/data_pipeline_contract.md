# Data Pipeline Contract

- generated_utc: `2026-06-28T11:45:48Z`
- current_training_lane: `remote_bootstrap`
- stage_contract: `stage1..stage5 JSONL must exist before claim-grade training`
- validation_contract: `datasets/validation.jsonl` must remain parseable and above the minimum sample gate

## Current Stage Presence

- `stage1`: `41` rows
- `stage2`: `30` rows
- `stage3`: `8` rows
- `stage4`: `8` rows
- `stage5`: `12` rows

## Boundary

- Stage files now exist in the pinned repo layout.
- Stage4 and stage5 currently include local seed rows to keep the offline-clean contract closed without extra network traffic.
- Final claim-grade corpus evidence still belongs to the real 45K run on the target machine.
