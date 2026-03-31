# Stale Script Report

Legacy or supporting entrypoints that remain intentionally non-canonical:

- `run.sh`: helper entry for tests, SITL demo, and cleanroom verification.
- `scripts/train_smoke.py`: proof-only lane, not the official 45K launcher.
- `scripts/train_tpu_turbo.py`: future or phase-2 TPU lane.
- `scripts/smart_runner.py`: legacy helper, superseded by `scripts/final_orchestrator.py`.

No stale canonical 45K launcher was found outside `zero_touch_start.sh` and `scripts/final_orchestrator.py` in this pass.
