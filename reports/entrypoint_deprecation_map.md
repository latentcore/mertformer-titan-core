# Entrypoint Deprecation Map

Supporting or legacy entrypoints that should not replace the canonical ladder.

| Path | Allowed Role | Replace Canonical? | Notes |
| --- | --- | --- | --- |
| `run.sh` | supporting launcher for test/demo/helper flows | no | Keep for `--test`, `--sitl-demo`, `--cleanroom-verify`, and legacy helper flows; canonical 45K launcher is `zero_touch_start.sh`. |
| `scripts/smart_runner.py` | legacy training helper | no | Superseded by `scripts/final_orchestrator.py` for the canonical 45K path. |
| `scripts/operator_mode_gate.py` | supporting preflight or operator checks | no | Useful support command, not the canonical closure front door. |
| `scripts/release_build30.sh` | release helper | no | Called by the one-command closure flow. |
| `scripts/train_smoke.py` | smoke-only validation | no | Not the official 45K launcher. |
| `scripts/train_tpu_turbo.py` | TPU-specific experiment path | no | Future or phase-2 validation lane. |
| `snake_demo.py` | showcase demo | no | Demonstration surface, not an operational entrypoint. |
