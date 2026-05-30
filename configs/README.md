# Configs

Canonical configuration surface for stable, named configuration contracts.

Current scope:
- chess onefile profile contract
- profile intent, not generated run outputs

Generated run-time resolved configs remain under run-local `reports/resolved_config.json`.

Boundary:
- Accelerate launch profiles and target-machine reproducibility files belong under `repro/`.
- Example: `repro/accelerate_8xgpu.yaml` is intentionally not stored here because it controls a run environment, not the model/config contract itself.
