# Run Contract

This file is the canonical runtime contract for the current 45K closure path.

## Canonical Entrypoint
- `bash zero_touch_start.sh`

## Modes
- `--check-only`: run the fast target-machine start gate and exact readiness contract without launching training.
- `--plan-only`: emit the contracts and planned steps only.
- `--dry-run`: emit the plan plus resolved train command, but do not launch training.
- `--post-only`: skip training and run the post-train state machine.
- `--no-post`: skip the post-train state machine after a successful training run.
- `--bench-only`, `--demo-only`, `--export-only`, `--readme-update-only`: select the post-train subset.

## Start Rules
- Training start is allowed only when `reports/train_readiness_decision.json` says `TRAIN_ALLOWED`.
- The start gate must produce exact blocker reason codes before any full training launch.
- The canonical `offline_clean` lane is strict precomputed KD and keeps `meta-llama/Llama-3.3-70B-Instruct` as the fixed teacher surface.
- `--check-only` intentionally skips the heavyweight `verify_all.sh` sweep and behaves as a target-machine readiness gate.
- This orchestrator uses a JSON lock file to prevent overlapping train-end launches.

## Resume Rules
- `--resume auto`: enable auto-discovery via `TITAN_AUTO_RESUME=1`.
- `--resume off`: disable resume via `TITAN_AUTO_RESUME=0`.
- `--resume /path/to/checkpoint.pt`: set `TITAN_RESUME_FROM` explicitly.

## Post-Train Rule
- Post-train closeout is delegated to `scripts/post_train_autorun.py`.
- Post-train closeout now refreshes both release-side zips and the dedicated training outputs bundle zip for target-machine retrieval.
- No trained evidence claim becomes true unless a real checkpoint is resolved and the downstream artifacts are refreshed from that checkpoint.
