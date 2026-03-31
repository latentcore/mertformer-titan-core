# Usage Guide (Build 30)

This guide is the operational quick-start for reviewers and operators.

## 1) Environment Setup

```bash
bash scripts/bootstrap_venv.sh
```

Baseline:
- Python: 3.11.x
- Default mode: `TITAN_OFFLINE=1`
- Virtualenv: `.titan-venv`

## 2) Fast Verification (Recommended)

```bash
bash scripts/verify_all.sh
```

This runs:
- Secret scan
- Pytest
- Preflight (offline)
- Operator-mode gate (safe offline)

## 3) Canonical 45K Start Gate

```bash
bash zero_touch_start.sh --check-only
```

Use this command before any real 45K training attempt. It refreshes the exact readiness verdict, blocker reason codes, and the canonical runtime contracts.
Treat the 45K run as the first serious architecture validation run, not the final capability ceiling.

Current repo-side state:
- `TRAIN_ALLOWED` via `offline_clean`
- remaining optional blocker: `online_teacher:MISSING_HF_TOKEN`

## 4) Canonical 45K Launcher

```bash
TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh
```

Optional online teacher lane:
```bash
HF_TOKEN=... TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh
```

Common control flags:
- `--plan-only`
- `--dry-run`
- `--resume auto`
- `--resume off`
- `--resume /abs/path/to/checkpoint.pt`
- `--no-post`
- `--post-only`
- `--bench-only`
- `--demo-only`
- `--export-only`
- `--readme-update-only`

## 5) Offline Test Entry

```bash
TITAN_OFFLINE=1 bash run.sh --test
```

Expected:
- No external login/download
- Exit after preflight checks

`run.sh` remains the helper entry for legacy offline/test/demo flows. It is no longer the canonical 45K train-end launcher.

## 6) Operator Gate (Manual)

```bash
.titan-venv/bin/python scripts/operator_mode_gate.py \
  --no-pytest \
  --overfit-dataset datasets/validation.jsonl
```

## 7) Demo Commands

Snake live demo:
```bash
.titan-venv/bin/python snake_demo.py
```

Generate 30s proof video (headless):
```bash
.titan-venv/bin/python snake_demo.py \
  --headless \
  --record assets/snake_demo_proof.mp4 \
  --record-seconds 30
```

## 8) Common Issues

- `onnx export fail` on newer Torch/Python:
  - Use updated exporter path in `scripts/test_onnx_export.py` (already guarded for dynamo/legacy compatibility).
- `wandb/hf token missing`:
  - In the recommended offline-clean lane this is expected and non-blocking.
  - `HF_TOKEN` is required only if you intentionally choose the online teacher lane.
- `venv command path mismatch`:
  - Use module style commands: `.titan-venv/bin/python -m ...`

## 8.5) Auto Cache Sweep Wrapper

Use this wrapper when you want command execution + guaranteed cache cleanup in one call:

```bash
python3 scripts/run_and_clean_pycache.py --full-clean --include-tool-caches --include-venv-caches -- \
  python3 -m pytest -q
```

## 9) Security Policy (Must)

- Never print or commit secret values.
- `.env` must not be packaged.
- Generated logs are artifacts; keep only sanitized docs tracked.

## 10) Pre-Delivery Checklist

- [ ] `bash scripts/verify_all.sh` PASS
- [ ] `bash zero_touch_start.sh --check-only` reviewed
- [ ] `TITAN_OFFLINE=1 bash run.sh --test` PASS
- [ ] README and README_TR docs links valid
- [ ] No missing EN/TR markdown pair
- [ ] Packaging denylist clean (`.env`, venv, caches, logs)
