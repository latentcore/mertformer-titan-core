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

## 3) Offline Test Entry

```bash
TITAN_OFFLINE=1 bash run.sh --test
```

Expected:
- No external login/download
- Exit after preflight checks

## 4) Operator Gate (Manual)

```bash
.titan-venv/bin/python scripts/operator_mode_gate.py \
  --no-pytest \
  --overfit-dataset datasets/validation.jsonl
```

## 5) Demo Commands

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

## 6) Common Issues

- `onnx export fail` on newer Torch/Python:
  - Use updated exporter path in `scripts/test_onnx_export.py` (already guarded for dynamo/legacy compatibility).
- `wandb/hf token missing`:
  - In offline mode this is expected and non-blocking.
- `venv command path mismatch`:
  - Use module style commands: `.titan-venv/bin/python -m ...`

## 6.5) Auto Cache Sweep Wrapper

Use this wrapper when you want command execution + guaranteed cache cleanup in one call:

```bash
python3 scripts/run_and_clean_pycache.py --full-clean --include-tool-caches -- \
  python3 -m pytest -q
```

## 7) Security Policy (Must)

- Never print or commit secret values.
- `.env` must not be packaged.
- Generated logs are artifacts; keep only sanitized docs tracked.

## 8) Pre-Delivery Checklist

- [ ] `bash scripts/verify_all.sh` PASS
- [ ] `TITAN_OFFLINE=1 bash run.sh --test` PASS
- [ ] README and README_TR docs links valid
- [ ] No missing EN/TR markdown pair
- [ ] Packaging denylist clean (`.env`, venv, caches, logs)
