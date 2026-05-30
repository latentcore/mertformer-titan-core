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
- `TRAIN_ALLOWED` via `remote_bootstrap`
- strict local `offline_clean` remains blocked without local logits or local actionable Phase-0
- remaining non-winning blocker: `online_teacher:MISSING_HF_TOKEN`

## 4) Canonical 45K Launcher

```bash
HF_TOKEN=... TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh
```

Strict local offline-clean lane (only when logits or local Phase-0 are satisfied):
```bash
TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh
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

## 4.5) Optional Target-Machine Speed Controls

These controls are wired for smoke/full-run tuning, but they are **not** benchmark claims. Keep them off for the conservative baseline lane unless the equivalence tests and a short target-machine smoke run pass.

Equivalence check for the optional packed projection and Liquid train paths:

```bash
python3 -m pytest -q tests/test_packed_projection_equivalence.py tests/test_liquid_safeguard.py
```

Example Ocean 2x H200 smoke command after the canonical gate is reviewed:

```bash
HF_TOKEN=... \
TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable \
TITAN_BATCH_SIZE=1024 TITAN_BATCH_SIZE_FALLBACKS=1024,512,256 \
TITAN_DATALOADER_PIN=1 TITAN_DATALOADER_NONBLOCKING=1 \
TITAN_FFN_PACK=1 TITAN_MOE_PACK=1 TITAN_MLA_KV_PACK=1 \
TITAN_LIQUID_FAST_PATH=0 TITAN_LIQUID_TRAIN_IMPL=packed_pair \
MERTFORMER_LOWBIT_KERNEL=0 MERTFORMER_FUSED_BACKWARD=0 \
bash zero_touch_start.sh --dry-run
```

Operational boundaries:
- `repro/accelerate_8xgpu.yaml` lives under `repro/` because it is a reproducibility/run configuration, not a stable model config contract under `configs/`.
- `TITAN_BATCH_SIZE_FALLBACKS=1024,512,256` is used only for clear OOM signals; non-OOM failures stop without changing batch.
- `TITAN_FFN_PACK`, `TITAN_MOE_PACK`, and `TITAN_MLA_KV_PACK` are default-off and covered by equivalence tests.
- The first Ocean long run keeps `TITAN_LIQUID_FAST_PATH=0` and avoids `packed_pair_compile`.
- If `MERTFORMER_LOWBIT_KERNEL=1` is enabled, packed projection paths fall back to the baseline path to avoid bypassing the experimental low-bit inference kernel boundary.
- Real throughput must come from target-machine logs; do not promote projected speedups to measured claims.

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
  - On a local review machine this can be expected.
  - On the recommended `remote_bootstrap` target-machine lane, `HF_TOKEN` must be injected before launch.
  - The strict local `offline_clean` lane can remain green only when logits already exist locally or local Phase-0 is actionable.
- `venv command path mismatch`:
  - Use module style commands: `.titan-venv/bin/python -m ...`
- Optional speed flags cause mismatch or instability:
  - Disable the relevant flag first (`TITAN_FFN_PACK=0`, `TITAN_MOE_PACK=0`, `TITAN_MLA_KV_PACK=0`, or `TITAN_LIQUID_TRAIN_IMPL=baseline`) and rerun the equivalence tests before any long run.

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
- [ ] Optional packed/Liquid equivalence tests PASS if any speed flags are enabled
- [ ] `TITAN_OFFLINE=1 bash run.sh --test` PASS
- [ ] README and README_TR docs links valid
- [ ] No missing EN/TR markdown pair
- [ ] Packaging denylist clean (`.env`, venv, caches, logs)
