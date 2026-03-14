# Review Checklist (External Engineering)

This checklist is designed for an external engineering review (e.g., security, compliance, reproducibility, ops).

## 1) Clone and Verify (Offline)

Run the deterministic bootstrap:

```bash
bash scripts/bootstrap_venv.sh
```

Run the offline verify-all pipeline:

```bash
bash scripts/verify_all.sh
```

Expected:
- Secret scan PASS
- `pytest` PASS (with documented skips if any)
- Preflight PASS in offline mode
- Operator gate PASS in safe mode (offline)

## 2) Offline-First Contract

Confirm the defaults:
- `TITAN_OFFLINE=1` by default (no HF/WandB login or dataset download)
- Online behavior requires explicit env/flags

Evidence:
- `run.sh` (offline/online gating)
- `scripts/titan_preflight.py` (offline secrets behavior + mock tokenizer)

## 3) Secrets Hygiene

Confirm:
- `.env` is gitignored
- `logs/` is gitignored (artifacts only; `logs/README.md` is the only tracked doc)
- CI secret scan exists and runs on tracked files

Evidence:
- `.gitignore`
- `scripts/secret_scan.py`
- `.github/workflows/ci.yml`

## 4) Dataset Provenance & Licenses (P0 Gate)

Confirm the inventory is complete:
- Auto inventory: `datasets/inventory.md` / `datasets/inventory_TR.md`
- Human-maintained sources: `datasets/SOURCES*.md`
- License checklist: `datasets/LICENSES*.md`
- Snapshot hash registry: `datasets/hashes.json`

Hard rule for production training:
- No **core training** dataset may remain with license status `TBD` (optional/demo datasets must stay disabled until verified)
- Snapshot hashes must be filled for the exact data snapshots used

## 5) Reproducibility

Confirm:
- Python baseline pinned: `pyproject.toml` (`>=3.11,<3.12`)
- Bootstrap script exists and is documented
- Optional accelerate config example exists: `repro/accelerate_default.yaml`

Evidence:
- `repro/python.md`
- `scripts/bootstrap_venv.sh`

## 6) CI Coverage

Confirm CI gates:
- Secret scan
- Ruff (lightweight)
- Preflight (offline)
- Pytest
- Operator gate (safe, offline)

Evidence:
- `.github/workflows/ci.yml`

## 7) Demo (Non-ML Visual)

Install demo deps:

```bash
bash scripts/bootstrap_venv.sh --demo
```

Run:

```bash
.titan-venv/bin/python snake_demo.py
```

Expected:
- Header: `MERTFORMER TITAN v1.0 [LIVE DEMO]`
- Telemetry overlay: `Reasoning Speed: 30ms`, `Tokens: 1.58b`, `Score: X`
- Auto-play and auto-restart on death

## Evidence Packaging (What to Attach)

- The `scripts/verify_all.sh` console output (no tokens)
- Any generated logs under `logs/` (optional, sanitized, not committed)
- Current commit SHA and environment notes

