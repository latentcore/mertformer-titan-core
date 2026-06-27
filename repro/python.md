# Python Baseline (Review-Ready)

This repository targets a **Python 3.11** baseline to minimize ecosystem surprises
(stable wheels and the `snake_demo.py` pygame demo). The exact, authoritative
runtime versions are recorded in `repro/pip_freeze.txt` and `repro/env.lock`
(currently `torch==2.10.0`, `transformers==5.3.0`); treat those lock files as the
source of truth rather than this prose. The "compatibility" rationale here is
descriptive intent, not a measured guarantee that 3.11 is the only supported
interpreter for the pinned versions.

## Local Setup (macOS/Linux)

```bash
python3.11 -m venv .titan-venv
.titan-venv/bin/python -m pip install -U pip wheel setuptools
.titan-venv/bin/python -m pip install -r requirements.txt
.titan-venv/bin/python -m pip install -e ".[dev]"
```

Optional demo dependencies:

```bash
.titan-venv/bin/python -m pip install -e ".[demo]"
```

Or use the bootstrap helper:

```bash
bash scripts/bootstrap_venv.sh        # dev
bash scripts/bootstrap_venv.sh --demo # dev + demo (pygame)
```

## Notes
- Offline-first runtime: set `TITAN_OFFLINE=1` (default).
- Online mode requires `HF_TOKEN` (and optionally `WANDB_API_KEY` if `TITAN_WANDB=1`).
