# Python Baseline (Review-Ready)

This repository is pinned for a **Python 3.11** baseline to minimize ecosystem surprises
(PyTorch/Transformers compatibility, stable wheels, and the `snake_demo.py` pygame demo).

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
