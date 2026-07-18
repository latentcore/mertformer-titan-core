# 5-minute CPU quickstart (no GPU, no downloads, $0)

Everything here runs offline on a laptop CPU/MPS. None of it needs a GPU, an HF token, or the teacher.

```bash
# 1) set up the local venv (once)
bash scripts/verify_all.sh        # bootstraps .titan-venv, runs the full gate (pytest + ruff + mypy + checks)

# 2) "can it train at all?" — tiny smoke (seconds)
.titan-venv/bin/python scripts/train_smoke.py --steps 20 --device cpu --cleanup

# 3) the $0 LiquidRouter ablation pilot (Liquid ON vs OFF), 3-step smoke
.titan-venv/bin/python scripts/run_liquid_ablation.py --steps 3 --device cpu --batch-size 2 --seq-len 64
#    → reports/ablations/liquid_ablation_results.json   (see docs/KAGGLE_PILOT.md for the real run)

# 4) liquid implementation micro-benchmark
.titan-venv/bin/python scripts/benchmark_liquid_impls.py --iters 10

# 5) numerical-equivalence gate (CfC/MoE tolerance, CPU)
.titan-venv/bin/python scripts/cfc_moe_tolerance_check.py
```

Tests only: `.titan-venv/bin/python -m pytest -q` (expects the recorded `553 passed, 5 skipped`; see
`reports/FACTS.json`). The skips are environment-gated (no GPU locally; optional
onnxruntime/UnitaryQINN dependency unavailable), not CUDA alone.
