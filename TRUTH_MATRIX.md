# TRUTH MATRIX — claim → evidence

Every load-bearing claim with its evidence class. Turkish: [TRUTH_MATRIX_TR.md](TRUTH_MATRIX_TR.md).
Evidence classes: **measured** (ran, reproducible) · **target** (design intent, not yet measured) · **vision** (research direction).

| Claim | Class | Evidence |
|---|---|---|
| Architecture: 18L / 2048 / GQA(16:8) / MoE 8-top2 / Liquid[4,10,16] / BitNet b1.58 | measured | code: `config/config.py`, `model/transformers.py`, `layers/` |
| Measured runtime params = 3,672,982,022 (~3.67B) | measured | `reports/param_accounting_report.md`, `reports/FACTS.json` |
| Design-target params = 2.64B | target | `economics/flops_estimator.py` `DEFAULT_PARAMS` |
| Test suite: 721 passed, 9 skipped (offline) | measured | `pytest` — see [REPRODUCE.md](REPRODUCE.md) |
| Checkpoint save→restore→resume integrity (K4) | measured (local, toy scale) | `scripts/checkpoint_restore_drill.py`, `resume_compat_check.py`; **not yet proven at 45K scale** |
| CfC/MoE fast-path numerical parity (≤1%) | measured (toy scale) | `scripts/cfc_moe_tolerance_check.py` + report |
| Liquid layers improve accuracy | **NOT supported** | [ABLATION.md](ABLATION.md): OFF 96.32% / ON 94.69%, Δ−1.63pp, p=0.305, d=−0.43 — inconclusive; cost (~30% slower) is certain |
| Liquid speed / latency advantage | **no claim** | [ABLATION.md](ABLATION.md) §"NO CLAIM" — all speed numbers confounded until a verified 45K run |
| GPT-3.5-class capability / on-device throughput / NPU latency | target/vision | not measured — requires the 45K run + device profiling |
| Canonical 3.67B model trained & converging | **unverified** | never trained; no real checkpoint — see [STATUS.md](STATUS.md) |
| Documented-not-changed findings (z-loss 2e-6, dt=1.0, GPU-perf) | measured (mechanism) | [DECISIONS.md](DECISIONS.md) — deliberately deferred pre-45K so the run is not confounded |

Generated cross-reference: `reports/final_truth_matrix.md`, `reports/source_of_truth_map.md`.
