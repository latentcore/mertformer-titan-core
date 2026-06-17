# Ablation Results

Status: **Pending measurement** (requires full training hardware).

How to fill this table:
- Run each ablation config (see `ablations/*/README.md`).
- Record final loss, convergence notes, and any stability issues.
- Add benchmark deltas (HumanEval/MBPP/GSM8K) after training runs.

| Ablation | Goal | Status | Notes |
| --- | --- | --- | --- |
| no_moe | Measure dense-only baseline | Pending | Run on training hardware |
| no_liquid | Measure impact of the CfC LiquidMixer (`use_liquid`, layers [2,4,6]; MoE+router on in both arms) | **12-seed verdict** (see `ABLATION.md`) | 12-seed multi-seed: OFF 96.32% / ON 94.69% ID exact-acc, Δ−1.63 pp, p=0.305, inconclusive — **no measured benefit, ~30% slower**. The earlier single-seed Kaggle pilot (2026-06-14, Δ(off−on)=+0.50, "directionally helps") was largely one lucky seed and is **superseded**. Pilot data `reports/ablations/liquid_ablation_results.json` + plot `reports/ablations/liquid_ablation_pilot_curve.png` |
| dense_only | MoE + Liquid off | Pending | Run on training hardware |
| bitlinear_off | BitNet off baseline | Pending | Run on training hardware |
