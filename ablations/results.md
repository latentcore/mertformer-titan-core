# Ablation Results

Status: **Pending measurement** (requires full training hardware).

How to fill this table:
- Run each ablation config (see `ablations/*/README.md`).
- Record final loss, convergence notes, and any stability issues.
- Add benchmark deltas (HumanEval/MBPP/GSM8K) after training runs.

| Ablation | Goal | Status | Notes |
| --- | --- | --- | --- |
| no_moe | Measure dense-only baseline | Pending | Run on training hardware |
| no_liquid | Measure impact of Liquid layers | Pilot signal (2026-06-14) | $0 Kaggle T4x2, 500 steps: Δ(off−on)=+0.50, Liquid directionally helps; pilot-signal-only, single seed. Full-scale on training hardware. Data `reports/ablations/liquid_ablation_results.json` + plot `reports/ablations/liquid_ablation_pilot_curve.png` |
| dense_only | MoE + Liquid off | Pending | Run on training hardware |
| bitlinear_off | BitNet off baseline | Pending | Run on training hardware |
