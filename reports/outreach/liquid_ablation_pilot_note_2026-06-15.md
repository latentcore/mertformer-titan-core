# LiquidRouter Ablation — $0 Kaggle Pilot Note (2026-06-15)

**Claim mode: measured (pilot signal only). This is NOT a benchmark claim.**

## What I ran
A controlled LiquidRouter on-vs-off ablation on a free Kaggle GPU (T4 x2), $0:
- ~100M-param proxy MertFormer, pure next-token cross-entropy (no teacher, no KD).
- **Identical data + identical init (seed 1234); the only difference is `use_liquid`.**
- 500 steps per variant, seq 256, batch 8.

## Result
| Variant | mean_last10 loss | params |
| --- | --- | --- |
| Liquid ON | 11.489 | 100.4M |
| Liquid OFF | 11.993 | 99.6M |

`Δ(off − on) = +0.50` → **Liquid directionally helps** (lower loss). Liquid ON also
descended earlier in training. Plot: `reports/ablations/liquid_ablation_pilot_curve.png`.

## Honest boundary
- **Single seed**; the gap (0.50) sits inside the curves' step-to-step noise (constant lr,
  no warmup/decay) — so this is a *direction* signal, not a measured effect size.
- Tiny corpus (35,634 tokens vs a 128k vocab); end-loss hovers near the random baseline
  (ln 128000 ≈ 11.76) even though best steps dipped to ~8.8 — a toy-data artifact, expected.
- This proves the ablation pipeline works and points a direction. It is **not** evidence of
  trained capability, a benchmark, or that the full model "works". That needs a larger,
  multi-seed, measured run on real compute (the 45K architecture-validation run).

## Why it still matters
It is the repository's first measured, reproducible ($0) signal on the headline
architectural component. The honest framing — "directional pilot, not a claim" — is the
point: measured-but-modest beats an inflated number that collapses on inspection.

Reproduce: `python scripts/run_liquid_ablation.py --steps 500 --device cuda --batch-size 8 --seq-len 256`
