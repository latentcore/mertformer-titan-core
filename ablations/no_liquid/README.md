# Ablation: No Liquid

**Purpose**: Measure the impact of Liquid layers on stability and routing quality.

**Config change**:
- `use_liquid: false`

**Status**: $0 Kaggle pilot recorded (2026-06-14) — directional signal only, not a claim.

**Pilot signal (measured, claim-safe)**:
- Setup: ~100M proxy, T4 x2, 500 steps, pure next-token CE; identical data + identical init (seed 1234), only `use_liquid` differs.
- Result: `liquid ON mean_last10 = 11.489` vs `liquid OFF mean_last10 = 11.993`; `Δ(off−on) = +0.50` → Liquid directionally helps (lower loss).
- Boundary: noisy curves (constant lr, no warmup/decay), single seed, tiny corpus (35,634 tok / 128k vocab). This is a pilot signal only; not a benchmark claim until a larger measured run.
- Evidence: `reports/ablations/liquid_ablation_kaggle_20260614.json`.
- The full-scale ablation (45K, multi-seed) still requires training hardware; record there in `ablations/results.md`.
