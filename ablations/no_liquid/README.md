# Ablation: No Liquid

**Purpose**: Measure the impact of Liquid layers on stability and routing quality.

**Config change**:
- `use_liquid: false`

**Status**: **Superseded by the 12-seed multi-seed ablation** (see `ABLATION.md`): OFF 96.32% / ON 94.69% ID exact-accuracy, Δ−1.63 pp, p=0.305 — **no measured benefit, inconclusive** (~30% slower). The single-seed $0 Kaggle pilot below (2026-06-14) was largely one lucky seed; kept for provenance, not a claim.

**Pilot signal (measured, claim-safe)**:
- Setup: ~100M proxy, T4 x2, 500 steps, pure next-token CE; identical data + identical init (seed 1234), only `use_liquid` differs. (`use_liquid` toggles the CfC LiquidMixer at layers [2,4,6] in `layers/liquid.py`; MoE + the Conv1d LiquidRouter stay ON in both arms — this measures the CfC mixer, not the router.)
- Result: `liquid ON mean_last10 = 11.489` vs `liquid OFF mean_last10 = 11.993`; `Δ(off−on) = +0.50` (single-seed; **superseded** — the 12-seed run above shows no measured benefit).
- Boundary: noisy curves (constant lr, no warmup/decay), single seed, tiny corpus (35,634 tok / 128k vocab). This is a pilot signal only; not a benchmark claim until a larger measured run.
- Evidence: `reports/ablations/liquid_ablation_results.json` (full 500-step curves) + `reports/ablations/liquid_ablation_pilot_curve.png` (plot) + `reports/ablations/liquid_ablation_kaggle_20260614.json` (summary).
- The full-scale ablation (45K, multi-seed) still requires training hardware; record there in `ablations/results.md`.

**2026-07-31 note:** an independent external test found the `977.18s` vs `214.81s` (~4.55x) timing delta in this pilot's own JSON — dismissed above as confounded — closer in magnitude to a component-level `LiquidMixer`-vs-`GQA` measurement (~9.4x, different hardware, different method) than to the 12-seed ablation's ~30%. May still be confounded; may also be partly explained by sequence-length scaling (see `BACKLOG.md`/`ABLATION.md`). Not re-litigating the "superseded" status above, just flagging that this timing number may be worth a second look before the 45K run.

**2026-07-31, looked again, same day:** a canonical-`hidden_size` measurement on this repo's own RTX 4060 found train-mode `LiquidMixer` far slower than either number above (~797-1620x vs `GQA`) plus an `OutOfMemoryError` at `seq_len` >= 2048, and a decode-mode measurement found `LiquidMixer` 8-23x *faster* than `GQA`. Full numbers: `BACKLOG.md`. Still not the 45K-scale answer this note asked for, but no longer just "worth a second look" — looked.
