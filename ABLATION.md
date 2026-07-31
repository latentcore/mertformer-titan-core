# Ablations — Canonical Results

> Language: **English** | [Türkçe](ABLATION_TR.md)
> Claim mode: `measured` (small-scale, directional). This is **NOT** a benchmark claim about
> the 3.67B model. The canonical 3.67B run has not happened yet.

This is the single canonical surface for MertFormer Titan ablation results. Raw artifacts and
SHA256 manifest live in [`reports/ablations/liquid_ablation_final_20260615/`](reports/ablations/liquid_ablation_final_20260615/).

## LiquidRouter / CfC mixer — 12-seed ablation (FINAL, 2026-06-15)

**Setup (toy scale):** `use_liquid` ON vs OFF, 12 seeds × 2 arms = 24 runs. Small preset
(hidden 384, 8 layers, 6/2 heads, 8 experts top-2, `liquid_idx=[1,4]`, vocab 14), 2-digit
addition, Kaggle T4, ~900 s/run, AMP off. Held-out ID = in-distribution (2-digit); OOD = 3-digit
magnitude extrapolation. Source of truth: `final_summary.json` + `MANIFEST.json` (sha256-chained).

**Result (primary data):**

| Metric | Liquid OFF | Liquid ON | Δ (ON−OFF) | stats |
|---|---|---|---|---|
| Held-out ID exact-acc | **96.32%** | **94.69%** | −1.63 pp | p=0.305, Cohen's d=−0.43, 95% CI [−4.63, +1.17] |
| OOD (3-digit) exact-acc | 0.0% | 0.0% | +0.00 pp | p=nan (both at floor) |
| Final train loss | 1.2689 | 1.2704 | +0.0015 | ON marginally worse |
| Steps (equal wall-clock) | ~6056 | ~4363 | ×0.72 | ON ~30% fewer steps |
| Params | 42.10M | 43.28M | +1.18M (+2.8%) | small param cost |

`decoupling_detected = false`. Seed variance is high (e.g. ON: seed0 = 100%, seed1 = 83.8%).

### What this experiment can and cannot say (measurement validity)
- **Dynamic range is collapsed.** ID is ceiling-bound (~95% both arms) and OOD is floor-bound
  (0% both arms). A small architectural effect can only show in the middle band, which this task
  does not have. So `OOD = 0/0` is **not** evidence for or against Liquid (a floor effect cannot
  discriminate).
- **Comparison is iso-wall-time, not iso-step.** OFF ran ~40% more steps in the same seconds, so a
  *scientific* "Liquid adds nothing" claim would need iso-step / iso-token; only the *deployment*
  claim ("under a fixed time budget Liquid loses") is supported here. The runner prints this
  caveat itself (COMPUTE ASYMMETRY).
- **Underpowered.** With this seed variance, 12 seeds cannot detect a small-but-real effect;
  `p=0.305` means "failed to reject", not "proven zero". The minimum detectable effect was not
  computed.
- **Pilot belief mechanism.** The earlier single-seed pilot signal (Δ(off−on)=+0.50) was largely
  one lucky seed (seed0 ON = 100%); the 12-seed mean is 94.69%.

### Verdict
At this toy scale, under a fixed time budget, **Liquid showed no visible accuracy benefit and was
~30% slower per wall-clock**, with a small (+2.8%) parameter cost. But because the task is
ceiling/floor-saturated, the test is underpowered, and the comparison is iso-time, the honest
reading is **inconclusive on Liquid's value at scale — the cost is certain, the benefit is
unmeasured (not disproven).** Whether to keep Liquid in the 45K run is a separate decision; this
experiment does not provide positive evidence for it.

### Liquid speed/latency: NO CLAIM
This repo makes **no absolute speed/latency claim for Liquid** until a verified 45K run produces
real, scale-representative data. The pilot and H200 numbers are confounded (torch.compile warmup,
run ordering, no T4 fast-path). The only controlled observation is directional: at toy scale, on
T4, iso-wall-time, Liquid ON was slower — not a production claim.

### External signal (2026-07-31, informational only — not a repo measurement)
An independent, external test against this repo's own `layers/liquid.py`/`layers/mla.py` (different
hardware — consumer GPU, `hidden_size=256`, `seq_len=128`, no BitNet/MoE co-training) measured
`LiquidMixer` at ~9.4x `GQA`'s per-call wall-clock time, and surfaced a mechanism not previously
written down here: `LiquidCell`'s recurrence is a sequential loop over the time dimension, so its
cost scales with `seq_len` in a way attention's does not — meaning the ~30% figure above (measured
at this ablation's much shorter effective sequence) may not hold at the canonical `seq_len=4096`.
Full detail and a candidate cheap pre-45K validation item: see [BACKLOG.md](BACKLOG.md), entry
"External signal on Liquid/CfC wall-clock cost." Does not change the verdict above or the
`DECIDED: Keep` call in [reports/liquid_keep_or_drop_brief.md](reports/liquid_keep_or_drop_brief.md).

## Other ablations (pending — require training hardware)
`ablations/` holds scaffolds for `no_moe`, `dense_only`, `bitlinear_off`. These require real GPU
training and have not been run. Component value remains a hypothesis until measured at scale.

Two additional `layers/moe.py` components are flag-gated off by default (`use_structural_plasticity`,
`use_cross_expert_sync_bus`) and, like the components above, have never been ablated: `structural_plasticity`
(periodic prune/grow of active experts by usage EMA) and `cross_expert_sync_bus` (optional
attention-independent cross-expert coordination signal). Neither has a scaffold under `ablations/`
yet and neither is exercised on the canonical training path. Same discipline as Liquid before its
12-seed ablation: presence in code is not evidence of value. Flagged 2026-07-13; no scaffold or
run scheduled — add one when training hardware is available to spend on this specific question.
