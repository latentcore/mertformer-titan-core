# White Paper: LiquidRouter Architecture
**Temporal-Aware Routing for Sparse Mixture of Experts (Truth-Locked Build30 V2)**

## 1. Abstract
MertFormer Titan uses **LiquidRouter** as the MoE gating path for temporal token routing. In the current implementation, LiquidRouter is a **causal depthwise Conv1d + rolling state buffer** mechanism in `layers/moe.py`. This should be read separately from the CfC path, which is implemented in `layers/liquid.py` (`LiquidMixer/LiquidCell`).

## 2. The Problem: Stateless MoE Instability
Traditional MoE routers can over-concentrate traffic into a few experts and oscillate between experts across adjacent tokens. This can increase routing variance and degrade practical edge efficiency.

## 3. Current Build30 V2 Routing Mechanism
LiquidRouter currently combines two signals:
- **Main path:** token-local projection for expert logits.
- **Fluid path:** causal depthwise Conv1d over a short token history window (`history_window`) with a rolling runtime state.

Routing is executed as **token-choice top-k** in MoE dispatch.

**V2 note:** MoE dispatch now supports a parallel gather/scatter path (optionally enabled) for higher throughput.

### Mathematical Sketch (Implementation-Aligned)
Let token features be `x_t`.
- Main logits: `g_main(t) = W_main * x_t`
- Fluid logits: `g_fluid(t) = W_fluid * Conv1d_causal(x_{t-k+1:t})`
- Router output: `g(t) = g_main(t) + g_fluid(t)`
- Selection: token-choice `top-k(g(t))`

## 4. CfC Separation (Important)
CfC dynamics are present in the architecture, but not as the router kernel:
- **Router:** Conv1d + state buffer (`layers/moe.py`)
- **CfC path:** `LiquidMixer/LiquidCell` (`layers/liquid.py`)

This separation is intentional in Build30 V2 and must be reflected in partner-facing claims.

## 5. Hardware Intent and Claim Boundary
- Target intent: reduce unstable routing transitions and improve edge-runtime behavior.
- Any latency/energy superiority remains **target/estimate** until measured on real devices.
- No precedence or superiority claim is made without independent evidence.

## 6. Conclusion
LiquidRouter is a temporal Conv routing component within MertFormer Titan’s sparse MoE stack. Build30 V2 documents this as an implementation-ready, claim-safe mechanism aligned with offline-first edge constraints.

## 7. Experiment (Measured Ablation) and Limits

**Claim mode: measured (small-scale, directional). This is NOT a benchmark claim.**
**Canonical results: [`ABLATION.md`](ABLATION.md).**

**Important scope distinction:** the ablation toggles `cfg.use_liquid` — i.e. the **CfC LiquidMixer**
in `layers/liquid.py`. The **Conv1d LiquidRouter** this paper describes (`layers/moe.py`) and the MoE
stack stay ON in both arms, so the signal measures the CfC mixer; it does **not** directly test the
router.

**12-seed final (2026-06-15) — SUPERSEDES the earlier single-seed pilot.** 12 seeds × ON/OFF, small
preset (hidden 384, 8 layers), 2-digit addition, Kaggle T4, ~900 s/run:
- Held-out ID exact-acc: OFF 96.32% vs ON 94.69% (Δ −1.63 pp, p=0.305, d=−0.43, 95% CI [−4.63, +1.17]).
- OOD (3-digit): 0% / 0% (floor — non-discriminative). ON ~30% fewer steps per wall-clock; +2.8% params.
- **Verdict:** no visible accuracy benefit, cost certain (~30% slower), but the task is ceiling/floor-
  saturated and the test is underpowered + iso-time → **inconclusive on Liquid's value at scale, not
  disproven.** No positive evidence to bind the 45K run to Liquid.

**Liquid speed/latency: NO CLAIM** until a verified 45K run produces scale-representative data (pilot/
H200 numbers are confounded). See `ABLATION.md`.

**2026-07-31 addendum:** an external, independent test (different hardware, small scale — see
`ABLATION.md`'s "External signal" note and `BACKLOG.md`) suggests the ~30% figure above may
understate the CfC mixer's cost at the canonical `seq_len=4096`, since its sequential recurrence
scales with sequence length in a way attention does not. Informational only; does not change the
claim boundary above.

**Limits (honest):** small toy task, T4, ~15 min/seed — not proof about the 3.67B model. The earlier
single-seed pilot (Δ(off−on)=+0.50) was largely one lucky seed and is superseded. **No arXiv submission
on toy-scale data — skeleton now, submission only after a measured 45K run.**

Evidence (sha256-chained): [`reports/ablations/liquid_ablation_final_20260615/`](reports/ablations/liquid_ablation_final_20260615/)
(`final_summary.json`, `MANIFEST.json`, plots). The 2026-06-14 single-seed pilot
(`reports/ablations/liquid_ablation_results.json`, `liquid_ablation_pilot_curve.png`) is retained as
**superseded** history.


## 8. arXiv Submission (Post-45K)

**Proposed title:** "LiquidRouter: Temporal-Aware Expert Routing for Sparse MoE via Causal Depthwise Convolution"

**Target category:** cs.LG (primary), cs.CL (secondary)

**Status:** Skeleton ready. Submission blocked on 45K checkpoint + benchmark results. Fill sections 3-7 with measured data before submitting.
