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

## 7. Experiment (Pilot Signal) and Limits

**Claim mode: measured (pilot signal only). This is NOT a benchmark claim.**

**Important scope distinction:** The $0 pilot below toggles the `cfg.use_liquid` flag — i.e. the
**CfC LiquidMixer** in `layers/liquid.py` (layers [2,4,6]). The **Conv1d LiquidRouter** this paper
describes (`layers/moe.py`) and the MoE stack stay ON in both arms. So this signal measures the CfC
mixer; it does not directly test the router.

- Setup: ~100M proxy MertFormer, pure next-token CE (no teacher, no KD), Kaggle T4×2, $0.
- Identical data + identical init (seed 1234); the only difference is `use_liquid`. 500 steps, seq 256, batch 8.
- Result: liquid ON mean_last10 = 11.489 vs OFF = 11.993 → **Δ(off−on) = +0.50** (the CfC mixer
  directionally helps; lower loss).

**Limits (honest):** Single seed; the gap (0.50) sits inside the step-to-step noise of the constant-lr,
no-warmup curves — a direction signal, not a measured effect size. Tiny corpus (35,634 tokens / 128k
vocab); end-loss hovers near the random baseline (ln 128000 ≈ 11.76). This proves the ablation pipeline
works and points a direction; it is NOT evidence of trained capability, a benchmark, or that the model
"works". That needs a larger, multi-seed, measured run (the 45K architecture-validation run). **No arXiv
submission on a single-seed pilot — skeleton now, submission after the measured 45K run.**

Evidence: `reports/ablations/liquid_ablation_results.json` (full 500-step curves),
`reports/ablations/liquid_ablation_pilot_curve.png` (plot),
`reports/outreach/liquid_ablation_pilot_note_2026-06-15.md` (public note).
