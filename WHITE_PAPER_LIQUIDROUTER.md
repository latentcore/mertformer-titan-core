# White Paper: LiquidRouter Architecture
**Temporal-Aware Routing for Sparse Mixture of Experts (Truth-Locked Build30)**

## 1. Abstract
MertFormer Titan uses **LiquidRouter** as the MoE gating path for temporal token routing. In the current implementation, LiquidRouter is a **causal depthwise Conv1d + rolling state buffer** mechanism in `layers/moe.py`. This should be read separately from the CfC path, which is implemented in `layers/liquid.py` (`LiquidMixer/LiquidCell`).

## 2. The Problem: Stateless MoE Instability
Traditional MoE routers can over-concentrate traffic into a few experts and oscillate between experts across adjacent tokens. This can increase routing variance and degrade practical edge efficiency.

## 3. Current Build30 Routing Mechanism
LiquidRouter currently combines two signals:
- **Main path:** token-local projection for expert logits.
- **Fluid path:** causal depthwise Conv1d over a short token history window (`history_window`) with a rolling runtime state.

Routing is executed as **token-choice top-k** in MoE dispatch.

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

This separation is intentional in Build30 and must be reflected in partner-facing claims.

## 5. Hardware Intent and Claim Boundary
- Target intent: reduce unstable routing transitions and improve edge-runtime behavior.
- Any latency/energy superiority remains **target/estimate** until measured on real devices.
- No precedence or superiority claim is made without independent evidence.

## 6. Conclusion
LiquidRouter is a temporal Conv routing component within MertFormer Titan’s sparse MoE stack. Build30 documents this as an implementation-ready, claim-safe mechanism aligned with offline-first edge constraints.
