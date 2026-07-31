"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright 2026 Mert Yünlü
Licensed under the Apache License, Version 2.0 (see LICENSE).

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

from config.build_label import BUILD_LABEL as __version__
__author__ = "Mert Yünlü"

import sys
import os
from pathlib import Path

# Add project root to path to import config
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

try:
    from config.config import cfg
except ImportError:
    print("❌ Critical: Config not found!")
    sys.exit(1)

def estimate_params():
    print(f"🔧 Titan Scaling Audit: Using Dynamic Config ({cfg.model_name})")
    
    # Config from Dynamic Config
    vocab_size = cfg.vocab_size
    hidden_size = cfg.hidden_size
    num_layers = cfg.num_layers
    intermediate_size = getattr(cfg, "intermediate_size", hidden_size * 4)
    # NOTE (fixed 2026-07-27): MoE experts use a SEPARATE intermediate size from
    # dense-layer FFNs (layers/moe.py BitSwiGLU(hidden_size, moe_intermediate);
    # config/config.py's own internal param-estimator, `_estimate_total_params`,
    # already makes this distinction as `moe_ffn_per_layer` vs `dense_ffn_per_layer`
    # -- this script previously reused the dense `intermediate_size` for MoE experts
    # too, undercounting every MoE layer whenever moe_intermediate > intermediate_size
    # (true here: 8192 vs 5632).
    moe_intermediate = getattr(cfg, "moe_intermediate", intermediate_size)
    num_experts = getattr(cfg, "num_experts", 8)
    active_experts = getattr(cfg, "active_experts", 2)
    moe_every_n = getattr(cfg, "moe_every_n_layers", 0)

    # 1. Embeddings
    embedding_params = vocab_size * hidden_size

    # 2. Per Layer Attention (approx, non-MLA)
    # NOTE: label-only correction. The real attention is GQA (see layers/mla.py,
    # config.config L223); there is no latent down/up KV projection. The formula
    # below is the standard 4-projection (Q/K/V/O) MHA/GQA estimate, NOT true MLA.
    num_heads = cfg.num_heads
    head_dim = getattr(cfg, "head_dim", hidden_size // num_heads)

    # NOTE (fixed 2026-07-29): this was `4 * hidden_size * (num_heads * head_dim)` --
    # the MHA formula, which assumes K and V are as wide as Q. layers/mla.py implements
    # GQA: k_proj/v_proj are `num_kv_heads * head_dim` wide, not `num_heads * head_dim`.
    # At the canonical 16q/8kv that overcounted attention by 4,194,304 params per layer
    # (16,777,216 vs the real 12,582,912) = +75,497,472 across 18 layers, and put this
    # script in direct contradiction with config/config.py::_estimate_total_params, which
    # already did it correctly. Verified against a live GQA() instance: the expression
    # below equals the summed numel of its q/k/v/o projections exactly.
    num_kv_heads = int(getattr(cfg, "num_kv_heads", num_heads))
    q_proj = hidden_size * (num_heads * head_dim)
    kv_proj = hidden_size * (num_kv_heads * head_dim)
    o_proj = (num_heads * head_dim) * hidden_size
    attn_per_layer = q_proj + 2 * kv_proj + o_proj
    # GQA also owns two _QKRMSNorm modules (q_norm, k_norm), each a learnable weight of
    # head_dim. Small but real; previously omitted.
    attn_per_layer += 2 * head_dim

    # 3. Feed Forward (dense layers) / MoE (routed layers)
    dense_ffn_per_layer = 3 * hidden_size * intermediate_size
    moe_ffn_per_layer = 3 * hidden_size * moe_intermediate

    # MoE Layer Cost
    # NOTE (fixed 2026-07-27): layers/moe.py always instantiates one additional
    # "shared expert" (BitSwiGLU(hidden_size, moe_intermediate), unconditionally
    # added to every token's output via a learnable sigmoid gate -- see
    # MoE.__init__'s `self.shared_expert`/`self.shared_gate` and MoE.forward's
    # `shared_out = self.shared_expert(x_flat)`). It is a 9th, always-instantiated,
    # always-active expert-sized block that is NOT part of `num_experts` (the
    # routed/sparse pool) -- previously omitted entirely from both totals below.
    # NOTE (fixed 2026-07-29): the router is layers/moe.py's LiquidRouter, not a bare
    # nn.Linear. Besides main_proj (hidden x num_experts) it owns a depthwise Conv1d
    # `fluid_mixer` (groups=hidden, kernel=history_window=4 -> hidden * 4 weights,
    # bias=False) and a second projection `fluid_gate` (hidden x num_experts).
    history_window = 4  # LiquidRouter.history_window
    router_params = (
        hidden_size * num_experts          # main_proj
        + hidden_size * history_window     # fluid_mixer
        + hidden_size * num_experts        # fluid_gate
    )
    shared_expert_params = moe_ffn_per_layer + 1  # +1 = the scalar shared_gate param
    moe_per_layer = (moe_ffn_per_layer * num_experts) + shared_expert_params + router_params

    # Layer Breakdown
    if cfg.use_moe and moe_every_n > 0:
        moe_count = num_layers // moe_every_n
        dense_count = num_layers - moe_count
    else:
        moe_count = 0
        dense_count = num_layers

    # NOTE (fixed 2026-07-29): the Liquid/CfC mixers were omitted ENTIRELY from both the
    # total and the active sums -- ~50.35M params (1.37% of the model) at the canonical
    # config. Each LiquidMixer holds a LiquidCell with FOUR hidden x hidden BitLinear
    # projections (input_w, hidden_w, tau_input_w, tau_hidden_w, bias=False), a tau_bias
    # of shape (1, hidden) and an nn.LayerNorm(hidden). They are ALWAYS active (dense),
    # so they enter the active-parameter sum in full. Placement mirrors
    # layers/mertformer_block.py: explicit liquid_layers_idx wins, else the
    # liquid_every_n_layers cadence.
    use_liquid = bool(getattr(cfg, "use_liquid", False))
    liquid_layers_idx = list(getattr(cfg, "liquid_layers_idx", None) or [])
    liquid_every_n = int(getattr(cfg, "liquid_every_n_layers", 0) or 0)
    if not use_liquid:
        liquid_count = 0
    elif liquid_layers_idx:
        liquid_count = len([i for i in liquid_layers_idx if 0 <= int(i) < num_layers])
    elif liquid_every_n > 0:
        liquid_count = num_layers // liquid_every_n
    else:
        liquid_count = 0
    liquid_per_layer = 4 * hidden_size * hidden_size + hidden_size + 2 * hidden_size
    liquid_total = liquid_count * liquid_per_layer

    print(f"   ► Layers: {num_layers} (Dense: {dense_count}, MoE: {moe_count}, Liquid: {liquid_count})")

    # Total Calculation
    total_params = embedding_params
    total_params += num_layers * attn_per_layer
    total_params += dense_count * dense_ffn_per_layer
    total_params += moe_count * moe_per_layer
    total_params += liquid_total                        # Liquid/CfC mixers (always dense)
    total_params += num_layers * (2 * hidden_size)     # Norms
    total_params += hidden_size                        # Final Norm

    # Active Params (Inference Cost / Compute Cost)
    # Active per MoE layer = top-k routed experts + the always-active shared expert + router.
    active_moe_cost = (moe_ffn_per_layer * active_experts) + shared_expert_params + router_params

    active_params = embedding_params
    active_params += num_layers * attn_per_layer
    active_params += dense_count * dense_ffn_per_layer
    active_params += moe_count * active_moe_cost
    active_params += liquid_total  # Liquid mixers are dense: fully active every token
    active_params += num_layers * (2 * hidden_size) + hidden_size

    print(f"\n--- TITAN SCALING AUDIT (DYNAMIC) ---")
    print(f"Embedding Params: {embedding_params / 1e6:.2f} M")
    print(f"Dense Layers Params: {(dense_count * (attn_per_layer + dense_ffn_per_layer)) / 1e6:.2f} M")
    if moe_count > 0:
        print(f"MoE Layers Params: {(moe_count * (attn_per_layer + moe_per_layer)) / 1e6:.2f} M")
    
    print(f"\n🧠 TOTAL PARAMETERS (VRAM Size): {total_params / 1e6:.2f} M (~{total_params / 1e9:.3f} B)")
    print(f"⚡ ACTIVE PARAMETERS (Compute Speed): {active_params / 1e6:.2f} M (~{active_params / 1e9:.3f} B)")

    # [2026-07-29] Cross-check against config/config.py's independent estimator.
    # This repo carries TWO parameter estimators (this script computes ACTIVE params,
    # which config's does not, so they cannot simply be merged). They silently disagreed
    # for a long time -- this script used the MHA attention formula while config used the
    # correct GQA one -- and nothing detected it. This assertion makes any future
    # divergence loud instead of silent. Tolerance is 0.5%: both are analytical estimates
    # over the same fields, so they should agree almost exactly.
    try:
        from config.config import _estimate_total_params

        config_total = _estimate_total_params(cfg)
        if config_total > 0:
            drift = abs(total_params - config_total) / config_total
            status = "OK" if drift <= 0.005 else "DRIFT"
            print(
                f"🔎 Cross-check vs config._estimate_total_params: "
                f"{config_total / 1e9:.3f} B  (delta {drift * 100:.3f}%  [{status}])"
            )
            if drift > 0.005:
                print(
                    "   ⚠️  The two independent parameter estimators disagree by >0.5%. "
                    "One of them is wrong -- do NOT quote either number until resolved.",
                    file=sys.stderr,
                )
    except Exception as exc:  # noqa: BLE001 - cross-check must never break the report
        print(f"🔎 Cross-check skipped ({type(exc).__name__}: {exc})", file=sys.stderr)
    
    if moe_count > 0:
        sparsity = 1.0 - (active_params / total_params)
        print(f"📉 Sparsity Ratio: {sparsity * 100:.1f}% (Idle Params)")
    
    # Throughput & Scaling Laws
    batch_size = cfg.batch_size
    seq_len = cfg.max_seq_len
    max_steps = int(os.environ.get("TITAN_MAX_STEPS", getattr(cfg, "max_steps", 45000)))
    
    global_batch_size = batch_size 
    
    tokens_per_step = global_batch_size * seq_len
    total_tokens = tokens_per_step * max_steps
    
    print(f"\n--- TOKEN ANALYSIS ---")
    print(f"Training Steps: {max_steps}")
    print(f"Global Batch: {global_batch_size}")
    print(f"Total Tokens: {total_tokens / 1e9:.3f} B")
    
    # Chinchilla Optimal (20 tokens per param)
    chinchilla_tokens = total_params * 20
    
    print(f"Chinchilla Optimal (20x): {chinchilla_tokens / 1e9:.3f} B tokens")
    
    # Knowledge Distillation Boost
    # ASSUMPTION (UNMEASURED): a 70B teacher's soft logits are *assumed* worth ~3x
    # raw text, motivated by Hinton/DeepSeek-style distillation literature. This 3.0
    # multiplier has NOT been measured in this repo; treat it as a target/assumption,
    # not a verified result. The verdict below is therefore a target projection.
    teacher_efficiency_factor = 3.0  # unmeasured assumption (literature-motivated)
    effective_tokens = total_tokens * teacher_efficiency_factor

    print(f"\n--- DISTILLATION FACTOR (UNMEASURED ASSUMPTION) ---")
    print(f"Teacher: Llama-3.3-70B (Soft Labels)")
    print(f"Assumed Efficiency Boost: {teacher_efficiency_factor}x (unmeasured)")
    print(f"Effective (Assumed) Training Tokens: {effective_tokens / 1e9:.3f} B")

    print(f"\n--- VERDICT (TARGET PROJECTION, not measured) ---")
    if effective_tokens < chinchilla_tokens:
        print(f"⚠️  TARGET STATUS: UNDER-TRAINED (even under the assumed teacher boost).")
        print(f"   Assumed effective: {effective_tokens/1e9:.2f}B vs Optimal: {chinchilla_tokens/1e9:.2f}B.")
    else:
        ratio = effective_tokens / total_params
        print(f"🎯 TARGET STATUS: Chinchilla budget met *if* the assumed {teacher_efficiency_factor}x holds ({ratio:.1f} virtual tok/param).")
        print(f"   Raw tokens are low; this projection assumes (unverified) teacher logic fills the gap.")
        if ratio > 100:
             print("   Mode: Teacher-enhanced target analysis; boost factor and S25 readiness remain unmeasured.")
        else:
             print("   Status: Healthy pre-training baseline target (assumption-dependent).")

if __name__ == "__main__":
    estimate_params()
