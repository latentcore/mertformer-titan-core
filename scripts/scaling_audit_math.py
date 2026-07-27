"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

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

    # 4 Projections * (In * Out)
    attn_per_layer = 4 * (hidden_size * (num_heads * head_dim))

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
    router_params = hidden_size * num_experts
    shared_expert_params = moe_ffn_per_layer + 1  # +1 = the scalar shared_gate param
    moe_per_layer = (moe_ffn_per_layer * num_experts) + shared_expert_params + router_params

    # Layer Breakdown
    if cfg.use_moe and moe_every_n > 0:
        moe_count = num_layers // moe_every_n
        dense_count = num_layers - moe_count
    else:
        moe_count = 0
        dense_count = num_layers

    print(f"   ► Layers: {num_layers} (Dense: {dense_count}, MoE: {moe_count})")

    # Total Calculation
    total_params = embedding_params
    total_params += num_layers * attn_per_layer
    total_params += dense_count * dense_ffn_per_layer
    total_params += moe_count * moe_per_layer
    total_params += num_layers * (2 * hidden_size)     # Norms
    total_params += hidden_size                        # Final Norm

    # Active Params (Inference Cost / Compute Cost)
    # Active per MoE layer = top-k routed experts + the always-active shared expert + router.
    active_moe_cost = (moe_ffn_per_layer * active_experts) + shared_expert_params + router_params

    active_params = embedding_params
    active_params += num_layers * attn_per_layer
    active_params += dense_count * dense_ffn_per_layer
    active_params += moe_count * active_moe_cost
    active_params += num_layers * (2 * hidden_size) + hidden_size

    print(f"\n--- TITAN SCALING AUDIT (DYNAMIC) ---")
    print(f"Embedding Params: {embedding_params / 1e6:.2f} M")
    print(f"Dense Layers Params: {(dense_count * (attn_per_layer + dense_ffn_per_layer)) / 1e6:.2f} M")
    if moe_count > 0:
        print(f"MoE Layers Params: {(moe_count * (attn_per_layer + moe_per_layer)) / 1e6:.2f} M")
    
    print(f"\n🧠 TOTAL PARAMETERS (VRAM Size): {total_params / 1e6:.2f} M (~{total_params / 1e9:.3f} B)")
    print(f"⚡ ACTIVE PARAMETERS (Compute Speed): {active_params / 1e6:.2f} M (~{active_params / 1e9:.3f} B)")
    
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
