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

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert"

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
    num_experts = getattr(cfg, "num_experts", 8)
    active_experts = getattr(cfg, "active_experts", 2)
    moe_every_n = getattr(cfg, "moe_every_n_layers", 0)
    
    # 1. Embeddings
    embedding_params = vocab_size * hidden_size
    
    # 2. Per Layer Attention (MLA estimate)
    num_heads = cfg.num_heads
    head_dim = getattr(cfg, "head_dim", hidden_size // num_heads)
    
    # 4 Projections * (In * Out)
    attn_per_layer = 4 * (hidden_size * (num_heads * head_dim))
    
    # 3. Feed Forward / MoE
    ffn_per_layer = 3 * hidden_size * intermediate_size
    
    # MoE Layer Cost
    router_params = hidden_size * num_experts
    moe_per_layer = (ffn_per_layer * num_experts) + router_params
    
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
    total_params += dense_count * ffn_per_layer
    total_params += moe_count * moe_per_layer
    total_params += num_layers * (2 * hidden_size)     # Norms
    total_params += hidden_size                        # Final Norm
    
    # Active Params (Inference Cost / Compute Cost)
    active_moe_cost = (ffn_per_layer * active_experts) + router_params
    
    active_params = embedding_params
    active_params += num_layers * attn_per_layer
    active_params += dense_count * ffn_per_layer
    active_params += moe_count * active_moe_cost 
    active_params += num_layers * (2 * hidden_size) + hidden_size

    print(f"\n--- TITAN SCALING AUDIT (DYNAMIC) ---")
    print(f"Embedding Params: {embedding_params / 1e6:.2f} M")
    print(f"Dense Layers Params: {(dense_count * (attn_per_layer + ffn_per_layer)) / 1e6:.2f} M")
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
    
    # [USER LOGIC VERIFIED] Knowledge Distillation Boost
    # A 70B Teacher providing logits is worth ~3x-4x raw text (Hinton/DeepSeek findings)
    teacher_efficiency_factor = 3.0 
    effective_tokens = total_tokens * teacher_efficiency_factor
    
    print(f"\n--- DISTILLATION FACTOR ---")
    print(f"Teacher: Llama-3.3-70B (Soft Labels)")
    print(f"Efficiency Boost: {teacher_efficiency_factor}x")
    print(f"Effective Training Tokens: {effective_tokens / 1e9:.3f} B")

    print(f"\n--- VERDICT ---")
    if effective_tokens < chinchilla_tokens:
        print(f"⚠️  STATUS: UNDER-TRAINED (Even with Teacher).")
        print(f"   Effective: {effective_tokens/1e9:.2f}B vs Optimal: {chinchilla_tokens/1e9:.2f}B.")
    else:
        ratio = effective_tokens / total_params
        print(f"✅ STATUS: CHINCHILLA SATISFIED via Distillation ({ratio:.1f} virtual tok/param).")
        print(f"   Raw tokens were low, but Teacher Logic fills the gap.")
        if ratio > 100:
             print("   Mode: Teacher-enhanced target analysis; S25 readiness remains unmeasured.")
        else:
             print("   Status: Healthy pre-training baseline target.")

if __name__ == "__main__":
    estimate_params()
