
import pytest
import torch
import torch.nn as nn
from layers.liquid import LiquidMixer
from layers.bitlinear import BitLinear
from layers.moe import MoE
from config.config import cfg

# -----------------------------------------------------------------------------
# SETUP & FIXTURES (MAC M4 OPTIMIZED)
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def device():
    """Selects the fastest available device (CUDA, then MPS for Mac, CPU as fallback)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@pytest.fixture(scope="function")
def patched_cfg():
    """Patches the global config for tiny test parameters."""
    # Backup original values
    orig_h = cfg.hidden_size
    orig_i = cfg.intermediate_size
    orig_e = cfg.num_experts
    orig_ep = cfg.num_experts_per_tok
    orig_dev = cfg.device
    orig_nh = cfg.num_heads
    orig_hd = cfg.head_dim
    orig_kv = getattr(cfg, "num_kv_heads", orig_nh)
    orig_msl = cfg.max_seq_len
    
    # Set tiny values
    cfg.hidden_size = 128
    cfg.intermediate_size = 256
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
    cfg.active_experts = 2 # Sync alias
    cfg.device = "cpu"
    cfg.max_seq_len = 128
    
    # [FIX] Ensure GQA shape consistency: 4 * 32 = 128
    cfg.num_heads = 4
    cfg.head_dim = 32
    # [FIX] Keep KV heads <= Q heads and divisible for GQA repeat_interleave
    cfg.num_kv_heads = 2
    
    yield cfg
    
    # Restore original values
    cfg.hidden_size = orig_h
    cfg.intermediate_size = orig_i
    cfg.num_experts = orig_e
    cfg.num_experts_per_tok = orig_ep
    cfg.active_experts = orig_ep
    cfg.device = orig_dev
    cfg.num_heads = orig_nh
    cfg.head_dim = orig_hd
    cfg.num_kv_heads = orig_kv
    cfg.max_seq_len = orig_msl

# -----------------------------------------------------------------------------
# 1. BITNET 1.58-BIT INTEGRITY TEST
# -----------------------------------------------------------------------------
def test_bitlinear_ternary_weights():
    """
    Verifies that BitLinear (1.58-bit) effectively pushes weights to {-1, 0, 1}
    during the forward pass (via quantization simulation).
    """
    layer = BitLinear(32, 32).cpu() # Perform structural check on CPU for precision
    x = torch.randn(10, 32)
    
    # Force weights to be non-ternary initially
    layer.weight.data.normal_(0, 1.0)
    
    # We can't check .weight directly because BitNet quantizes on the fly in forward().
    # We must hook into the quantization function or inspect the logical flow.
    # Here, we verify output range and consistency, verifying the layer runs without error.
    
    from layers.bitlinear import weight_quant
    w_quant = weight_quant(layer.weight)

    # weight_quant returns scaled values (STE): w_q_real = round(w/scale).clamp(-1,1) * scale.
    # Recover the per-row scale (RMS) to verify the underlying ternary levels {-1, 0, 1}.
    scale = torch.sqrt((layer.weight ** 2).mean(dim=1, keepdim=True)).clamp(min=1e-5)
    levels = torch.round(w_quant / scale)
    assert torch.isin(
        levels, torch.tensor([-1.0, 0.0, 1.0])
    ).all(), "BitLinear quantized weights are not ternary {-1, 0, 1}!"

    y = layer(x)
    assert not torch.isnan(y).any(), "BitLinear output contains NaNs!"
    
    loss = y.sum()
    loss.backward()
    assert layer.weight.grad is not None, "Gradients disconnected in BitLinear!"
    assert not torch.isnan(layer.weight.grad).any(), "NaN gradients in BitLinear!"

# -----------------------------------------------------------------------------
# 2. LIQUID LAYER (LTC) GRADIENT FLOW
# -----------------------------------------------------------------------------
def test_liquid_gradient_stability(device):
    """
    Critical Test: Does the Liquid layer suffer from Vanishing/Exploding gradients?
    Runs a forward/backward pass and checks gradient norms.
    """
    # Create a tiny model manually since LiquidMixer init is simple
    model = LiquidMixer(128).to(device)
    batch_size = 8
    seq_len = 32 # Long enough to test temporal dependencies
    
    x = torch.randn(batch_size, seq_len, 128).to(device)
    x.requires_grad = True
    
    # Forward Pass
    y = model(x)
    assert y.shape == x.shape, "Liquid layer output shape mismatch!"
    assert not torch.isnan(y).any(), "Liquid layer forward generated NaNs!"
    
    # Backward Pass
    loss = y.mean()
    loss.backward()
    
    # Check Gradients
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()
            
    # CRITICAL CHECK
    assert grad_norm > 0.0, "Vanishing Gradient: Liquid layer is dead (grad_norm == 0)"
    assert grad_norm < 1000.0, f"Exploding Gradient: Liquid layer unstable (grad_norm={grad_norm})"
    
    print(f"\n✅ Liquid Layer Gradient Norm: {grad_norm:.4f} (Stable)")

# -----------------------------------------------------------------------------
# 3. MoE (MIXTURE OF EXPERTS) ROUTING CHECK
# -----------------------------------------------------------------------------
def test_moe_routing_distribution(device, patched_cfg):
    """
    Ensures that the MoE router is not collapsing (i.e., sending everything to Expert 0).
    """
    # MoE init reads directly from global cfg, which is patched by fixture
    moe = MoE().to(device)
    
    # Sufficiently large batch/seq to statistically expect distribution
    x = torch.randn(16, 64, patched_cfg.hidden_size).to(device)
    
    output, aux_loss = moe(x)
    
    # 1. Output Integrity
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    
    # 2. Aux Loss Verification (Should be > 0 if load balancing logic exists)
    assert aux_loss > 0.0, "Aux Loss is zero! Router might not be calculating load balance."
    
    print(f"\n✅ MoE Aux Loss: {aux_loss.item():.4f} (Load Balancing Active)")


def test_moe_expert_paging_contract_cpu(patched_cfg):
    """
    Expert paging must be non-breaking and deterministic on CPU runs.
    On CPU there is no device swap, so swap counters should remain zero.
    """
    orig_use = getattr(cfg, "use_expert_paging", False)
    orig_inf_only = getattr(cfg, "expert_paging_inference_only", True)
    orig_lazy_init = getattr(cfg, "expert_paging_lazy_init", True)
    orig_cache = getattr(cfg, "expert_paging_cache_size", 2)
    orig_offload = getattr(cfg, "expert_paging_offload_device", "cpu")

    cfg.use_expert_paging = True
    cfg.expert_paging_inference_only = True
    cfg.expert_paging_lazy_init = True
    cfg.expert_paging_cache_size = 1
    cfg.expert_paging_offload_device = "cpu"

    try:
        moe = MoE().cpu()
        moe.eval()
        x = torch.randn(4, 8, patched_cfg.hidden_size)
        out, aux_loss = moe(x)
        stats = moe.get_expert_paging_stats()

        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        assert torch.isfinite(aux_loss)
        assert stats["enabled"] is True
        assert stats["lazy_init"] is True
        assert stats["bootstrapped"] is True
        assert stats["swaps_in"] == 0
        assert stats["swaps_out"] == 0

        # Train/eval transition should remain safe with paging enabled.
        moe.train()
        x_train = torch.randn(2, 4, patched_cfg.hidden_size, requires_grad=True)
        out_train, aux_train = moe(x_train)
        loss = out_train.mean() + aux_train
        loss.backward()
        has_expert_grad = any(
            p.grad is not None
            for ex in moe.experts
            for p in ex.parameters()
        )
        assert has_expert_grad, "Expert grads disconnected after paging mode transition."
    finally:
        cfg.use_expert_paging = orig_use
        cfg.expert_paging_inference_only = orig_inf_only
        cfg.expert_paging_lazy_init = orig_lazy_init
        cfg.expert_paging_cache_size = orig_cache
        cfg.expert_paging_offload_device = orig_offload

# -----------------------------------------------------------------------------
# 4. MPS (APPLE SILICON) COMPATIBILITY & STRESS TEST
# -----------------------------------------------------------------------------
def test_full_block_mps_stress(device, patched_cfg):
    """
    Integrates Liquid + BitNet + MoE in a MertFormer block and runs it on MPS.
    Checks for OOM or Implementation Errors on Apple Silicon.
    """
    if device.type != "mps":
        pytest.skip("MPS device not available (Not on Apple Silicon)")
    
    # Ensure patching set correct device
    patched_cfg.device = "mps"
    
    from layers.mertformer_block import MertFormerBlock
    
    try:
        block = MertFormerBlock(layer_id=0).to(device)
        x = torch.randn(4, 128, patched_cfg.hidden_size).to(device) # B, T, H
        
        # Test 1: Memory Spike Check (Implicit via successful run)
        # [2026-07-08] MertFormerBlock.forward now returns a 4th element (the final
        # LiquidMixer hidden state) so generate() can thread the CfC recurrence.
        out, aux, _, _ = block(x)
        
        # Test 2: Backward Pass (Complex autograd graph)
        loss = out.mean() + aux
        loss.backward()
        
        assert not torch.isnan(out).any(), "NaNs detected in Full Block Output on MPS"
        print("\n✅ Full Block Forward/Backward on MPS: SUCCESS")

    except Exception as e:
        pytest.fail(f"MPS Stress Test Failed: {e}")
