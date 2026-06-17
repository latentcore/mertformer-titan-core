"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert"

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg
from layers.bitlinear import BitLinear, activation_quant, weight_quant

_MLA_KV_PACK_ENABLED = os.environ.get("TITAN_MLA_KV_PACK", "0") == "1"


def set_mla_kv_pack_enabled(enabled: bool) -> None:
    """Runtime toggle for MLA K+V packed projection."""
    global _MLA_KV_PACK_ENABLED
    _MLA_KV_PACK_ENABLED = bool(enabled)


def _mla_packed_kv(
    x: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    kv_out_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute K and V projections with one packed BitLinear fallback matmul."""
    kv_weight = torch.cat([k_weight, v_weight], dim=0)
    x_q = activation_quant(x)
    w_q = weight_quant(kv_weight)
    kv_out = F.linear(x_q, w_q, None)
    return kv_out.split(kv_out_dim, dim=-1)

# Flash Attention 2 (Optional, for 20-40% speedup on A100/H100)
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    if os.environ.get("TITAN_VERBOSE", "0") == "1":
        print("⚠️  TR: Flash Attention 2 mevcut değil. / EN: Flash Attention 2 not available.")


def _is_onnx_export() -> bool:
    """Return True when running under torch.onnx export trace."""
    onnx_mod = getattr(torch, "onnx", None)
    if onnx_mod is None:
        return False
    check_fn = getattr(onnx_mod, "is_in_onnx_export", None)
    if check_fn is None:
        return False
    try:
        return bool(check_fn())
    except Exception:
        return False



class _QKRMSNorm(nn.Module):
    """
    Lightweight RMSNorm for QK Normalization - attention stability.

    NOTE: nn.RMSNorm requires PyTorch 2.4+; this class is backward compatible.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [V26.5 FIX] Compute RMS in FP32 for stability
        # x.float() -> pow(2) -> mean -> rsqrt
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        # Scale back to original dtype
        return x * rms.to(x.dtype) * self.weight.to(x.dtype)



class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - Caching Optimized.
    """

    # Buffer type declarations (registered in __init__ via register_buffer).
    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 500000.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute inv_freq
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Register once; update in-place / via _buffers writes (no re-register)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        
        # Initial Cache Population
        init_device = device if device is not None else inv_freq.device
        self._update_cache(max_seq_len, init_device)

    @torch.no_grad()
    def _update_cache(self, seq_len: int, device: Optional[torch.device]) -> None:
        seq_len = int(seq_len)
        if device is None:
            device = self.inv_freq.device
        self.max_seq_len = max(seq_len, self.max_seq_len)
        t = torch.arange(self.max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        # Buffer-safe cache writes: no re-register, no buffer replacement.
        # Expect model/device placement to be handled externally via model.to(...).
        if self.cos_cached.device != cos.device or self.cos_cached.dtype != cos.dtype:
            raise RuntimeError(
                "RoPE cache device/dtype mismatch. Move the full model with model.to(...). "
                f"cache={self.cos_cached.device}/{self.cos_cached.dtype}, "
                f"target={cos.device}/{cos.dtype}"
            )
        if self.sin_cached.device != sin.device or self.sin_cached.dtype != sin.dtype:
            raise RuntimeError(
                "RoPE cache device/dtype mismatch. Move the full model with model.to(...). "
                f"cache={self.sin_cached.device}/{self.sin_cached.dtype}, "
                f"target={sin.device}/{sin.dtype}"
            )

        self.cos_cached.resize_(cos.shape)
        self.cos_cached.copy_(cos)
        self.sin_cached.resize_(sin.shape)
        self.sin_cached.copy_(sin)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[2]
            
        total_len = seq_len + offset
        
        if (
            self.cos_cached.numel() == 0
            or total_len > self.cos_cached.shape[2]
            or self.cos_cached.device != x.device
        ):
            self._update_cache(total_len, x.device)
            
        return (
            self.cos_cached[..., offset:total_len, :].to(dtype=x.dtype), # Slice seq dimension
            self.sin_cached[..., offset:total_len, :].to(dtype=x.dtype)
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized RoPE application using pre-computed cos/sin.
    """
    # q, k: [B, H, T, D]
    # cos, sin: [1, 1, T, D]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



class GQA(nn.Module):
    """
    Grouped-Query Attention (GQA) - LLaMA-3 compatible attention mechanism.

    Implemented as GQA: a reduced set of KV heads (``num_kv_heads``) is
    projected and then replicated to the query heads at runtime. A true
    latent-MLA (low-rank KV bottleneck) is intentionally NOT implemented.
    Formerly exported as ``MLA``; renamed to match the implementation
    (see DECISIONS.md: "MLA -> GQA rename").

    Features:
    - LLaMA-3 interleaved RoPE
    - Optional decoupled RoPE
    - FP32-stable Softmax
    - BitLinear projections
    - Causal mask buffer optimization
    - KV Cache support (inference speedup) [V21.0]
    """

    def __init__(self) -> None:
        """GQA initializer."""
        super().__init__()
        # Config-driven parameters
        self.hidden_size = int(cfg.hidden_size)
        self.num_heads = int(cfg.num_heads)
        self.head_dim = getattr(cfg, "head_dim", self.hidden_size // self.num_heads)
        self.num_kv_heads = getattr(cfg, "num_kv_heads", self.num_heads)

        # Guard against invalid GQA configs that would silently produce zero KV heads.
        # Requirement: 1 <= num_kv_heads <= num_heads and num_heads % num_kv_heads == 0.
        if int(self.num_kv_heads) <= 0:
            raise ValueError(f"num_kv_heads must be >= 1, got {self.num_kv_heads}")
        if int(self.num_kv_heads) > int(self.num_heads):
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) must be <= num_heads ({self.num_heads}) for GQA."
            )
        if int(self.num_heads) % int(self.num_kv_heads) != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads}) for GQA."
            )
        
        # RoPE settings
        self.rope_dim = getattr(cfg, "rope_dim", None) # Default to full head_dim
        self.rope_base = getattr(cfg, "rope_base", 100000.0)

        rope_dim_eff = self.head_dim if self.rope_dim is None else int(self.rope_dim)
        if rope_dim_eff <= 0 or rope_dim_eff > self.head_dim:
            raise ValueError(
                f"rope_dim must be in (0, head_dim], got rope_dim={rope_dim_eff}, head_dim={self.head_dim}"
            )
        if rope_dim_eff % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim_eff}")
        self._rope_dim_eff = rope_dim_eff

        # Cached rotary embeddings (optimized)
        self.rotary_emb = RotaryEmbedding(
            dim=self._rope_dim_eff,
            max_seq_len=getattr(cfg, "max_seq_len", 8192),
            base=self.rope_base
        )
        
        # Projections (BitLinear for efficiency)
        self.q_proj = BitLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # [V27.1 FIX] Correct KV Head Projection for GQA
        self.k_proj = BitLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = BitLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = BitLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # QK Norm
        self.q_norm = _QKRMSNorm(self.head_dim)
        self.k_norm = _QKRMSNorm(self.head_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(getattr(cfg, "attention_dropout", 0.0))
        self.use_flash_attn_inference = bool(getattr(cfg, "use_flash_attn_inference", False))

        # Max sequence guard
        self.max_seq = int(getattr(cfg, "max_seq_len", 8192))
        self.use_hierarchical_kv_cache = bool(getattr(cfg, "use_hierarchical_kv_cache", False))
        self.hkv_short_window = int(getattr(cfg, "hkv_short_window", 512))
        self.hkv_long_stride = int(getattr(cfg, "hkv_long_stride", 8))
        self.hkv_max_long_blocks = int(getattr(cfg, "hkv_max_long_blocks", 128))

    def _pool_long_kv(self, tensor: torch.Tensor, stride: int, max_blocks: int) -> torch.Tensor:
        """
        Downsample long-context KV using chunk mean pooling.
        tensor: [B, Hk, S, D]
        """
        bsz, hk, slen, d = tensor.shape
        if slen == 0:
            return tensor.new_zeros((bsz, hk, 0, d))

        stride = max(1, int(stride))
        blocks = slen // stride
        pooled = tensor.new_zeros((bsz, hk, 0, d))

        if blocks > 0:
            trimmed = tensor[:, :, : blocks * stride, :]
            pooled = trimmed.reshape(bsz, hk, blocks, stride, d).mean(dim=3)

        rem = slen - blocks * stride
        if rem > 0:
            rem_chunk = tensor[:, :, blocks * stride :, :].mean(dim=2, keepdim=True)
            pooled = torch.cat([pooled, rem_chunk], dim=2)

        if max_blocks > 0 and pooled.size(2) > max_blocks:
            pooled = pooled[:, :, -max_blocks:, :]
        return pooled

    def _build_hierarchical_kv(self, k_full: torch.Tensor, v_full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build short/long context split for decode-time KV attention.
        Returns compressed-long + short tail.
        """
        total_len = k_full.size(2)
        short_window = max(1, min(self.hkv_short_window, total_len))
        long_len = total_len - short_window

        if long_len <= 0:
            return k_full, v_full

        k_long = k_full[:, :, :long_len, :]
        v_long = v_full[:, :, :long_len, :]
        k_short = k_full[:, :, long_len:, :]
        v_short = v_full[:, :, long_len:, :]

        k_long_pooled = self._pool_long_kv(k_long, self.hkv_long_stride, self.hkv_max_long_blocks)
        v_long_pooled = self._pool_long_kv(v_long, self.hkv_long_stride, self.hkv_max_long_blocks)

        k_ctx = torch.cat([k_long_pooled, k_short], dim=2)
        v_ctx = torch.cat([v_long_pooled, v_short], dim=2)
        return k_ctx, v_ctx

    def forward(
        self,
        x: torch.Tensor,
        decoupled_rope: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass - Sequence processing with multi-head attention.
        """
        B, T, C = x.shape

        # Projection: compute Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if _MLA_KV_PACK_ENABLED and os.environ.get("MERTFORMER_LOWBIT_KERNEL", "0") != "1":
            kv_out_dim = self.num_kv_heads * self.head_dim
            k_flat, v_flat = _mla_packed_kv(
                x,
                self.k_proj.weight,
                self.v_proj.weight,
                kv_out_dim,
            )
            k = k_flat.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v_flat.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        else:
            k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # V23.0: QK Normalization (attention stabilization)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # KV Cache Logic & RoPE Offset
        kv_seq_len = T
        if past_key_value is not None:
            past_k, past_v = past_key_value
            # past_k shape: [B, H, T_past, D]
            kv_seq_len = past_k.shape[2] + T

        # [V26.5 SAFEGUARD] KV Sequence Length Guard
        if kv_seq_len > self.max_seq:
             raise ValueError(
                 f"kv_seq_len ({kv_seq_len}) exceeds max_seq ({self.max_seq}). "
                 "Increase cfg.max_seq_len or use sliding window attention."
             )

        # Optimized cached RoPE application
        # 1. Get cached cos/sin for current positions
        cos, sin = self.rotary_emb(q, seq_len=T, offset=kv_seq_len - T)
        
        # 2. Apply RoPE (Decoupled vs Interleaved logic handled in helper or simplified here)
        # For simplicity in this 'Grandmaster' fix, we use standard interleaved application logic suited for LLaMA
        # If decoupled is requested, we strictly split dimensions.
        
        rope_dim = self._rope_dim_eff
        if decoupled_rope:
            # Decoupled mode: apply RoPE on trailing rope_dim subspace.
            q_rope = q[..., -rope_dim:]
            k_rope = k[..., -rope_dim:]
            q_rope, k_rope = apply_rope_optimized(q_rope, k_rope, cos, sin)
            if rope_dim < self.head_dim:
                q = torch.cat([q[..., :-rope_dim], q_rope], dim=-1)
                k = torch.cat([k[..., :-rope_dim], k_rope], dim=-1)
            else:
                q, k = q_rope, k_rope
        else:
            # Standard mode: apply RoPE on leading rope_dim subspace.
            if rope_dim < self.head_dim:
                q_rope = q[..., :rope_dim]
                k_rope = k[..., :rope_dim]
                q_rope, k_rope = apply_rope_optimized(q_rope, k_rope, cos, sin)
                q = torch.cat([q_rope, q[..., rope_dim:]], dim=-1)
                k = torch.cat([k_rope, k[..., rope_dim:]], dim=-1)
            else:
                q, k = apply_rope_optimized(q, k, cos, sin)

        # KV Cache concatenation
        k_full, v_full = k, v
        if past_key_value is not None:
            k_full = torch.cat([past_k, k], dim=2)
            v_full = torch.cat([past_v, v], dim=2)

        # Update the cache
        present_key_value = (k_full, v_full) if use_cache else None

        # Optional hierarchical short/long split for decode-time attention.
        # We keep full KV for cache persistence; only attention view is compressed.
        if self.use_hierarchical_kv_cache and past_key_value is not None and T == 1:
            k, v = self._build_hierarchical_kv(k_full, v_full)
        else:
            k, v = k_full, v_full

        # [V27.1 FIX] GQA Broadcasting (Repeat KV heads to match Q heads)
        if self.num_kv_heads != self.num_heads:
             # Shape: [B, num_kv_heads, T, D] -> [B, num_heads, T, D]
             # We repeat each KV head (num_heads // num_kv_heads) times
             n_rep = self.num_heads // self.num_kv_heads
             k = k.repeat_interleave(n_rep, dim=1)
             v = v.repeat_interleave(n_rep, dim=1)

        # -------------------------------------------------------------------------
        # FLASH ATTENTION 2
        # -------------------------------------------------------------------------
        use_flash = (
            FLASH_ATTN_AVAILABLE
            and q.is_cuda
            and past_key_value is None
            and (self.training or self.use_flash_attn_inference)
            and not _is_onnx_export()
        )
        if use_flash:
            q_flash = q.transpose(1, 2).contiguous()  # [B, T, H, D]
            k_flash = k.transpose(1, 2).contiguous()
            v_flash = v.transpose(1, 2).contiguous()
            
            out = flash_attn_func(
                q_flash,
                k_flash,
                v_flash,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                causal=True,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
            )
            out = out.transpose(1, 2)
        else:
            # Prefer SDPA path to avoid explicit full masks in common cases.
            # Keep ONNX export on matmul path (opset12 cannot export SDPA op).
            if hasattr(F, "scaled_dot_product_attention") and not _is_onnx_export():
                dropout_p = self.attn_dropout.p if self.training else 0.0
                if past_key_value is None:
                    # Mask-free causal path.
                    out = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=None,
                        dropout_p=dropout_p,
                        is_causal=True,
                    )
                elif T == 1:
                    # Decode step with past: single query can attend to all keys.
                    out = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=None,
                        dropout_p=dropout_p,
                        is_causal=False,
                    )
                else:
                    # Offset-aware causal mask for prefill+cache multi-token chunks.
                    q_pos = torch.arange(kv_seq_len - T, kv_seq_len, device=x.device)
                    k_pos = torch.arange(kv_seq_len, device=x.device)
                    causal_mask = q_pos[:, None] >= k_pos[None, :]
                    out = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=causal_mask,
                        dropout_p=dropout_p,
                        is_causal=False,
                    )
            else:
                # Legacy fallback: keep explicit masking only for bounded seq.
                if kv_seq_len > 2048:
                    raise RuntimeError(
                        "High-sequence attention fallback requires PyTorch SDPA or FlashAttention."
                    )
                q_pos = torch.arange(kv_seq_len - T, kv_seq_len, device=x.device)
                k_pos = torch.arange(kv_seq_len, device=x.device)
                causal_mask = q_pos[:, None] >= k_pos[None, :]

                scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
                scores = scores.float()  # Upcast to FP32 for softmax stability
                scores.masked_fill_(~causal_mask, float("-inf"))
                attn_weights = F.softmax(scores, dim=-1).to(x.dtype)
                attn_weights = self.attn_dropout(attn_weights)
                out = torch.matmul(attn_weights, v)

        # Output projection
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out), present_key_value
