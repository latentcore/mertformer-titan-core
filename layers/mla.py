"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 27) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD27"
__author__ = "Mert"

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg
from layers.bitlinear import BitLinear

# TR: Flash Attention 2 (Opsiyonel, A100/H100'de %20-40 hız artışı için)
# EN: Flash Attention 2 (Optional, for 20-40% speedup on A100/H100)
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("⚠️  TR: Flash Attention 2 mevcut değil. / EN: Flash Attention 2 not available.")



class _QKRMSNorm(nn.Module):
    """
    TR: QK Normalization için hafif RMSNorm - attention stabilizasyonu.
    EN: Lightweight RMSNorm for QK Normalization - attention stability.
    
    NOT: nn.RMSNorm PyTorch 2.4+ gerektirir, bu sınıf geriye uyumludur.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TR: [V26.5 FIX] Stabilite için FP32'de RMS hesapla
        # EN: [V26.5 FIX] Compute RMS in FP32 for stability
        # TR: x.float() -> pow(2) -> mean -> rsqrt
        # EN: x.float() -> pow(2) -> mean -> rsqrt
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        # TR: Orijinal dtype'a geri ölçekle / EN: Scale back to original dtype
        return x * rms.to(x.dtype) * self.weight.to(x.dtype)



class RotaryEmbedding(nn.Module):
    """
    TR: Rotary Position Embedding (RoPE) - Caching Optimazed.
    EN: Rotary Position Embedding (RoPE) - Caching Optimized.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 500000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute inv_freq
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        
        # Initial Cache Population
        self._update_cache(max_seq_len, device)

    def _update_cache(self, seq_len: int, device):
        self.max_seq_len = max(seq_len, self.max_seq_len)
        t = torch.arange(self.max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[2]
            
        total_len = seq_len + offset
        
        if total_len > self.cos_cached.shape[2]:
            self._update_cache(total_len, x.device)
            
        return (
            self.cos_cached[..., offset:total_len, :].to(dtype=x.dtype), # Slice seq dimension
            self.sin_cached[..., offset:total_len, :].to(dtype=x.dtype)
        )

def rotate_half(x):
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
    TR: Optimized RoPE application using pre-computed cos/sin.
    EN: Optimized RoPE application using pre-computed cos/sin.
    """
    # q, k: [B, H, T, D]
    # cos, sin: [1, 1, T, D]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



class MLA(nn.Module):
    """
    TR: Multi-Head Latent Attention - LLaMA-3 uyumlu attention mekanizması.
    EN: Multi-Head Latent Attention - LLaMA-3 compatible attention mechanism.

    Özellikler / Features:
    - LLaMA-3 interleaved RoPE
    - Opsiyonel decoupled RoPE
    - FP32 stabil Softmax
    - BitLinear projeksiyonlar
    - Causal mask buffer optimizasyonu
    - KV Cache desteği (inference hızlandırma) [V21.0]
    """

    def __init__(self) -> None:
        """TR: MLA başlatıcı. EN: MLA initializer."""
        super().__init__()
        # Config-driven parametreler
        self.hidden_size = int(cfg.hidden_size)
        self.num_heads = int(cfg.num_heads)
        self.head_dim = getattr(cfg, "head_dim", self.hidden_size // self.num_heads)
        self.num_kv_heads = getattr(cfg, "num_kv_heads", self.num_heads)
        self.rope_theta = getattr(cfg, "rope_theta", 10000.0)
        
        # RoPE settings
        self.rope_dim = getattr(cfg, "rope_dim", None) # Default to full head_dim
        self.rope_base = getattr(cfg, "rope_base", 500000.0)

        # [V27.0] Cached Rotary Embeddings (Optimized)
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim if self.rope_dim is None else (self.head_dim // 2), 
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

        # Causal mask buffer (hız optimizasyonu için)
        self.max_seq = int(getattr(cfg, "max_seq_len", 8192))
        mask = torch.tril(torch.ones(self.max_seq, self.max_seq), diagonal=0).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        decoupled_rope: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        TR: İleri yayılım - Multi-head attention ile sequence processing.
        EN: Forward pass - Sequence processing with multi-head attention.
        """
        B, T, C = x.shape

        # Projeksiyon: Q, K, V hesapla
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # V23.0: QK Normalization (attention stabilizasyonu)
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

        # [V27.0] Optimized Cached RoPE Application
        # 1. Get cached cos/sin for current positions
        cos, sin = self.rotary_emb(v, seq_len=T, offset=kv_seq_len - T)
        
        # 2. Apply RoPE (Decoupled vs Interleaved logic handled in helper or simplified here)
        # For simplicity in this 'Grandmaster' fix, we use standard interleaved application logic suited for LLaMA
        # If decoupled is requested, we strictly split dimensions.
        
        if decoupled_rope:
             # Decoupled mode specific handling (split last dim)
             rope_dim = self.rope_dim
             q_cont, q_rope = q[..., :-rope_dim], q[..., -rope_dim:]
             k_cont, k_rope = k[..., :-rope_dim], k[..., -rope_dim:]
             
             q_rope, k_rope = apply_rope_optimized(q_rope, k_rope, cos, sin)
             
             q = torch.cat([q_cont, q_rope], dim=-1)
             k = torch.cat([k_cont, k_rope], dim=-1)
        else:
             # Standard RoPE (Full Dim)
             q, k = apply_rope_optimized(q, k, cos, sin)

        # KV Cache concatenation
        if past_key_value is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Cache'i güncelle
        present_key_value = (k, v) if use_cache else None

        # [V27.1 FIX] GQA Broadcasting (Repeat KV heads to match Q heads)
        if self.num_kv_heads != self.num_heads:
             # Shape: [B, num_kv_heads, T, D] -> [B, num_heads, T, D]
             # We repeat each KV head (num_heads // num_kv_heads) times
             n_rep = self.num_heads // self.num_kv_heads
             k = k.repeat_interleave(n_rep, dim=1)
             v = v.repeat_interleave(n_rep, dim=1)

        # Causal mask calculation
        # Attention Mask: [1, 1, T, S] -> Broadcast to [B, H, T, S]
        # Query length: T, Key length: kv_seq_len (past + current)
        
        # Select relevant mask part
        # Rows: kv_seq_len - T to kv_seq_len (current query positions)
        # Cols: 0 to kv_seq_len (all keys)
        causal_mask = self.causal_mask[kv_seq_len - T:kv_seq_len, :kv_seq_len]
        
        # -------------------------------------------------------------------------
        # FLASH ATTENTION 2
        # -------------------------------------------------------------------------
        if FLASH_ATTN_AVAILABLE and self.training and past_key_value is None:
            q_flash = q.transpose(1, 2)  # [B, T, H, D]
            k_flash = k.transpose(1, 2)
            v_flash = v.transpose(1, 2)
            
            out = flash_attn_func(
                q_flash, k_flash, v_flash,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                causal=True,
                softmax_scale=1.0 / math.sqrt(self.head_dim)
            )
            out = out.transpose(1, 2)
        else:
            # Standard Scaled Dot Product Attention
            # [V27.0] Score calculation
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            scores = scores.float() # Upcast to FP32 for softmax stability
            
            # Masking
            scores.masked_fill_(~causal_mask, float("-inf"))
            
            attn_weights = F.softmax(scores, dim=-1).to(x.dtype) # Downcast back after softmax
            attn_weights = self.attn_dropout(attn_weights)
            
            out = torch.matmul(attn_weights, v)

        # Output projeksiyonu
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out), present_key_value
