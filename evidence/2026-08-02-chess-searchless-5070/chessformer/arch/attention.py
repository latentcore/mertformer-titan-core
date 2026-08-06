"""Grouped-query attention with RoPE.

Mirrors ``vendor/upstream/layers/mla.py`` (``RotaryEmbedding``,
``rotate_half``, ``apply_rope_optimized``, ``GQA``).

DELIBERATE DEVIATION -- ``attention_mode``
------------------------------------------
The canonical module is a *language model* attention: it always masks causally
(``mla.py:475`` ``is_causal=True``). ``scripts/chess_5080_onefile.py:2533``
copied that verbatim onto a chess board while its own model card
(``:7451``) claims "Board attention is intentionally non-causal: the model sees
the whole board state at once". The code contradicted the claim: with a causal
mask over the 12 meta + 64 square tokens, square a1 could attend only to the
meta tokens and never to the rest of the board.

A board is a *state*, not a sequence -- its tokens have no temporal order, so
there is nothing to mask. ``attention_mode="bidirectional"`` (the default here)
lets every square see every other square. ``attention_mode="causal"`` keeps the
canonical LM behaviour bit-for-bit and is what the parity test compares against.

The KV-cache / hierarchical-KV / flash-attention decode paths from the canonical
file are intentionally not carried over: this model does a single fixed-length
encoder pass per position and never decodes autoregressively, so those branches
would be unreachable code.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .bitlinear import make_linear
from .norm import QKRMSNorm


class RotaryEmbedding(nn.Module):
    """Cached rotary position embeddings (mirrors ``mla.RotaryEmbedding``)."""

    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)
        self.base = float(base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(self.max_seq_len, inv_freq.device)

    @torch.no_grad()
    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        self.max_seq_len = max(int(seq_len), self.max_seq_len)
        t = torch.arange(self.max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._buffers["cos_cached"] = emb.cos()[None, None, :, :]
        self._buffers["sin_cached"] = emb.sin()[None, None, :, :]

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[2]
        if (
            self.cos_cached.numel() == 0
            or seq_len > self.cos_cached.shape[2]
            or self.cos_cached.device != x.device
        ):
            self._build_cache(seq_len, x.device)
        return (
            self.cos_cached[..., :seq_len, :].to(dtype=x.dtype),
            self.sin_cached[..., :seq_len, :].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_optimized(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GQA(nn.Module):
    """Grouped-query attention. KV heads are projected once and replicated to Q heads."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = int(cfg.hidden_size)
        self.num_heads = int(cfg.num_heads)
        self.head_dim = cfg.resolved_head_dim()
        self.num_kv_heads = int(cfg.num_kv_heads)
        self.causal = cfg.attention_mode == "causal"
        self.use_rope = bool(cfg.use_rope)

        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be >= 1, got {self.num_kv_heads}")
        if self.num_kv_heads > self.num_heads:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) must be <= num_heads ({self.num_heads}) for GQA."
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads}) for GQA."
            )

        rope_dim = self.head_dim if cfg.rope_dim is None else int(cfg.rope_dim)
        self._rope_dim_eff = rope_dim
        if self.use_rope:
            if not (0 < rope_dim <= self.head_dim):
                raise ValueError(f"rope_dim must be in (0, head_dim], got {rope_dim}")
            if rope_dim % 2 != 0:
                raise ValueError(f"rope_dim must be even, got {rope_dim}")
            self.rotary_emb = RotaryEmbedding(
                dim=rope_dim, max_seq_len=int(cfg.max_seq_len), base=float(cfg.rope_base)
            )
        else:
            self.rotary_emb = None

        use_bn = bool(cfg.use_bitnet)
        self.q_proj = make_linear(use_bn, self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = make_linear(use_bn, self.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = make_linear(use_bn, self.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = make_linear(use_bn, self.num_heads * self.head_dim, self.hidden_size)

        self.q_norm = QKRMSNorm(self.head_dim)
        self.k_norm = QKRMSNorm(self.head_dim)
        self.attn_dropout = nn.Dropout(float(cfg.attention_dropout))
        self.max_seq = int(cfg.max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if T > self.max_seq:
            raise ValueError(f"seq_len ({T}) exceeds max_seq ({self.max_seq})")

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_rope:
            cos, sin = self.rotary_emb(q, seq_len=T)
            rope_dim = self._rope_dim_eff
            if rope_dim < self.head_dim:
                q_rope, k_rope = apply_rope_optimized(
                    q[..., :rope_dim], k[..., :rope_dim], cos, sin
                )
                q = torch.cat([q_rope, q[..., rope_dim:]], dim=-1)
                k = torch.cat([k_rope, k[..., rope_dim:]], dim=-1)
            else:
                q, k = apply_rope_optimized(q, k, cos, sin)

        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=self.causal
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)
