"""Tests for the RoPE helpers used by the (decoupled) attention path in layers/mla.py.

``GQA.forward`` applies rotary embeddings via ``apply_rope_optimized`` (and, in decoupled
mode, only over the trailing ``rope_dim`` subspace). RoPE is a rotation, so the invariants
below hold regardless of scale: identity at cos=1/sin=0, per-vector norm preservation, and
correct ``rotate_half`` behaviour. The last test mirrors the decoupled trailing-subspace
slice from ``GQA.forward`` to lock that only the tail is rotated.
"""
import torch

from layers.mla import apply_rope_optimized, rotate_half


def _rope_cos_sin(seq_len: int, dim: int, seed: int = 0):
    """Build a valid RoPE cos/sin (half-dim angles duplicated, per convention)."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(seq_len, dim // 2, generator=g)
    cos = torch.cat([theta.cos(), theta.cos()], dim=-1).view(1, 1, seq_len, dim)
    sin = torch.cat([theta.sin(), theta.sin()], dim=-1).view(1, 1, seq_len, dim)
    return cos, sin


def test_rotate_half():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_rope_identity_when_no_rotation():
    b, h, t, d = 2, 2, 3, 8
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    cos = torch.ones(1, 1, t, d)
    sin = torch.zeros(1, 1, t, d)
    q_out, k_out = apply_rope_optimized(q, k, cos, sin)
    assert torch.allclose(q_out, q) and torch.allclose(k_out, k)


def test_rope_preserves_vector_norm():
    b, h, t, d = 1, 1, 4, 8
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    cos, sin = _rope_cos_sin(t, d, seed=1)
    q_out, _ = apply_rope_optimized(q, k, cos, sin)
    assert torch.allclose(q_out.norm(dim=-1), q.norm(dim=-1), atol=1e-4)


def test_decoupled_mode_rotates_only_trailing_subspace():
    # Mirrors GQA.forward's decoupled branch: RoPE on the trailing rope_dim only.
    b, h, t, d, rope_dim = 1, 1, 2, 8, 4
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    cos, sin = _rope_cos_sin(t, rope_dim, seed=2)
    q_rope, _ = apply_rope_optimized(q[..., -rope_dim:], k[..., -rope_dim:], cos, sin)
    q_out = torch.cat([q[..., :-rope_dim], q_rope], dim=-1)
    assert torch.allclose(q_out[..., :-rope_dim], q[..., :-rope_dim])   # head untouched
    assert not torch.allclose(q_out[..., -rope_dim:], q[..., -rope_dim:])  # tail rotated
