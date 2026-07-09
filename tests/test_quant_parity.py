"""Parity test: ``layers.liquid.jit_quant`` must match ``layers.bitlinear.weight_quant``.

DECISIONS.md / bitlinear.py:164-165 record these two ternary (1.58-bit) quantizers as
locked in parity — both per-row RMS scaling. ``weight_quant`` is the training-forward
quantizer (Straight-Through Estimator); ``jit_quant`` is the JIT/inference quantizer
(no STE). Their FORWARD VALUES must be identical: if one drifts to absmean scaling while
the other stays RMS, train and eval silently diverge. This is the executable form of that
invariant.
"""
import torch

from layers.bitlinear import weight_quant
from layers.liquid import jit_quant


def _rand_weight(rows: int = 16, cols: int = 32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=g)


def test_jit_quant_matches_weight_quant_forward():
    for seed in range(5):
        w = _rand_weight(seed=seed)
        train_forward = weight_quant(w)   # STE only affects the backward pass
        infer = jit_quant(w)
        assert torch.allclose(train_forward, infer, atol=1e-6), (
            f"ternary quant parity broken (seed={seed}) — bitlinear.weight_quant and "
            f"liquid.jit_quant disagree; keep both per-row RMS."
        )


def test_weight_quant_levels_are_ternary():
    w = _rand_weight(seed=7)
    scale = torch.sqrt((w ** 2).mean(dim=1, keepdim=True)).clamp(min=1e-5)
    levels = torch.round(weight_quant(w) / scale)
    assert set(torch.unique(levels).tolist()).issubset({-1.0, 0.0, 1.0})


def test_weight_quant_ste_keeps_gradient_finite():
    w = _rand_weight(seed=3).requires_grad_(True)
    weight_quant(w).sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()
