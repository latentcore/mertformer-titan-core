"""Coverage for the QINN Cayley transform (layers/qinn.py).

Exercises newton_schulz_inverse accuracy, the orthogonality of the UnitaryQINN
transform (norm preservation), exact bypass when disabled, and finite output.
Pure CPU, deterministic (fixed seeds) — safe for the offline gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import cfg  # noqa: E402
from layers.qinn import UnitaryQINN, newton_schulz_inverse  # noqa: E402


def test_newton_schulz_inverse_approximates_inverse():
    """On the well-conditioned Cayley denominator M = I - S (S skew-symmetric),
    Newton-Schulz must approximate the true inverse: inv(M) @ M ≈ I."""
    torch.manual_seed(0)
    dim = 16
    a = torch.randn(dim, dim) * 0.1
    s = a - a.t()  # skew-symmetric
    m = torch.eye(dim) - s
    inv = newton_schulz_inverse(m, num_iters=12)
    eye = torch.eye(dim)
    max_err = (inv @ m - eye).abs().max().item()
    assert torch.allclose(inv @ m, eye, atol=1e-2), max_err


def test_newton_schulz_inverse_requires_square():
    with pytest.raises(AssertionError):
        newton_schulz_inverse(torch.randn(4, 5))


def test_unitary_qinn_preserves_norm_when_enabled():
    """The Cayley transform U = (I-S)^{-1}(I+S) is orthogonal, so x @ U^T must
    preserve the per-token L2 norm. This is the operational orthogonality check."""
    prev = getattr(cfg, "use_qinn", False)
    cfg.use_qinn = True
    try:
        torch.manual_seed(1)
        layer = UnitaryQINN(dim=32, num_iters=12)
        layer.A.data = torch.randn(32, 32) * 0.1  # non-trivial skew so the test is meaningful
        x = torch.randn(2, 5, 32)
        out = layer(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        n_in = x.norm(dim=-1)
        n_out = out.norm(dim=-1)
        assert torch.allclose(n_in, n_out, atol=3e-2, rtol=3e-2), (n_in - n_out).abs().max().item()
    finally:
        cfg.use_qinn = prev


def test_unitary_qinn_exact_bypass_when_disabled():
    prev = getattr(cfg, "use_qinn", False)
    cfg.use_qinn = False
    try:
        layer = UnitaryQINN(dim=8)
        x = torch.randn(1, 3, 8)
        assert torch.equal(layer(x), x)
    finally:
        cfg.use_qinn = prev
