"""Quantum-inspired unitary layer (Cayley transform).

Mirrors ``vendor/upstream/layers/qinn.py`` (``newton_schulz_inverse``,
``UnitaryQINN``). The only change is that ``enabled`` is an explicit constructor
argument instead of a read of the global ``cfg`` singleton.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def newton_schulz_inverse(mat: torch.Tensor, num_iters: int = 6) -> torch.Tensor:
    """Approximate matrix inverse via Newton-Schulz iteration, computed in fp32."""
    orig_dtype = mat.dtype
    mat = mat.float()

    *batch, n, m = mat.shape
    assert n == m, "newton_schulz_inverse: square matrix expected."

    eye = torch.eye(n, device=mat.device, dtype=mat.dtype)
    if batch:
        eye = eye.expand(*batch, n, n)

    # Scale by ||A||_inf so that the iteration converges.
    row_sum = mat.abs().sum(dim=-1)
    norm_inf = row_sum.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    norm_inf = torch.clamp(norm_inf, min=1e-6)
    mat_scaled = mat / norm_inf

    x = mat_scaled.transpose(-1, -2)
    for _ in range(num_iters):
        ax = torch.matmul(mat_scaled, x)
        x = torch.matmul(x, (2.0 * eye - ax))

    return (x / norm_inf).to(orig_dtype)


class UnitaryQINN(nn.Module):
    """``out = x @ U^T`` with ``U = (I - S)^{-1}(I + S)``, ``S = A - A^T`` skew-symmetric."""

    def __init__(self, dim: int, num_iters: int = 6, enabled: bool = True) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_iters = int(num_iters)
        self.enabled = bool(enabled)
        self.A = nn.Parameter(torch.randn(dim, dim) * 1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        orig_dtype = x.dtype
        x = x.float()

        s = self.A - self.A.t()
        eye = torch.eye(self.dim, device=s.device, dtype=s.dtype)
        m_inv = newton_schulz_inverse(eye - s, num_iters=self.num_iters)
        u = torch.matmul(m_inv, eye + s)

        if not torch.isfinite(u).all():
            u = torch.where(torch.isfinite(u), u, eye)

        return torch.matmul(x, u.t()).to(orig_dtype)
