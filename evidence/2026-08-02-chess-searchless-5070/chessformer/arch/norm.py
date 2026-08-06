"""Normalization layers.

``RMSNorm`` mirrors ``vendor/upstream/layers/mertformer_block.py::RMSNorm`` and
``_QKRMSNorm`` mirrors ``vendor/upstream/layers/mla.py::_QKRMSNorm``. Both
compute the reciprocal square root in fp32 and cast back, which is what makes
them stable under bf16 autocast.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms.to(x.dtype) * self.weight.to(x.dtype)


class QKRMSNorm(nn.Module):
    """Per-head RMS norm applied to q/k before RoPE (attention stabilization)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms.to(x.dtype) * self.weight.to(x.dtype)
