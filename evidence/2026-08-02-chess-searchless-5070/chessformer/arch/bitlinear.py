"""BitNet b1.58 quantization.

Exact mirror of ``vendor/upstream/layers/bitlinear.py`` (functions
``activation_quant``, ``weight_quant``, class ``BitLinear``), minus the opt-in
low-bit kernel dispatch -- that path routes to ``mertformer_sdk`` kernels which
are not vendored here, and every one of its branches falls back to the same
eager math implemented below.

Parity is asserted numerically by ``tests/test_arch_parity.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """Per-token INT8 activation quantization with a straight-through estimator."""
    max_abs = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    scale = 127.0 / max_abs
    x_q = torch.round(x * scale).clamp(-127, 127) / scale
    return x + (x_q - x).detach()


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """Per-row RMS-scaled ternary weight quantization (BitNet b1.58) with STE.

    PARITY: the canonical file notes this must stay locked to
    ``liquid.jit_quant`` (both per-row RMS). ``chessformer/arch/liquid.py``
    keeps the same pairing.
    """
    scale = torch.sqrt((w ** 2).mean(dim=1, keepdim=True)).clamp(min=1e-5)
    w_norm = w / scale
    w_q = torch.round(w_norm).clamp(-1.0, 1.0)
    w_q_real = w_q * scale
    return w + (w_q_real - w).detach()


class BitLinear(nn.Linear):
    """``nn.Linear`` whose forward quantizes activations to INT8 and weights to ternary."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = activation_quant(x)
        w_q = weight_quant(self.weight)
        return F.linear(x_q, w_q, self.bias)


def make_linear(use_bitnet: bool, in_features: int, out_features: int, bias: bool = False) -> nn.Linear:
    """Single switch used by every projection in the trunk."""
    if use_bitnet:
        return BitLinear(in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)
