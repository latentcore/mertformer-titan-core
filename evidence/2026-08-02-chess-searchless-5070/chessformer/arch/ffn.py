"""Dense SwiGLU feed-forward block.

Mirrors ``vendor/upstream/layers/ffn.py::MertFormerFFN._forward_baseline``. The
optional packed gate+up path from upstream is a fusion of the same math and is
omitted; ``tests/test_arch_parity.py`` compares against the baseline path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .bitlinear import make_linear


class MertFormerFFN(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h = int(cfg.hidden_size)
        inter = int(cfg.intermediate_size)
        use_bn = bool(cfg.use_bitnet)
        self.gate_proj = make_linear(use_bn, h, inter)
        self.up_proj = make_linear(use_bn, h, inter)
        self.down_proj = make_linear(use_bn, inter, h)
        self.dropout = nn.Dropout(float(cfg.ffn_dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inter = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x_inter = self.dropout(x_inter)
        return self.down_proj(x_inter)
