"""Transformer block assembly.

Mirrors ``vendor/upstream/layers/mertformer_block.py::MertFormerBlock``:
the same pre-norm order, the same ``residual_scale = (2*num_layers)^-0.5``
DeepNorm-style scaling, Liquid *before* the FFN/MoE (so the router sees tokens
that already carry temporal context), and QINN last.

Carries DRIFT FIX #3: the Liquid hidden state is threaded in and back out, as
the canonical block does at ``mertformer_block.py:230``. ``chess_5080_onefile.py``
called ``self.liquid(x)`` and dropped the returned state.

The KV-cache argument from the canonical signature is not carried: this trunk is
an encoder over a fixed 76-token board and never decodes step by step.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import ModelConfig
from .attention import GQA
from .ffn import MertFormerFFN
from .liquid import LiquidMixer
from .moe import MoE
from .norm import RMSNorm
from .qinn import UnitaryQINN


class MertFormerBlock(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        h = int(cfg.hidden_size)
        eps = float(cfg.rms_norm_eps)

        self.norm1 = RMSNorm(h, eps=eps)
        self.norm2 = RMSNorm(h, eps=eps)
        self.residual_scale = (2 * int(cfg.num_layers)) ** -0.5

        self.attn = GQA(cfg)

        every_n = int(cfg.moe_every_n_layers)
        self.is_moe_layer = bool(cfg.use_moe) and every_n > 0 and ((self.layer_id + 1) % every_n == 0)
        self.ff: nn.Module = MoE(cfg) if self.is_moe_layer else MertFormerFFN(cfg)

        self.qinn: Optional[UnitaryQINN] = None
        if cfg.use_qinn:
            q_every = max(1, int(cfg.qinn_every_n_layers))
            if (self.layer_id + 1) % q_every == 0:
                self.qinn = UnitaryQINN(h, num_iters=int(cfg.qinn_iters), enabled=True)

        self.liquid: Optional[LiquidMixer] = None
        if cfg.use_liquid and self.layer_id in cfg.resolved_liquid_layers():
            self.liquid = LiquidMixer(h, use_bitnet=bool(cfg.use_bitnet))

    def forward(
        self,
        x: torch.Tensor,
        liquid_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # 1) attention
        h = self.norm1(x)
        x = x + self.attn(h) * self.residual_scale

        # 2) liquid (before the router, matching the canonical order)
        present_liquid_state: Optional[torch.Tensor] = None
        if self.liquid is not None:
            x, present_liquid_state = self.liquid(x, h_init=liquid_state, return_state=True)

        # 3) feed-forward / MoE
        h = self.norm2(x)
        if self.is_moe_layer:
            ff_out, aux_loss = self.ff(h)
        else:
            ff_out = self.ff(h)
            aux_loss = h.new_zeros(())
        x = x + ff_out * self.residual_scale

        # 4) QINN
        if self.qinn is not None:
            x = self.qinn(x)

        if aux_loss.ndim > 0:
            aux_loss = aux_loss.sum()
        return x, aux_loss, present_liquid_state
