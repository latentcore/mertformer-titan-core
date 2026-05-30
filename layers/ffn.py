"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert"

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# TR: Yerel import'lar / EN: Local imports
from config.config import cfg
from layers.bitlinear import BitLinear, activation_quant, weight_quant


_FFN_PACK_ENABLED = os.environ.get("TITAN_FFN_PACK", "0") == "1"


def set_ffn_pack_enabled(enabled: bool) -> None:
    """Runtime toggle for the lossless FFN gate+up packed projection path."""
    global _FFN_PACK_ENABLED
    _FFN_PACK_ENABLED = bool(enabled)


def _ffn_packed_bitlinear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply BitLinear fallback math to a packed output weight matrix."""
    x_q = activation_quant(x)
    w_q = weight_quant(weight)
    return F.linear(x_q, w_q, None)


class MertFormerFFN(nn.Module):
    """
    TR: Feed Forward Network - BitNet SwiGLU mimarisi.
    EN: Feed Forward Network - BitNet SwiGLU architecture.

    Özellikler / Features:
    - SwiGLU aktivasyon fonksiyonu (Llama-3 / Mistral standardı)
    - BitLinear katmanlar ile 1.58-bit ağırlık simülasyonu
    - Configurable intermediate size
    """

    def __init__(self) -> None:
        """TR: MertFormerFFN başlatıcı. EN: MertFormerFFN initializer."""
        super().__init__()

        hidden_size = cfg.hidden_size
        intermediate_size = getattr(cfg, "intermediate_size", hidden_size * 4)
        ffn_dropout = float(getattr(cfg, "ffn_dropout", 0.0))

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # TR: SwiGLU: gate, up, down (tamamı BitLinear)
        # EN: SwiGLU: gate, up, down (all BitLinear)
        self.gate_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = BitLinear(intermediate_size, hidden_size, bias=False)
        
        # TR: Dropout katmanı / EN: Dropout layer
        self.dropout = nn.Dropout(ffn_dropout)

    def _forward_packed(self, x: torch.Tensor) -> torch.Tensor:
        """TR: gate+up projeksiyonlarını tek matmul'da hesaplar. EN: Packs gate+up."""
        packed_weight = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
        gate_up = _ffn_packed_bitlinear(x, packed_weight)
        gate, up = gate_up.chunk(2, dim=-1)
        x_inter = F.silu(gate) * up
        x_inter = self.dropout(x_inter)
        return self.down_proj(x_inter)

    def _forward_baseline(self, x: torch.Tensor) -> torch.Tensor:
        # TR: Gate ve Up projeksiyonları / EN: Gate and Up projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # TR: SwiGLU: SiLU(gate) * up / EN: SwiGLU: SiLU(gate) * up
        x_inter = F.silu(gate) * up

        # TR: Dropout (opsiyonel, ffn_dropout > 0 ise) / EN: Dropout (optional, if ffn_dropout > 0)
        x_inter = self.dropout(x_inter)

        # TR: Down projeksiyon (orijinal hidden boyutuna dönüş)
        # EN: Down projection (return to original hidden dimension)
        return self.down_proj(x_inter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TR: İleri yayılım - SwiGLU aktivasyonu ile feed-forward.
        EN: Forward pass - Feed-forward with SwiGLU activation.

        Args:
            x (torch.Tensor): Girdi tensörü / Input tensor
        Returns:
            torch.Tensor: Çıktı tensörü / Output tensor
        """
        if _FFN_PACK_ENABLED and os.environ.get("MERTFORMER_LOWBIT_KERNEL", "0") != "1":
            return self._forward_packed(x)
        return self._forward_baseline(x)
