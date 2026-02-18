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

__version__ = "1.0-BUILD30"
__author__ = "Mert"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# TR: Yerel import'lar / EN: Local imports
from config.config import cfg
from layers.bitlinear import BitLinear


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TR: İleri yayılım - SwiGLU aktivasyonu ile feed-forward.
        EN: Forward pass - Feed-forward with SwiGLU activation.

        Args:
            x (torch.Tensor): Girdi tensörü / Input tensor
        Returns:
            torch.Tensor: Çıktı tensörü / Output tensor
        """
        # TR: Gate ve Up projeksiyonları / EN: Gate and Up projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # TR: SwiGLU: SiLU(gate) * up / EN: SwiGLU: SiLU(gate) * up
        x_inter = F.silu(gate) * up

        # TR: Dropout (opsiyonel, ffn_dropout > 0 ise) / EN: Dropout (optional, if ffn_dropout > 0)
        x_inter = self.dropout(x_inter)

        # TR: Down projeksiyon (orijinal hidden boyutuna dönüş)
        # EN: Down projection (return to original hidden dimension)
        out = self.down_proj(x_inter)
        return out
