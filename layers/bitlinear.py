"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v27.0-FINAL (Locked & Sealed)
Status : PRODUCTION READY (LOCKED)
==============================================================================
"""

__version__ = "27.0-FINAL"
__author__ = "Mert"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """
    TR: Aktivasyonları INT8 quantizasyonuna dönüştürür (Straight-Through Estimator ile).
    EN: Quantizes activations to INT8 using Straight-Through Estimator.

    Args:
        x (torch.Tensor): Girdi aktivasyon tensörü / Input activation tensor
    Returns:
        torch.Tensor: Quantize edilmiş aktivasyon / Quantized activation
    """
    # Maksimum mutlak değeri bul ve ölçekleme faktörü hesapla
    max_abs = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    scale = 127.0 / max_abs

    # Quantize et ve [-127, 127] aralığına sınırla
    x_q = torch.round(x * scale).clamp(-127, 127) / scale

    # Straight-Through Estimator: Gradient'ları quantize edilmemiş değerden geçir
    return x + (x_q - x).detach()


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """
    TR: Ağırlıkları ternary quantizasyonuna dönüştürür (BitNet 1.58-bit).
    EN: Quantizes weights to ternary quantization (BitNet 1.58-bit).

    Args:
        w (torch.Tensor): Girdi ağırlık tensörü / Input weight tensor
    Returns:
        torch.Tensor: Quantize edilmiş ağırlık / Quantized weight
    """
    # [V26.0 FIX] RMS Scale: Mean yerine RMS kullan (daha stabil)
    scale = torch.sqrt((w ** 2).mean(dim=1, keepdim=True)).clamp(min=1e-5)

    # Normalize et ve [-1, 1] aralığına quantize et
    w_norm = w / scale
    w_q = torch.round(w_norm).clamp(-1.0, 1.0)
    w_q_real = w_q * scale

    # Straight-Through Estimator
    return w + (w_q_real - w).detach()


class BitLinear(nn.Linear):
    """
    TR: BitNet Linear katmanı - 1.58-bit quantizasyon ile hafıza optimizasyonu.
    EN: BitNet Linear layer - Memory optimization with 1.58-bit quantization.

    Özellikler / Features:
    - Aktivasyonlar INT8 quantize edilir / Activations are INT8 quantized
    - Ağırlıklar ternary quantize edilir / Weights are ternary quantized
    - STE ile eğitim kararlılığı sağlanır / Training stability via STE
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TR: İleri yayılım - Aktivasyon ve ağırlık quantizasyonu uygular.
        EN: Forward pass - Applies activation and weight quantization.

        Args:
            x (torch.Tensor): Girdi tensörü / Input tensor
        Returns:
            torch.Tensor: Çıktı tensörü / Output tensor
        """
        x_q = activation_quant(x)
        w_q = weight_quant(self.weight)
        return F.linear(x_q, w_q, self.bias)
