"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright 2026 Mert Yunlu
Licensed under the Apache License, Version 2.0 (see LICENSE).

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

from config.build_label import BUILD_LABEL as __version__
__author__ = "Mert Yünlü"

import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOWBIT_KERNEL_ENABLED = os.getenv("MERTFORMER_LOWBIT_KERNEL", "0") == "1"
_TENSORCORE_ENABLED = os.getenv("MERTFORMER_TENSORCORE", "0") == "1"
_FUSED_KERNEL_ENABLED = os.getenv("MERTFORMER_FUSED_KERNEL", "1") == "1"


def set_lowbit_kernel_enabled(enabled: bool) -> None:
    """
    TR: Low-bit kernel yolunu aç/kapat (opt-in).
    EN: Enable/disable low-bit kernel path (opt-in).
    """
    global _LOWBIT_KERNEL_ENABLED
    _LOWBIT_KERNEL_ENABLED = bool(enabled)


def _kernel_strict() -> bool:
    return os.getenv("MERTFORMER_KERNEL_STRICT", "0") == "1"


def _unavailable(backend: str) -> None:
    # TR: [strict-mode fix] Seçilen backend kullanılamıyorsa strict modda sessiz
    #     torch fallback yerine HATA ver; aksi halde None dondur (yumusak fallback).
    # EN: [strict-mode fix] When the selected backend is unavailable, raise under
    #     strict mode instead of silently torch-falling-back; else return None.
    if _kernel_strict():
        raise RuntimeError(
            f"MERTFORMER_KERNEL_STRICT=1 but selected backend '{backend}' is unavailable; "
            "refusing to silently fall back to torch."
        )
    return None


def _try_lowbit_kernel(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor | None:
    if not _LOWBIT_KERNEL_ENABLED:
        return None

    try:
        from mertformer_sdk.kernels.dispatcher import select_backend

        backend = select_backend(x, w)
        if backend == "triton_cuda":
            if _FUSED_KERNEL_ENABLED:
                from mertformer_sdk.kernels.triton_fused_bitlinear import (
                    is_triton_fused_available,
                    triton_fused_ternary_linear,
                )

                if is_triton_fused_available():
                    return triton_fused_ternary_linear(x, w, bias)

            from mertformer_sdk.kernels.triton_ternary import is_triton_available, triton_ternary_linear

            if not is_triton_available():
                return _unavailable("triton_cuda")
            # TR: Fused-olmayan triton_ternary'de STE/autograd yok. Training'de
            #     grad isteyen tensorlerde kullanma -> torch STE fallback'a dus.
            # EN: The non-fused triton_ternary has no STE/autograd. Never use it
            #     for grad-requiring tensors in training -> fall back to torch STE.
            if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
                return None
            return triton_ternary_linear(x, w, bias, use_tensorcore=_TENSORCORE_ENABLED)

        if backend == "cpp_cpu":
            from mertformer_sdk.kernels.cpp.loader import bitnet_cpu_linear

            # TR: [B3 fix] cpp_cpu cekirdegi duz matmul yapiyor; diger tum
            #     backend dallari gibi ONCE quantize et, yoksa BitNet b1.58
            #     yerine sessizce full-precision calisir (yanlis numerik).
            # EN: [B3 fix] the cpp_cpu kernel is a plain matmul; quantize first
            #     like every other backend branch, else it silently runs
            #     full-precision instead of BitNet b1.58 (wrong numerics).
            x_q = activation_quant(x)
            w_q = weight_quant(w)
            return bitnet_cpu_linear(x_q, w_q, bias)

        if backend == "metal_fallback":
            from mertformer_sdk.kernels.metal.engine import metal_linear

            x_q = activation_quant(x)
            w_q = weight_quant(w)
            return metal_linear(x_q, w_q, bias)

        if backend == "vulkan_fallback":
            from mertformer_sdk.kernels.vulkan.engine import vulkan_linear

            x_q = activation_quant(x)
            w_q = weight_quant(w)
            return vulkan_linear(x_q, w_q, bias)

        if backend == "npu_fallback":
            from mertformer_sdk.kernels.npu.engine import npu_linear

            x_q = activation_quant(x)
            w_q = weight_quant(w)
            return npu_linear(x_q, w_q, bias)

        if backend == "mps_optimized":
            # MPS-optimized path (no custom shader): keep deterministic quantized math.
            x_q = activation_quant(x)
            w_q = weight_quant(w)
            return F.linear(x_q, w_q, bias)

        return _unavailable(str(backend))
    except Exception:
        if _kernel_strict():
            raise
        return None


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
    # RMS Scale: Mean yerine RMS kullan (daha stabil), per-row (dim=1).
    # PARITE: layers/liquid.py:jit_quant ile kilitli (ikisi de per-row RMS). Bunu
    # absmean'e cevirirsen jit_quant'i da cevir, yoksa train/eval ayrisir.
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
        # Opt-in low-bit kernel path. Falls back safely.
        lowbit_out = _try_lowbit_kernel(x, self.weight, self.bias)
        if lowbit_out is not None:
            return lowbit_out

        x_q = activation_quant(x)
        w_q = weight_quant(self.weight)
        return F.linear(x_q, w_q, self.bias)
