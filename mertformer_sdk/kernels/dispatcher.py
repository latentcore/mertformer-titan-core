"""Kernel backend selection and fallback matrix.

This module keeps backend routing deterministic and config-driven.
All non-native backends must gracefully fallback to safe PyTorch math.
"""
from __future__ import annotations

import os
from typing import Literal

import torch

Backend = Literal[
    "triton_cuda",
    "cpp_cpu",
    "mps_optimized",
    "metal_fallback",
    "vulkan_fallback",
    "npu_fallback",
    "pytorch_fallback",
]


def select_backend(x: torch.Tensor, w: torch.Tensor) -> Backend:
    # Explicit override for tests/ops.
    forced = os.getenv("MERTFORMER_KERNEL_BACKEND", "").strip().lower()
    if forced in {
        "triton_cuda",
        "cpp_cpu",
        "mps_optimized",
        "metal_fallback",
        "vulkan_fallback",
        "npu_fallback",
        "pytorch_fallback",
    }:
        return forced  # type: ignore[return-value]

    # Optional backend feature flags (on by default for safe fallback interfaces).
    enable_cpp = os.getenv("MERTFORMER_ENABLE_CPP_KERNEL", "1") == "1"
    enable_metal = os.getenv("MERTFORMER_ENABLE_METAL_KERNEL", "1") == "1"
    enable_vulkan = os.getenv("MERTFORMER_ENABLE_VULKAN_KERNEL", "1") == "1"
    enable_npu = os.getenv("MERTFORMER_ENABLE_NPU_DIRECT", "1") == "1"

    if x.is_cuda and w.is_cuda:
        try:
            from .triton_fused_bitlinear import is_triton_fused_available

            if is_triton_fused_available():
                return "triton_cuda"
        except Exception:
            pass
        try:
            from .triton_ternary import is_triton_available

            if is_triton_available():
                return "triton_cuda"
        except Exception:
            pass
        if enable_npu:
            return "npu_fallback"
        return "pytorch_fallback"

    if x.device.type == "mps" and w.device.type == "mps":
        if enable_metal:
            return "metal_fallback"
        return "mps_optimized"

    if x.device.type == "cpu" and w.device.type == "cpu":
        if enable_cpp:
            return "cpp_cpu"
        if enable_vulkan:
            return "vulkan_fallback"
        return "pytorch_fallback"

    return "pytorch_fallback"
