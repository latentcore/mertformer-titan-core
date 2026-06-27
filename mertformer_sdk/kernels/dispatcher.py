"""Kernel backend selection and fallback matrix.

This module keeps backend routing deterministic and config-driven.
All non-native backends must gracefully fallback to safe PyTorch math.

HONEST-LABEL NOTE: Only ``triton_cuda`` (when Triton is available) maps to a
genuinely distinct kernel. The remaining labels
(``mps_optimized``, ``metal_fallback``, ``vulkan_fallback``, ``npu_fallback``,
``cpp_cpu`` without a real C++ build, and ``pytorch_fallback``) are NOT separate
hardware-accelerated kernels: downstream (see ``layers/bitlinear.py``) they all
resolve to the same quantize + ``F.linear`` / ``torch.matmul`` PyTorch math. The
distinct names exist for the test/parity matrix and for future real kernels; they
do not currently imply distinct numerics or hardware acceleration. The label set
and routing logic are kept frozen for parity tests and are intentionally NOT
collapsed here.
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
        # NOTE: enable_metal defaults to "1", so on the auto-selection path this
        # branch always returns "metal_fallback"; "mps_optimized" is unreachable
        # here unless the flag is explicitly disabled. It remains reachable via
        # the MERTFORMER_KERNEL_BACKEND override above. Both labels resolve to the
        # same PyTorch fallback math downstream (see module docstring); kept for
        # parity-matrix coverage, so the routing is intentionally left unchanged.
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
