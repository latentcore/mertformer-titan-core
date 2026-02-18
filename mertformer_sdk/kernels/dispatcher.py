"""Kernel backend selection and fallback matrix."""
from __future__ import annotations

import os
from typing import Literal

import torch

Backend = Literal["triton_cuda", "cpp_cpu", "mps_optimized", "pytorch_fallback"]


def select_backend(x: torch.Tensor, w: torch.Tensor) -> Backend:
    # Explicit override for tests/ops.
    forced = os.getenv("MERTFORMER_KERNEL_BACKEND", "").strip().lower()
    if forced in {"triton_cuda", "cpp_cpu", "mps_optimized", "pytorch_fallback"}:
        return forced  # type: ignore[return-value]

    if x.is_cuda and w.is_cuda:
        try:
            from .triton_ternary import is_triton_available

            if is_triton_available():
                return "triton_cuda"
        except Exception:
            pass
        return "pytorch_fallback"

    if x.device.type == "mps" and w.device.type == "mps":
        return "mps_optimized"

    if x.device.type == "cpu" and w.device.type == "cpu":
        return "cpp_cpu"

    return "pytorch_fallback"

