"""Loader for optional C++ CPU kernel with safe fallback."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch


@lru_cache(maxsize=1)
def load_cpp_kernel(build: bool = False):
    """
    Returns loaded extension module or None.
    build=False keeps CI/mac path fast and fallback-safe.
    """
    if not build:
        return None
    try:
        from torch.utils.cpp_extension import load
    except Exception:
        return None

    src = Path(__file__).resolve().parent / "bitnet_cpu.cpp"
    if not src.exists():
        return None
    try:
        return load(
            name="mertformer_bitnet_cpu_ext",
            sources=[str(src)],
            verbose=False,
            extra_cflags=["-O2"],
        )
    except Exception:
        return None


def bitnet_cpu_linear(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    ext = load_cpp_kernel(build=False)
    if ext is None:
        out = torch.matmul(x, w.t())
        if bias is not None:
            out = out + bias
        return out
    if bias is None:
        bias = torch.zeros(w.size(0), dtype=x.dtype, device=x.device)
    return ext.bitnet_cpu_linear(x, w, bias)

