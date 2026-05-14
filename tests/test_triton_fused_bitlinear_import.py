from __future__ import annotations

import torch

from mertformer_sdk.kernels.triton_fused_bitlinear import (
    is_triton_fused_available,
    triton_fused_ternary_linear,
)


def test_triton_fused_module_imports_without_cuda() -> None:
    assert isinstance(is_triton_fused_available(), bool)
    x = torch.randn(2, 4)
    w = torch.randn(3, 4)
    if not torch.cuda.is_available():
        try:
            triton_fused_ternary_linear(x, w)
        except RuntimeError as exc:
            assert "CUDA" in str(exc) or "Triton" in str(exc)

