from __future__ import annotations

import pytest
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
        # CPU/Triton-yok yolunda çağrının GERÇEKTEN RuntimeError fırlatması
        # zorunlu kılınır. Önceki try/except yapısında hata fırlamazsa hiçbir
        # assert çalışmadan test sessizce yeşil geçiyordu (fake-green gate).
        with pytest.raises(RuntimeError) as exc_info:
            triton_fused_ternary_linear(x, w)
        assert "CUDA" in str(exc_info.value) or "Triton" in str(exc_info.value)

