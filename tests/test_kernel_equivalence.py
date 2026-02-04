from __future__ import annotations

import pytest
import torch

try:
    from mertformer_sdk.kernels.triton_ternary import (
        triton_ternary_linear,
        is_triton_available,
        _quantize_activation,
        _quantize_weight,
    )
except Exception:
    triton_ternary_linear = None
    is_triton_available = lambda: False  # type: ignore


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_kernel_equivalence():
    if not is_triton_available():
        pytest.skip("Triton not available")

    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    w = torch.randn(6, 8, device="cuda", dtype=torch.float32)

    out = triton_ternary_linear(x, w)

    x_q, x_scale = _quantize_activation(x)
    w_q, w_scale = _quantize_weight(w)
    acc = x_q.int() @ w_q.int().t()
    ref = acc.float() * (w_scale.view(1, -1) / x_scale)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
