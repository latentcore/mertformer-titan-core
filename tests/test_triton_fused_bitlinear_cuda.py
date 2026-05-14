from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from layers.bitlinear import activation_quant, weight_quant
from mertformer_sdk.kernels.triton_fused_bitlinear import (
    is_triton_fused_available,
    triton_fused_ternary_linear,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_fused_bitlinear_forward_and_backward_cuda() -> None:
    if not is_triton_fused_available():
        pytest.skip("Triton not available")

    torch.manual_seed(20260510)
    x = torch.randn(5, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(6, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.randn(6, device="cuda", dtype=torch.float32, requires_grad=True)

    out = triton_fused_ternary_linear(x, w, b)
    ref = F.linear(activation_quant(x), weight_quant(w), b)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    loss = out.float().pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    assert b.grad is not None and torch.isfinite(b.grad).all()

