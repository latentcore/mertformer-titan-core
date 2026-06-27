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
except ImportError:
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


def test_cpp_cpu_backend_quantizes_matches_reference(monkeypatch):
    """B3: the cpp_cpu BitLinear backend must QUANTIZE before the kernel, so its
    output equals F.linear(activation_quant(x), weight_quant(w)) -- NOT a raw matmul.
    Runs on CPU (no CUDA/triton), so it is part of the green suite."""
    import torch.nn.functional as F
    from layers import bitlinear
    import mertformer_sdk.kernels.dispatcher as dispatcher

    torch.manual_seed(0)
    x = torch.randn(3, 8, dtype=torch.float32)
    w = torch.randn(6, 8, dtype=torch.float32)

    quant_ref = F.linear(bitlinear.activation_quant(x), bitlinear.weight_quant(w))
    raw_matmul = F.linear(x, w)

    # Force the cpp_cpu branch deterministically.
    monkeypatch.setattr(dispatcher, "select_backend", lambda *_a, **_k: "cpp_cpu")
    bitlinear.set_lowbit_kernel_enabled(True)
    try:
        out = bitlinear._try_lowbit_kernel(x, w, None)
    finally:
        bitlinear.set_lowbit_kernel_enabled(False)

    assert out is not None, "cpp_cpu branch did not execute"
    # Quantized (correct BitNet) path, clearly distinct from a raw full-precision matmul.
    torch.testing.assert_close(out, quant_ref, rtol=1e-2, atol=1e-2)
    assert not torch.allclose(out, raw_matmul, rtol=1e-3, atol=1e-3), \
        "cpp_cpu output equals a raw unquantized matmul (B3 not fixed)"

