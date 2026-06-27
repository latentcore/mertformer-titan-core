"""Experimental Triton kernel for ternary weight + INT8 activation GEMM.

Status: EXPERIMENTAL / NOT CANONICAL. This kernel is under active development
and is NOT the production path. The canonical / production fused BitLinear path
lives in ``triton_fused_bitlinear.py``; prefer that for real workloads.

The default ``_matmul_kernel`` here is a naive int32-accumulate loop
(``acc += tl.sum(a[:, :, None] * b[None, :, :], ...)``) and is intentionally
NOT tensor-core optimized. ``_matmul_kernel_tc`` (``use_tensorcore=True``) is a
tensor-core-friendly variant that is also experimental and unverified. There is
functional overlap with the canonical fused kernel; this module is kept for
development/benchmarking only and should not be relied on as the sealed path.
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:
    @triton.jit
    def _matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        k = 0
        while k < K:
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0).to(tl.int32)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0).to(tl.int32)
            # int32 accumulate (simple, not tensor-core optimized)
            # a: [BM, BK], b: [BK, BN] -> [BM, BK, BN] -> sum over BK
            acc += tl.sum(a[:, :, None] * b[None, :, :], axis=1)
            k += BLOCK_K
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    @triton.jit
    def _matmul_kernel_tc(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k = 0
        while k < K:
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
                other=0.0,
            ).to(tl.float16)
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
                other=0.0,
            ).to(tl.float16)
            # Tensor-core friendly dot (experimental)
            acc += tl.dot(a, b)
            k += BLOCK_K
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    scale = 127.0 / max_abs
    x_q = torch.round(x * scale).clamp(-127, 127).to(torch.int8)
    return x_q, scale


def _quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.sqrt((w ** 2).mean(dim=1, keepdim=True)).clamp(min=1e-5)
    w_norm = w / scale
    w_q = torch.round(w_norm).clamp(-1.0, 1.0).to(torch.int8)
    return w_q, scale.squeeze(1)


def triton_ternary_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_tensorcore: bool = False,
) -> torch.Tensor:
    """Ternary weight + INT8 activation GEMM using Triton (experimental)."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda or not w.is_cuda:
        raise RuntimeError("Triton kernel requires CUDA tensors")

    orig_shape = x.shape
    if x.dim() > 2:
        x_2d = x.view(-1, x.shape[-1])
    else:
        x_2d = x

    x_q, x_scale = _quantize_activation(x_2d)
    w_q, w_scale = _quantize_weight(w)

    M, K = x_q.shape
    N = w_q.shape[0]

    # output buffer
    out_dtype = torch.float32 if use_tensorcore else torch.int32
    out = torch.empty((M, N), device=x_q.device, dtype=out_dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    if use_tensorcore:
        _matmul_kernel_tc[grid](
            x_q,
            w_q,
            out,
            M,
            N,
            K,
            x_q.stride(0),
            x_q.stride(1),
            w_q.stride(1),
            w_q.stride(0),
            out.stride(0),
            out.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
    else:
        _matmul_kernel[grid](
            x_q,
            w_q,
            out,
            M,
            N,
            K,
            x_q.stride(0),
            x_q.stride(1),
            w_q.stride(1),
            w_q.stride(0),
            out.stride(0),
            out.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

    # dequant
    scale = w_scale.view(1, -1) / x_scale
    out_f = out.float() * scale
    if bias is not None:
        out_f = out_f + bias

    if x.dim() > 2:
        out_f = out_f.view(*orig_shape[:-1], out_f.shape[-1])
    return out_f
