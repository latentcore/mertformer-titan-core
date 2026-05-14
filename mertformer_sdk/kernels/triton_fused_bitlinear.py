"""Triton fused BitLinear kernels with STE-compatible training backward.

This module is an opt-in CUDA path. It is designed for real training attempts:
forward uses Triton int8 activation + ternary weight GEMM, while backward can
use Triton kernels for the same straight-through estimator gradient contract as
the PyTorch BitLinear fallback.

Claim boundary: this is a real CUDA/Triton implementation surface, but speedup
must be measured on target hardware before any performance claim is made.
"""
from __future__ import annotations

import os
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


def is_triton_fused_available() -> bool:
    return _TRITON_AVAILABLE


def _activation_quant_fake(x: torch.Tensor) -> torch.Tensor:
    max_abs = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    scale = 127.0 / max_abs
    x_q = torch.round(x * scale).clamp(-127, 127) / scale
    return x + (x_q - x).detach()


def _weight_quant_fake(w: torch.Tensor) -> torch.Tensor:
    scale = torch.sqrt((w.detach() ** 2).mean(dim=1, keepdim=True)).clamp(min=1e-5)
    w_q = torch.round(w / scale).clamp(-1.0, 1.0) * scale
    return w + (w_q - w).detach()


def _quantize_weight_int(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        _TRITON_AVAILABLE
        and w.is_cuda
        and os.getenv("MERTFORMER_FUSED_WEIGHT_QUANT", "triton").strip().lower()
        in {"1", "true", "yes", "on", "triton"}
    ):
        return _quantize_weight_int_triton(w)
    scale = torch.sqrt((w.detach() ** 2).mean(dim=1, keepdim=True)).clamp(min=1e-5)
    w_q = torch.round(w.detach() / scale).clamp(-1.0, 1.0).to(torch.int8)
    return w_q.contiguous(), scale.squeeze(1).contiguous().float()


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


if _TRITON_AVAILABLE:
    _MATMUL_AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    ]

    @triton.jit
    def _weight_quant_kernel(
        w_ptr,
        wq_ptr,
        wscale_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_wk: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_K)
        mask = offs < K
        vals = tl.load(w_ptr + row * stride_wn + offs * stride_wk, mask=mask, other=0.0).to(tl.float32)
        rms = tl.sqrt(tl.sum(vals * vals, axis=0) / K)
        scale = tl.maximum(rms, 1.0e-5)
        normed = vals / scale
        rounded = tl.where(normed >= 0.0, tl.floor(normed + 0.5), tl.ceil(normed - 0.5))
        q = tl.minimum(tl.maximum(rounded, -1.0), 1.0).to(tl.int8)
        tl.store(wq_ptr + row * K + offs, q, mask=mask)
        tl.store(wscale_ptr + row, scale)

    @triton.jit
    def _activation_quant_kernel(
        x_ptr,
        xq_ptr,
        xscale_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xk: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_K)
        mask = offs < K
        vals = tl.load(x_ptr + row * stride_xm + offs * stride_xk, mask=mask, other=0.0).to(tl.float32)
        max_abs = tl.max(tl.abs(vals), axis=0)
        scale = 127.0 / tl.maximum(max_abs, 1.0e-5)
        scaled = vals * scale
        q = tl.where(scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
        q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
        tl.store(xq_ptr + row * K + offs, q, mask=mask)
        tl.store(xscale_ptr + row, scale)

    @triton.jit
    def _rmsnorm_activation_quant_kernel(
        x_ptr,
        rms_ptr,
        xq_ptr,
        xscale_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xk: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_K)
        mask = offs < K
        vals = tl.load(x_ptr + row * stride_xm + offs * stride_xk, mask=mask, other=0.0).to(tl.float32)
        rms_w = tl.load(rms_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(vals * vals, axis=0) / K
        normed = vals * tl.rsqrt(var + eps) * rms_w
        max_abs = tl.max(tl.abs(normed), axis=0)
        scale = 127.0 / tl.maximum(max_abs, 1.0e-5)
        scaled = normed * scale
        q = tl.where(scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
        q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
        tl.store(xq_ptr + row * K + offs, q, mask=mask)
        tl.store(xscale_ptr + row, scale)

    @triton.autotune(configs=_MATMUL_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _int8_ternary_matmul_dequant_kernel_autotuned(
        a_ptr,
        b_ptr,
        xscale_ptr,
        wscale_ptr,
        bias_ptr,
        out_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * K + k_idxs[None, :],
                mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
                other=0,
            )
            b = tl.load(
                b_ptr + offs_n[None, :] * K + k_idxs[:, None],
                mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
                other=0,
            )
            acc += tl.dot(a, b)

        x_scale = tl.load(xscale_ptr + offs_m, mask=offs_m < M, other=1.0).to(tl.float32)
        w_scale = tl.load(wscale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        out = acc.to(tl.float32) * (w_scale[None, :] / x_scale[:, None])
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            out += bias[None, :]

        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


    @triton.jit
    def _int8_ternary_matmul_dequant_kernel_static(
        a_ptr,
        b_ptr,
        xscale_ptr,
        wscale_ptr,
        bias_ptr,
        out_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * K + k_idxs[None, :],
                mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
                other=0,
            )
            b = tl.load(
                b_ptr + offs_n[None, :] * K + k_idxs[:, None],
                mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
                other=0,
            )
            acc += tl.dot(a, b)

        x_scale = tl.load(xscale_ptr + offs_m, mask=offs_m < M, other=1.0).to(tl.float32)
        w_scale = tl.load(wscale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        out = acc.to(tl.float32) * (w_scale[None, :] / x_scale[:, None])
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            out += bias[None, :]

        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    @triton.autotune(configs=_MATMUL_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _grad_input_kernel_autotuned(
        grad_ptr,
        wq_ptr,
        wscale_ptr,
        gx_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_n = tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for n0 in range(0, N, BLOCK_N):
            n_idxs = n0 + offs_n
            grad = tl.load(
                grad_ptr + offs_m[:, None] * N + n_idxs[None, :],
                mask=(offs_m[:, None] < M) & (n_idxs[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            wq = tl.load(
                wq_ptr + n_idxs[:, None] * K + offs_k[None, :],
                mask=(n_idxs[:, None] < N) & (offs_k[None, :] < K),
                other=0,
            ).to(tl.float32)
            ws = tl.load(wscale_ptr + n_idxs, mask=n_idxs < N, other=0.0).to(tl.float32)
            acc += tl.dot(grad, wq * ws[:, None])
        tl.store(
            gx_ptr + offs_m[:, None] * K + offs_k[None, :],
            acc,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        )

    @triton.autotune(configs=_MATMUL_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _grad_weight_kernel_autotuned(
        grad_ptr,
        xq_ptr,
        xscale_ptr,
        gw_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_m = tl.arange(0, BLOCK_M)
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for m0 in range(0, M, BLOCK_M):
            m_idxs = m0 + offs_m
            grad = tl.load(
                grad_ptr + m_idxs[None, :] * N + offs_n[:, None],
                mask=(m_idxs[None, :] < M) & (offs_n[:, None] < N),
                other=0.0,
            ).to(tl.float32)
            xq = tl.load(
                xq_ptr + m_idxs[:, None] * K + offs_k[None, :],
                mask=(m_idxs[:, None] < M) & (offs_k[None, :] < K),
                other=0,
            ).to(tl.float32)
            xs = tl.load(xscale_ptr + m_idxs, mask=m_idxs < M, other=1.0).to(tl.float32)
            acc += tl.dot(grad, xq / xs[:, None])
        tl.store(
            gw_ptr + offs_n[:, None] * K + offs_k[None, :],
            acc,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
        )


def _quantize_weight_int_triton(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not w.is_cuda:
        raise RuntimeError("Triton weight quantization requires CUDA tensors")
    w_2d = w.detach().contiguous()
    N, K = w_2d.shape
    block_k = _next_power_of_2(K)
    w_q = torch.empty((N, K), device=w.device, dtype=torch.int8)
    w_scale = torch.empty((N,), device=w.device, dtype=torch.float32)
    _weight_quant_kernel[(N,)](
        w_2d,
        w_q,
        w_scale,
        N,
        K,
        w_2d.stride(0),
        w_2d.stride(1),
        BLOCK_K=block_k,
    )
    return w_q, w_scale


def _triton_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    rms_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    return_quant: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda or not w.is_cuda:
        raise RuntimeError("Triton fused BitLinear requires CUDA tensors")
    if x.shape[-1] != w.shape[1]:
        raise ValueError("input last dimension must match weight input dimension")

    orig_shape = x.shape
    x_2d = x.contiguous().view(-1, x.shape[-1])
    w_q, w_scale = _quantize_weight_int(w)
    if bias is not None:
        bias = bias.contiguous()

    M, K = x_2d.shape
    N = w_q.shape[0]
    block_k_quant = _next_power_of_2(K)
    x_q = torch.empty((M, K), device=x.device, dtype=torch.int8)
    x_scale = torch.empty((M,), device=x.device, dtype=torch.float32)

    quant_grid = (M,)
    if rms_weight is None:
        _activation_quant_kernel[quant_grid](
            x_2d,
            x_q,
            x_scale,
            M,
            K,
            x_2d.stride(0),
            x_2d.stride(1),
            BLOCK_K=block_k_quant,
        )
    else:
        _rmsnorm_activation_quant_kernel[quant_grid](
            x_2d,
            rms_weight.contiguous(),
            x_q,
            x_scale,
            M,
            K,
            x_2d.stride(0),
            x_2d.stride(1),
            eps,
            BLOCK_K=block_k_quant,
        )

    out = torch.empty((M, N), device=x.device, dtype=x.dtype if x.dtype != torch.float64 else torch.float32)
    use_autotune = os.getenv("MERTFORMER_FUSED_AUTOTUNE", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if use_autotune:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
        _int8_ternary_matmul_dequant_kernel_autotuned[grid](
            x_q,
            w_q,
            x_scale,
            w_scale,
            bias if bias is not None else out,
            out,
            M,
            N,
            K,
            bias is not None,
        )
    else:
        block_m = int(os.getenv("MERTFORMER_FUSED_BLOCK_M", "64"))
        block_n = int(os.getenv("MERTFORMER_FUSED_BLOCK_N", "128"))
        block_k = int(os.getenv("MERTFORMER_FUSED_BLOCK_K", "64"))
        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
        _int8_ternary_matmul_dequant_kernel_static[grid](
            x_q,
            w_q,
            x_scale,
            w_scale,
            bias if bias is not None else out,
            out,
            M,
            N,
            K,
            bias is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )
    out = out.view(*orig_shape[:-1], N)
    if return_quant:
        return out, x_q, x_scale, w_q, w_scale
    return out


def _triton_backward(
    grad_output: torch.Tensor,
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    x_shape: torch.Size,
    w_dtype: torch.dtype,
    *,
    need_grad_x: bool,
    need_grad_w: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    grad_flat = grad_output.contiguous().view(-1, grad_output.shape[-1])
    M, N = grad_flat.shape
    K = x_q.shape[1]

    grad_x = None
    grad_w = None
    grid_x = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(K, meta["BLOCK_K"]))
    grid_w = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(K, meta["BLOCK_K"]))
    if need_grad_x:
        grad_x_flat = torch.empty((M, K), device=grad_output.device, dtype=grad_output.dtype)
        _grad_input_kernel_autotuned[grid_x](
            grad_flat,
            w_q,
            w_scale,
            grad_x_flat,
            M,
            N,
            K,
        )
        grad_x = grad_x_flat.view(x_shape)
    if need_grad_w:
        grad_w = torch.empty((N, K), device=grad_output.device, dtype=w_dtype)
        _grad_weight_kernel_autotuned[grid_w](
            grad_flat,
            x_q,
            x_scale,
            grad_w,
            M,
            N,
            K,
        )
    return grad_x, grad_w


class _TritonFusedBitLinearSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor]):
        out, x_q, x_scale, w_q, w_scale = _triton_forward(x, w, bias, return_quant=True)
        ctx.save_for_backward(
            x,
            w,
            x_q,
            x_scale,
            w_q,
            w_scale,
            bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        )
        ctx.has_bias = bias is not None
        ctx.x_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, w, x_q_int, x_scale, w_q_int, w_scale, bias_saved = ctx.saved_tensors
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).float()

        use_triton_backward = os.getenv("MERTFORMER_FUSED_BACKWARD", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if use_triton_backward:
            grad_x, grad_w = _triton_backward(
                grad_output,
                x_q_int,
                x_scale,
                w_q_int,
                w_scale,
                ctx.x_shape,
                w.dtype,
                need_grad_x=ctx.needs_input_grad[0],
                need_grad_w=ctx.needs_input_grad[1],
            )
        else:
            x_q = _activation_quant_fake(x).reshape(-1, x.shape[-1]).float()
            w_q = _weight_quant_fake(w).float()
            grad_x = grad_flat.matmul(w_q).reshape_as(x).to(x.dtype) if ctx.needs_input_grad[0] else None
            grad_w = grad_flat.t().matmul(x_q).to(w.dtype) if ctx.needs_input_grad[1] else None
        grad_bias = None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_flat.sum(dim=0).to(bias_saved.dtype)
        return grad_x, grad_w, grad_bias


def triton_fused_ternary_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Training-compatible fused BitLinear forward with STE backward."""
    return _TritonFusedBitLinearSTE.apply(x, w, bias)


def triton_fused_rmsnorm_ternary_linear(
    x: torch.Tensor,
    rms_weight: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Forward-only fused RMSNorm + activation quant + ternary linear.

    This helper is intended for microbenchmarks and future block fusion. The
    canonical BitLinear training path uses triton_fused_ternary_linear because
    RMSNorm lives outside BitLinear in the current architecture.
    """
    return _triton_forward(x, w, bias, rms_weight=rms_weight, eps=eps)
