#!/usr/bin/env python3
"""Standalone BitNet ternary kernel benchmark (single-file, self-contained).

This file intentionally embeds:
- Triton ternary kernel code
- Quantization helpers
- Reference implementation
- Benchmark harness

Usage examples:
  python3 scripts/bitnet_kernel_benchmark_standalone.py
  python3 scripts/bitnet_kernel_benchmark_standalone.py --shapes 2048x2048x2048,4096x2048x2048 --iters 50
  python3 scripts/bitnet_kernel_benchmark_standalone.py --use-tensorcore
  # Notebook/Colab (no CLI args):
  # from scripts.bitnet_kernel_benchmark_standalone import run_default; run_default()

Performance note:
  - This benchmark runs on a single selected device.
  - Multi-GPU instances (e.g., T4 x2) are not aggregated by this script.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional, Sequence

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
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
                other=0,
            ).to(tl.int32)
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
                other=0,
            ).to(tl.int32)
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
    scale = torch.sqrt((w**2).mean(dim=1, keepdim=True)).clamp(min=1e-5)
    w_norm = w / scale
    w_q = torch.round(w_norm).clamp(-1.0, 1.0).to(torch.int8)
    return w_q, scale.squeeze(1)


def reference_ternary_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference ternary linear path using plain torch ops.

    Note:
    - CUDA does not implement int32 addmm/matmul in this form.
    - For CUDA, we use fp32 matmul on quantized int8 values as a safe fallback.
    """
    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1]) if x.dim() > 2 else x
    x_q, x_scale = _quantize_activation(x_2d)
    w_q, w_scale = _quantize_weight(w)

    if x_q.is_cuda:
        # CUDA fallback: use fp32 GEMM on quantized values.
        out_acc = torch.matmul(x_q.to(torch.float32), w_q.t().to(torch.float32))
    else:
        out_acc = torch.matmul(x_q.to(torch.int32), w_q.t().to(torch.int32)).to(torch.float32)

    out_f = out_acc * (w_scale.view(1, -1) / x_scale)
    if bias is not None:
        out_f = out_f + bias

    if x.dim() > 2:
        out_f = out_f.view(*orig_shape[:-1], out_f.shape[-1])
    return out_f


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
    x_2d = x.view(-1, x.shape[-1]) if x.dim() > 2 else x

    x_q, x_scale = _quantize_activation(x_2d)
    w_q, w_scale = _quantize_weight(w)

    M, K = x_q.shape
    N = w_q.shape[0]

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

    out_f = out.float() * (w_scale.view(1, -1) / x_scale)
    if bias is not None:
        out_f = out_f + bias

    if x.dim() > 2:
        out_f = out_f.view(*orig_shape[:-1], out_f.shape[-1])
    return out_f


@dataclass
class BenchRow:
    shape: str
    mode: str
    ms: float
    tokens_per_s: float
    tflops: float
    max_abs_err: Optional[float]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench(fn, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        _ = fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    _sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / float(iters)


def _parse_shapes(raw: str) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        m, k, n = [int(x) for x in part.split("x")]
        shapes.append((m, k, n))
    return shapes


def _fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def run_benchmark(
    shapes: list[tuple[int, int, int]],
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    use_tensorcore: bool,
) -> list[BenchRow]:
    rows: list[BenchRow] = []

    triton_ok = is_triton_available() and device.type == "cuda"

    for (m, k, n) in shapes:
        x = torch.randn((m, k), device=device, dtype=dtype)
        w = torch.randn((n, k), device=device, dtype=dtype)
        b = torch.randn((n,), device=device, dtype=dtype)

        def _dense():
            return torch.nn.functional.linear(x, w, b)

        dense_ms = _bench(_dense, warmup=warmup, iters=iters, device=device)
        dense_tps = m / (dense_ms / 1000.0)
        dense_tflops = (2.0 * m * n * k) / (dense_ms / 1000.0) / 1e12
        rows.append(
            BenchRow(
                shape=f"{m}x{k}x{n}",
                mode="torch_dense_linear",
                ms=dense_ms,
                tokens_per_s=dense_tps,
                tflops=dense_tflops,
                max_abs_err=None,
            )
        )

        def _ref_ternary():
            return reference_ternary_linear(x, w, b)

        ref_ms = _bench(_ref_ternary, warmup=warmup, iters=iters, device=device)
        ref_out = _ref_ternary()
        ref_tps = m / (ref_ms / 1000.0)
        ref_tflops = (2.0 * m * n * k) / (ref_ms / 1000.0) / 1e12
        rows.append(
            BenchRow(
                shape=f"{m}x{k}x{n}",
                mode="torch_reference_ternary",
                ms=ref_ms,
                tokens_per_s=ref_tps,
                tflops=ref_tflops,
                # self-reference baseline: error is 0 by definition (ref vs itself),
                # not a measured value. Use None convention like dense row above.
                max_abs_err=None,
            )
        )

        if triton_ok:
            def _triton_ternary():
                return triton_ternary_linear(x, w, b, use_tensorcore=use_tensorcore)
            try:
                triton_ms = _bench(_triton_ternary, warmup=warmup, iters=iters, device=device)
                triton_out = _triton_ternary()
                err = (triton_out - ref_out).abs().max().item()
                triton_tps = m / (triton_ms / 1000.0)
                triton_tflops = (2.0 * m * n * k) / (triton_ms / 1000.0) / 1e12
                rows.append(
                    BenchRow(
                        shape=f"{m}x{k}x{n}",
                        mode="triton_ternary_kernel_tc" if use_tensorcore else "triton_ternary_kernel",
                        ms=triton_ms,
                        tokens_per_s=triton_tps,
                        tflops=triton_tflops,
                        max_abs_err=err,
                    )
                )
            except Exception as exc:
                # Keep benchmark running in notebook/Colab even if Triton JIT fails.
                mode = "triton_ternary_kernel_tc_skipped" if use_tensorcore else "triton_ternary_kernel_skipped"
                rows.append(
                    BenchRow(
                        shape=f"{m}x{k}x{n}",
                        mode=mode,
                        ms=0.0,
                        tokens_per_s=0.0,
                        tflops=0.0,
                        max_abs_err=None,
                    )
                )
                print(
                    f"warning: Triton kernel skipped for {m}x{k}x{n}: "
                    f"{type(exc).__name__}: {exc}"
                )
                # Disable Triton path for remaining shapes in this run.
                triton_ok = False

    return rows


def _print_rows(rows: list[BenchRow]) -> None:
    print("")
    print("| Shape (M x K x N) | Mode | Avg ms | Tokens/s | Effective TFLOPS | Max |Δ| vs ref |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r.shape} | {r.mode} | {_fmt(r.ms)} | {_fmt(r.tokens_per_s, 2)} | {_fmt(r.tflops, 3)} | {_fmt(r.max_abs_err, 6)} |"
        )
    print("")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Standalone BitNet ternary kernel benchmark")
    # Jupyter/Colab injects "-f <kernel.json>" to scripts; accept it silently.
    parser.add_argument("-f", "--ipykernel-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--shapes",
        default="2048x2048x2048,4096x2048x2048",
        help="Comma-separated list of MxKxN shapes",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=30, help="Measured iterations")
    parser.add_argument("--seed", type=int, default=1453, help="Random seed")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Benchmark device",
    )
    parser.add_argument(
        "--use-tensorcore",
        action="store_true",
        help="Use experimental tensor-core Triton kernel path",
    )
    # parse_known_args keeps the script resilient to notebook/runtime extra flags.
    args, _ = parser.parse_known_args(argv)

    torch.manual_seed(args.seed)

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    shapes = _parse_shapes(args.shapes)
    if not shapes:
        raise ValueError("No valid shape provided")

    print("== BitNet Kernel Standalone Benchmark ==")
    print(f"device={dev} dtype={dtype} triton_available={is_triton_available()} use_tensorcore={args.use_tensorcore}")
    print("note: benchmark is single-device; multi-GPU instances are not aggregated.")
    if dev.type != "cuda":
        print("note: CUDA is not available; Triton kernel rows will be skipped.")
    elif not is_triton_available():
        print("note: Triton is not available in this environment; Triton kernel rows will be skipped.")

    rows = run_benchmark(
        shapes=shapes,
        device=dev,
        dtype=dtype,
        warmup=args.warmup,
        iters=args.iters,
        use_tensorcore=args.use_tensorcore,
    )
    _print_rows(rows)


def run_default() -> None:
    """Run benchmark with built-in defaults (no CLI args required)."""
    main([])


if __name__ == "__main__":
    main()
