#!/usr/bin/env python3
"""Claim-safe Liquid-vs-GQA canonical-shape micro-benchmark.

Compares `layers.liquid.LiquidMixer` against `layers.mla.GQA` on synthetic
tensors, at the model's own canonical shape (`config.config.cfg` defaults:
hidden_size=2048, num_heads=16, num_kv_heads=8, head_dim=128), in two modes:

  1. Train-mode: forward+backward wall-clock (ms/step), swept across seq_len,
     for each Liquid `train_impl` variant. Tests whether the sequential
     recurrence's relative cost against GQA changes with seq_len.
  2. Decode-mode: single-token incremental step latency as ALREADY-CACHED
     context grows -- GQA is given a real, pre-filled KV cache of the target
     size (`past_key_value`, `use_cache=True`); LiquidMixer is given a
     constant-size `[B,H]` hidden state (`h_init`, `return_state=True`) and
     run in `.eval()` mode (the same quantized-cache inference path
     `MertFormer.generate()` actually uses). Tests whether GQA's per-step
     cost grows with cached context while Liquid's stays flat.

This is a speed probe only; it does not prove 45K runtime, model quality, or
benchmark readiness -- consumer GPU, single machine, no statistical replication.
See BACKLOG.md, "External signal on Liquid/CfC wall-clock cost."

Examples:
  python3 scripts/liquid_vs_gqa_canonical_benchmark.py --device auto
  python3 scripts/liquid_vs_gqa_canonical_benchmark.py --device cuda --batch-size 1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import cfg
from layers.liquid import LIQUID_TRAIN_IMPLS, LiquidMixer
from layers.mla import GQA


def _pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _peak_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024**2))


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)


# ---------------------------------------------------------------------------
# Train-mode: forward+backward ms/step, GQA vs LiquidMixer(train_impl=...)
# ---------------------------------------------------------------------------


def _bench_train_gqa(
    *, hidden: int, batch_size: int, seq_len: int, warmup: int, bench: int,
    device: torch.device, amp_dtype: str, seed: int,
) -> dict[str, Any]:
    _set_seed(seed)
    model = GQA().to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    gen = torch.Generator(device="cpu").manual_seed(seed + 17)
    x = torch.randn(batch_size, seq_len, hidden, generator=gen).to(device)
    target = torch.randn(batch_size, seq_len, hidden, generator=gen).to(device)

    def step() -> float:
        opt.zero_grad(set_to_none=True)
        with _amp_context(device, amp_dtype):
            y, _ = model(x)
            loss = ((y.float() - target.float()) ** 2).mean()
        loss.backward()
        opt.step()
        return float(loss.detach().cpu().item())

    _reset_peak(device)
    _sync(device)
    for _ in range(warmup):
        step()
    _sync(device)
    started = time.perf_counter()
    for _ in range(bench):
        step()
    _sync(device)
    elapsed = time.perf_counter() - started
    ms = elapsed * 1000.0 / max(bench, 1)
    return {
        "label": "gqa",
        "seq_len": seq_len,
        "ms_per_step": ms,
        "steps_per_sec": bench / max(elapsed, 1e-12),
        "peak_mem_mb": _peak_mb(device),
    }


def _bench_train_liquid(
    *, impl: str, hidden: int, batch_size: int, seq_len: int, warmup: int,
    bench: int, device: torch.device, amp_dtype: str, fast_path: bool, seed: int,
) -> dict[str, Any]:
    _set_seed(seed)
    model = LiquidMixer(hidden, fast_path=fast_path, train_impl=impl).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    gen = torch.Generator(device="cpu").manual_seed(seed + 17)
    x = torch.randn(batch_size, seq_len, hidden, generator=gen).to(device)
    target = torch.randn(batch_size, seq_len, hidden, generator=gen).to(device)

    def step() -> float:
        opt.zero_grad(set_to_none=True)
        with _amp_context(device, amp_dtype):
            y = model(x)
            loss = ((y.float() - target.float()) ** 2).mean()
        loss.backward()
        opt.step()
        return float(loss.detach().cpu().item())

    _reset_peak(device)
    _sync(device)
    for _ in range(warmup):
        step()
    _sync(device)
    started = time.perf_counter()
    for _ in range(bench):
        step()
    _sync(device)
    elapsed = time.perf_counter() - started
    ms = elapsed * 1000.0 / max(bench, 1)
    return {
        "label": f"liquid[{impl}]",
        "seq_len": seq_len,
        "ms_per_step": ms,
        "steps_per_sec": bench / max(elapsed, 1e-12),
        "peak_mem_mb": _peak_mb(device),
    }


# ---------------------------------------------------------------------------
# Decode-mode: single-token step latency vs already-cached context size
# ---------------------------------------------------------------------------


def _bench_decode_gqa(
    *, hidden: int, batch_size: int, context_len: int, warmup: int, bench: int,
    device: torch.device, seed: int,
) -> dict[str, Any]:
    _set_seed(seed)
    model = GQA().to(device).eval()
    num_kv_heads = int(cfg.num_kv_heads)
    head_dim = int(cfg.head_dim)
    gen = torch.Generator(device="cpu").manual_seed(seed + 29)
    past_k = torch.randn(batch_size, num_kv_heads, context_len, head_dim, generator=gen).to(device)
    past_v = torch.randn(batch_size, num_kv_heads, context_len, head_dim, generator=gen).to(device)
    x_t = torch.randn(batch_size, 1, hidden, generator=gen).to(device)

    def step() -> None:
        with torch.no_grad():
            model(x_t, past_key_value=(past_k, past_v), use_cache=True)

    _sync(device)
    for _ in range(warmup):
        step()
    _sync(device)
    started = time.perf_counter()
    for _ in range(bench):
        step()
    _sync(device)
    elapsed = time.perf_counter() - started
    ms = elapsed * 1000.0 / max(bench, 1)
    return {"label": "gqa", "context_len": context_len, "ms_per_token": ms}


def _bench_decode_liquid(
    *, impl: str, hidden: int, batch_size: int, context_len: int, warmup: int,
    bench: int, device: torch.device, fast_path: bool, seed: int,
) -> dict[str, Any]:
    _set_seed(seed)
    model = LiquidMixer(hidden, fast_path=fast_path, train_impl=impl).to(device).eval()
    gen = torch.Generator(device="cpu").manual_seed(seed + 29)
    # context_len is informational only here: LiquidMixer's per-step cost is a
    # function of hidden size alone, not of how many prior tokens produced h --
    # that is exactly the O(1)-per-token hypothesis under test.
    h = torch.randn(batch_size, hidden, generator=gen).to(device)
    x_t = torch.randn(batch_size, 1, hidden, generator=gen).to(device)

    def step() -> None:
        with torch.no_grad():
            model(x_t, h_init=h, return_state=True)

    _sync(device)
    for _ in range(warmup):
        step()
    _sync(device)
    started = time.perf_counter()
    for _ in range(bench):
        step()
    _sync(device)
    elapsed = time.perf_counter() - started
    ms = elapsed * 1000.0 / max(bench, 1)
    return {"label": f"liquid[{impl}]", "context_len": context_len, "ms_per_token": ms}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-seq-lens", default="512,4096")
    parser.add_argument("--decode-context-lens", default="128,1024,3072")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--bench", type=int, default=20)
    parser.add_argument("--decode-warmup", type=int, default=10)
    parser.add_argument("--decode-bench", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1453)
    parser.add_argument("--fast-path", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--liquid-modes",
        default="baseline,packed_pair,packed_pair_compile",
        help=f"Comma-separated implementations; choices: {','.join(sorted(LIQUID_TRAIN_IMPLS))}",
    )
    parser.add_argument("--amp-dtype", default="bf16", choices=["none", "bf16", "fp16"])
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    device = _pick_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.amp_dtype == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        args.amp_dtype = "fp16"
    if device.type != "cuda":
        args.amp_dtype = "none"

    liquid_modes = [m.strip() for m in args.liquid_modes.split(",") if m.strip()]
    unknown = sorted(set(liquid_modes) - LIQUID_TRAIN_IMPLS)
    if unknown:
        raise SystemExit(f"Unknown liquid modes: {unknown}")

    hidden = int(cfg.hidden_size)
    train_seq_lens = [int(v) for v in args.train_seq_lens.split(",") if v.strip()]
    decode_context_lens = [int(v) for v in args.decode_context_lens.split(",") if v.strip()]

    config_header = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "amp_dtype": args.amp_dtype,
        "fast_path": bool(args.fast_path),
        "canonical_shape": {
            "hidden_size": hidden,
            "num_heads": int(cfg.num_heads),
            "num_kv_heads": int(cfg.num_kv_heads),
            "head_dim": int(cfg.head_dim),
            "max_seq_len": int(cfg.max_seq_len),
        },
        "batch_size": args.batch_size,
        "train_seq_lens": train_seq_lens,
        "decode_context_lens": decode_context_lens,
        "liquid_modes": liquid_modes,
    }
    print("=" * 100)
    print("MertFormer Liquid-vs-GQA canonical-shape benchmark")
    print("=" * 100)
    print(json.dumps(config_header, indent=2))

    train_rows: list[dict[str, Any]] = []
    for seq_len in train_seq_lens:
        print(f"\n[train-mode] seq_len={seq_len}")
        row = _bench_train_gqa(
            hidden=hidden, batch_size=args.batch_size, seq_len=seq_len,
            warmup=args.warmup, bench=args.bench, device=device,
            amp_dtype=args.amp_dtype, seed=args.seed,
        )
        train_rows.append(row)
        print(f"  {row['label']:<24} ms/step={row['ms_per_step']:.3f}")
        for impl in liquid_modes:
            row = _bench_train_liquid(
                impl=impl, hidden=hidden, batch_size=args.batch_size, seq_len=seq_len,
                warmup=args.warmup, bench=args.bench, device=device,
                amp_dtype=args.amp_dtype, fast_path=bool(args.fast_path), seed=args.seed,
            )
            train_rows.append(row)
            print(f"  {row['label']:<24} ms/step={row['ms_per_step']:.3f}")

    decode_rows: list[dict[str, Any]] = []
    for context_len in decode_context_lens:
        print(f"\n[decode-mode] context_len={context_len}")
        row = _bench_decode_gqa(
            hidden=hidden, batch_size=args.batch_size, context_len=context_len,
            warmup=args.decode_warmup, bench=args.decode_bench, device=device, seed=args.seed,
        )
        decode_rows.append(row)
        print(f"  {row['label']:<24} ms/token={row['ms_per_token']:.4f}")
        for impl in liquid_modes:
            row = _bench_decode_liquid(
                impl=impl, hidden=hidden, batch_size=args.batch_size, context_len=context_len,
                warmup=args.decode_warmup, bench=args.decode_bench, device=device,
                fast_path=bool(args.fast_path), seed=args.seed,
            )
            decode_rows.append(row)
            print(f"  {row['label']:<24} ms/token={row['ms_per_token']:.4f}")

    payload = {"config": config_header, "train_mode": train_rows, "decode_mode": decode_rows}
    print("\nRaw JSON:")
    print(json.dumps(payload, indent=2))
    print(
        "\nClaim boundary: component-level microbench on consumer GPU, single run, "
        "no statistical replication -- informational only, does not change the "
        "DECIDED: Keep call in reports/liquid_keep_or_drop_brief.md."
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
