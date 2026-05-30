#!/usr/bin/env python3
"""Claim-safe Liquid training implementation benchmark.

This script compares the real repo `layers.liquid.LiquidMixer` training
implementations on synthetic tensors. It is a speed/equivalence probe only; it
does not prove 45K runtime, model quality, or benchmark readiness.

Examples:
  python3 scripts/liquid_train_impl_benchmark.py --device auto
  TITAN_LIQUID_TRAIN_IMPL=packed_pair python3 scripts/liquid_train_impl_benchmark.py
  python3 scripts/liquid_train_impl_benchmark.py --device cuda --hidden 256 --seq-len 128 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from layers.liquid import LIQUID_TRAIN_IMPLS, LiquidMixer


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


def _finite_grads(model: torch.nn.Module) -> bool:
    seen = False
    for param in model.parameters():
        if param.grad is None:
            continue
        seen = True
        if not torch.isfinite(param.grad).all():
            return False
    return seen


def _run_once(
    model: LiquidMixer,
    opt: torch.optim.Optimizer,
    x: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
    check_grad: bool,
) -> tuple[float, bool]:
    opt.zero_grad(set_to_none=True)
    with _amp_context(device, amp_dtype):
        y = model(x)
        loss = ((y.float() - target.float()) ** 2).mean()
    loss.backward()
    grad_ok = _finite_grads(model) if check_grad else True
    opt.step()
    return float(loss.detach().cpu().item()), bool(grad_ok)


def _run_mode(
    *,
    impl: str,
    base_state: dict[str, torch.Tensor],
    hidden: int,
    batch_size: int,
    seq_len: int,
    warmup: int,
    bench: int,
    device: torch.device,
    amp_dtype: str,
    fast_path: bool,
    seed: int,
) -> dict[str, Any]:
    _set_seed(seed)
    model = LiquidMixer(hidden, fast_path=fast_path, train_impl=impl).to(device)
    model.load_state_dict(base_state)
    model.train()

    gen = torch.Generator(device="cpu").manual_seed(seed + 17)
    x = torch.randn(batch_size, seq_len, hidden, generator=gen).to(device)
    target = torch.randn(batch_size, seq_len, hidden, generator=gen).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    _reset_peak(device)
    _sync(device)

    loss_start = None
    grad_ok = False
    for idx in range(warmup):
        loss, ok = _run_once(model, opt, x, target, device, amp_dtype, check_grad=(idx == 0))
        if loss_start is None:
            loss_start = loss
        if idx == 0:
            grad_ok = ok

    _sync(device)
    started = time.perf_counter()
    loss_end = None
    for _ in range(bench):
        loss_end, _ = _run_once(model, opt, x, target, device, amp_dtype, check_grad=False)
    _sync(device)
    elapsed = time.perf_counter() - started

    ms = elapsed * 1000.0 / max(bench, 1)
    steps_s = bench / max(elapsed, 1e-12)
    tokens_s = steps_s * batch_size * seq_len
    return {
        "label": impl,
        "ms_per_step": ms,
        "steps_per_sec": steps_s,
        "tokens_per_sec": tokens_s,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "grad_ok": grad_ok,
        "peak_mem_mb": _peak_mb(device),
    }


def _equivalence_probe(hidden: int, seed: int) -> dict[str, Any]:
    _set_seed(seed)
    base = LiquidMixer(hidden, fast_path=False, train_impl="baseline").double()
    base.train()
    x = torch.randn(2, 9, hidden, dtype=torch.float64)

    def run(model: LiquidMixer):
        x1 = x.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        y = model(x1)
        loss = (y.float() ** 2).mean()
        loss.backward()
        grads = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        return y.detach(), x1.grad.detach().clone(), grads

    y0, xg0, pg0 = run(base)
    out: dict[str, Any] = {}
    for impl in ("precompute_input", "packed_pair"):
        candidate = LiquidMixer(hidden, fast_path=False, train_impl=impl).double()
        candidate.load_state_dict(base.state_dict())
        candidate.train()
        y1, xg1, pg1 = run(candidate)
        out_diff = torch.max(torch.abs(y0 - y1)).item()
        x_grad_diff = torch.max(torch.abs(xg0 - xg1)).item()
        p_grad_diff = max(
            torch.max(torch.abs(pg0[name] - pg1[name])).item()
            for name in pg0
        )
        out[impl] = {
            "out_diff": out_diff,
            "x_grad_diff": x_grad_diff,
            "param_grad_diff": p_grad_diff,
            "ok": out_diff < 1e-8 and x_grad_diff < 1e-8 and p_grad_diff < 1e-8,
        }
    return out


def _print_table(rows: list[dict[str, Any]]) -> None:
    baseline_ms = rows[0]["ms_per_step"]
    print("\nRESULT TABLE")
    print("-" * 112)
    print(
        f"{'impl':<24} {'ms/step':>10} {'steps/s':>10} {'tok/s':>12} "
        f"{'loss_s':>10} {'loss_e':>10} {'grad':>6} {'mem':>8} {'speedup':>9}"
    )
    print("-" * 112)
    for row in rows:
        speedup = baseline_ms / max(float(row["ms_per_step"]), 1e-12)
        print(
            f"{row['label']:<24} {row['ms_per_step']:>10.3f} "
            f"{row['steps_per_sec']:>10.3f} {row['tokens_per_sec']:>12.0f} "
            f"{row['loss_start']:>10.4f} {row['loss_end']:>10.4f} "
            f"{str(row['grad_ok']):>6} {row['peak_mem_mb']:>8.1f} {speedup:>8.3f}x"
        )
    print("-" * 112)
    best = min(rows, key=lambda item: item["ms_per_step"])
    best_speedup = baseline_ms / max(float(best["ms_per_step"]), 1e-12)
    print(f"BEST: {best['label']} = {best_speedup:.3f}x")
    print("Claim boundary: Liquid microbench only; 45K/H200 repo path still needs full measurement.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--equiv-hidden", type=int, default=0, help="CPU double equivalence hidden size; 0=min(hidden,256).")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--bench", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1453)
    parser.add_argument("--fast-path", type=int, default=1, choices=[0, 1])
    parser.add_argument("--amp-dtype", default="bf16", choices=["none", "bf16", "fp16"])
    parser.add_argument(
        "--modes",
        default="baseline,precompute_input,packed_pair,packed_pair_compile",
        help=f"Comma-separated implementations; choices: {','.join(sorted(LIQUID_TRAIN_IMPLS))}",
    )
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

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    unknown = sorted(set(modes) - LIQUID_TRAIN_IMPLS)
    if unknown:
        raise SystemExit(f"Unknown modes: {unknown}")
    if "baseline" not in modes:
        modes.insert(0, "baseline")

    print("=" * 100)
    print("MertFormer Liquid train-impl benchmark")
    print("=" * 100)
    equiv_hidden = args.equiv_hidden if args.equiv_hidden > 0 else min(args.hidden, 256)

    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "device": str(device),
                "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "amp_dtype": args.amp_dtype,
                "fast_path": bool(args.fast_path),
                "shape": {
                    "batch": args.batch_size,
                    "seq_len": args.seq_len,
                    "hidden": args.hidden,
                },
                "warmup": args.warmup,
                "bench": args.bench,
                "equiv_hidden": equiv_hidden,
            },
            indent=2,
        )
    )

    eq = _equivalence_probe(equiv_hidden, args.seed)
    print("\nEXACT-MATH EQUIVALENCE PROBE")
    print("-" * 88)
    for name, row in eq.items():
        print(
            f"{name:<20} out={row['out_diff']:.3e} "
            f"xgrad={row['x_grad_diff']:.3e} pgrad={row['param_grad_diff']:.3e} ok={row['ok']}"
        )

    _set_seed(args.seed)
    base_model = LiquidMixer(args.hidden, fast_path=bool(args.fast_path), train_impl="baseline")
    base_state = {name: value.detach().clone() for name, value in base_model.state_dict().items()}

    rows = [
        _run_mode(
            impl=impl,
            base_state=base_state,
            hidden=args.hidden,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            warmup=args.warmup,
            bench=args.bench,
            device=device,
            amp_dtype=args.amp_dtype,
            fast_path=bool(args.fast_path),
            seed=args.seed,
        )
        for impl in modes
    ]
    _print_table(rows)

    payload = {"equivalence": eq, "rows": rows}
    print("\nRaw JSON:")
    print(json.dumps(payload, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote: {out}")
    if not all(row.get("grad_ok") for row in rows):
        return 2
    if not all(item.get("ok") for item in eq.values()):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
