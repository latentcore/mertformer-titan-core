#!/usr/bin/env python3
"""
Device-agnostic micro-benchmark for the Liquid training implementations
(layers/liquid.py LIQUID_TRAIN_IMPLS: baseline / precompute_input / packed_pair /
packed_pair_compile). Builds a tiny MertFormer with use_liquid=True under each impl and times
forward+backward, reporting ms/step. Runs anywhere (CPU/MPS/CUDA); only meaningful for relative
comparison. An impl that errors is recorded, not fatal.

Usage:  python scripts/benchmark_liquid_impls.py --iters 20
Output: reports/benchmarks/liquid_impl_benchmark.json + a printed table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import cfg  # noqa: E402
from layers.liquid import LIQUID_TRAIN_IMPLS  # noqa: E402
from model.transformers import MertFormer  # noqa: E402

PATCH = dict(hidden_size=128, intermediate_size=256, num_layers=2, num_heads=4, num_kv_heads=2,
             head_dim=32, vocab_size=512, use_moe=True, num_experts=4, num_experts_per_tok=2,
             active_experts=2, use_liquid=True, liquid_layers_idx=[0], use_qinn=False,
             use_gradient_checkpointing=False)


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _sync(dev: str) -> None:
    if dev == "cuda":
        torch.cuda.synchronize()


def bench_impl(impl: str, iters: int, dev: str) -> dict:
    keys = list(PATCH) + ["device", "liquid_train_impl"]
    orig = {k: getattr(cfg, k) for k in keys if hasattr(cfg, k)}
    try:
        for k, v in PATCH.items():
            setattr(cfg, k, v)
        cfg.device = dev
        cfg.liquid_train_impl = impl
        torch.manual_seed(0)
        model = MertFormer().to(dev)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ce = nn.CrossEntropyLoss()
        x = torch.randint(0, 512, (2, 64), device=dev)
        y = torch.randint(0, 512, (2, 64), device=dev)
        for _ in range(3):  # warmup
            logits, aux, _ = model(x)
            (ce(logits.reshape(-1, 512), y.reshape(-1)) + aux.float()).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        _sync(dev)
        t0 = time.time()
        for _ in range(iters):
            logits, aux, _ = model(x)
            (ce(logits.reshape(-1, 512), y.reshape(-1)) + aux.float()).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        _sync(dev)
        return {"impl": impl, "ms_per_step": round((time.time() - t0) / iters * 1000, 2), "ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"impl": impl, "ms_per_step": None, "ok": False, "error": str(exc)}
    finally:
        for k, v in orig.items():
            setattr(cfg, k, v)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()
    dev = _device()
    print(f"[bench] device={dev} iters={args.iters}")
    results = [bench_impl(impl, args.iters, dev) for impl in sorted(LIQUID_TRAIN_IMPLS)]
    for r in results:
        print(f"  {r['impl']:24s} {'ERR: ' + r['error'][:50] if not r['ok'] else str(r['ms_per_step']) + ' ms/step'}")
    out = PROJECT_ROOT / "reports" / "benchmarks" / "liquid_impl_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": dev, "iters": args.iters, "results": results}, indent=2), encoding="utf-8")
    print(f"[bench] written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
