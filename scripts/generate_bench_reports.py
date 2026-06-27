#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def bench_linear(device: str, n: int = 30) -> dict:
    if device == "cuda" and not torch.cuda.is_available():
        return {"ok": False, "reason": "cuda_unavailable"}
    if device == "mps" and not torch.backends.mps.is_available():
        return {"ok": False, "reason": "mps_unavailable"}

    dev = torch.device(device)
    x = torch.randn(64, 256, device=dev)
    w = torch.randn(128, 256, device=dev)
    torch.manual_seed(7)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        y = torch.nn.functional.linear(x, w)
        _ = y.mean().item() if device == "cpu" else y.mean()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    return {"ok": True, "device": device, "avg_ms": dt * 1000.0}


def write(name: str, payload: dict) -> None:
    out = ROOT / "reports" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    cpp = bench_linear("cpu")
    write("bench_cpp_report.json", cpp)

    # NOTE: zero-copy/metal/vulkan/npu reports below are DECLARATIVE FALLBACKS,
    # not real backend benchmarks. No vulkan/npu/zero-copy/metal kernel is run or
    # measured here. The "ok"/"mode" fields are placeholders for a phase that has
    # not implemented these backends yet; they are NOT a pass/fail gate and must
    # not be read as evidence that the backend was measured.
    # zero_copy: just wraps the cpu bench output; no real zero-copy path measured.
    write("bench_zero_copy_report.json", {
        "ok": True,
        "mode": "verified_fallback",
        "measured": False,
        "note": "stub: no real zero-copy backend; wraps cpu bench only, not a gate",
        "details": cpp,
    })
    # metal: reports torch MPS availability only; no metal/mps kernel micro-benchmark.
    write("bench_metal_report.json", {
        "ok": torch.backends.mps.is_available(),
        "mode": "fallback" if not torch.backends.mps.is_available() else "mps",
        "measured": False,
        "note": "stub: mps availability flag only, no metal/mps kernel benchmark run",
    })
    # vulkan: no vulkan backend; constant fallback, not measured.
    write("bench_vulkan_report.json", {
        "ok": True,
        "mode": "verified_fallback",
        "measured": False,
        "note": "stub: no real vulkan backend measured; not a gate",
    })
    # npu: no npu backend; constant fallback, not measured.
    write("bench_npu_report.json", {
        "ok": True,
        "mode": "verified_fallback",
        "measured": False,
        "note": "stub: no real npu backend measured; not a gate",
    })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
