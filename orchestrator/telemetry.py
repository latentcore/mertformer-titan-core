"""
Telemetry helpers for expected-vs-actual tracking and system snapshots.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def gpu_memory_gb() -> Optional[Dict[str, float]]:
    if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return None
    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total_gb = props.total_memory / (1024 ** 3)
        allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
        return {
            "device": float(device),
            "allocated_gb": float(allocated_gb),
            "reserved_gb": float(reserved_gb),
            "total_gb": float(total_gb),
        }
    except Exception:
        return None


def router_entropy(loads: Iterable[float]) -> Optional[float]:
    values = [float(x) for x in loads if x is not None]
    if not values:
        return None
    total = sum(values)
    if total <= 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    if not probs:
        return 0.0
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def system_snapshot() -> Dict[str, Optional[float]]:
    snapshot: Dict[str, Optional[float]] = {
        "timestamp_utc": utc_timestamp(),
        "cpu_percent": None,
        "ram_used_gb": None,
        "ram_total_gb": None,
    }
    if psutil is not None:
        try:
            snapshot["cpu_percent"] = float(psutil.cpu_percent(interval=None))
            mem = psutil.virtual_memory()
            snapshot["ram_used_gb"] = float(mem.used / (1024 ** 3))
            snapshot["ram_total_gb"] = float(mem.total / (1024 ** 3))
        except Exception:
            pass
    gpu = gpu_memory_gb()
    if gpu:
        snapshot.update({f"gpu_{k}": v for k, v in gpu.items()})
    return snapshot


@dataclass
class LossSlopeTracker:
    window: int = 20
    history: List[Tuple[float, float]] = field(default_factory=list)

    def update(self, loss: float, timestamp: Optional[float] = None) -> None:
        ts = float(timestamp if timestamp is not None else time.time())
        self.history.append((ts, float(loss)))
        if len(self.history) > self.window:
            self.history = self.history[-self.window :]

    def slope_per_hour(self) -> Optional[float]:
        if len(self.history) < 2:
            return None
        xs = [h[0] for h in self.history]
        ys = [h[1] for h in self.history]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denom = sum((x - x_mean) ** 2 for x in xs)
        if denom <= 0:
            return None
        slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
        return float(slope * 3600.0)


@dataclass
class ExpectedVsActual:
    expected: Dict[str, float]
    tolerance: Dict[str, float]

    def compare(self, actual: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        report: Dict[str, Dict[str, float]] = {}
        for key, target in self.expected.items():
            if key not in actual:
                continue
            value = float(actual[key])
            tol = float(self.tolerance.get(key, 0.0))
            delta = value - target
            ok = abs(delta) <= tol
            report[key] = {
                "target": target,
                "actual": value,
                "delta": delta,
                "tolerance": tol,
                "ok": 1.0 if ok else 0.0,
            }
        return report
