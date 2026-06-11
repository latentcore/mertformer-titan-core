"""
Telemetry helpers for expected-vs-actual tracking and system snapshots.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

GPU_MEMORY_FIELDS: Tuple[str, ...] = (
    "gpu_device",
    "gpu_allocated_gb",
    "gpu_reserved_gb",
    "gpu_total_gb",
)

NVIDIA_SMI_FIELDS: Tuple[str, ...] = (
    "gpu_util_percent",
    "gpu_temp_c",
    "gpu_power_w",
    "gpu_power_limit_w",
    "gpu_mem_used_gb_smi",
    "gpu_mem_total_gb_smi",
)

DISK_USAGE_FIELDS: Tuple[str, ...] = (
    "disk_used_gb",
    "disk_free_gb",
    "disk_total_gb",
)

SYSTEM_SNAPSHOT_FIELDS: Tuple[str, ...] = (
    "timestamp_utc",
    "cpu_percent",
    "ram_used_gb",
    "ram_total_gb",
    *GPU_MEMORY_FIELDS,
    *NVIDIA_SMI_FIELDS,
    *DISK_USAGE_FIELDS,
)


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


def nvidia_smi_snapshot() -> Dict[str, Optional[float]]:
    if shutil.which("nvidia-smi") is None:
        return {}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,power.draw,power.limit,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if proc.returncode != 0:
            return {}
        rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not rows:
            return {}

        device_index = 0
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                device_index = int(torch.cuda.current_device())
            except Exception:
                device_index = 0
        if device_index >= len(rows):
            device_index = 0

        parts = [p.strip() for p in rows[device_index].split(",")]
        if len(parts) < 6:
            return {}

        def parse_float(value: str) -> Optional[float]:
            if not value or value.upper() in {"N/A", "[N/A]"}:
                return None
            try:
                return float(value)
            except Exception:
                return None

        mem_used_mb = parse_float(parts[4])
        mem_total_mb = parse_float(parts[5])
        return {
            "gpu_util_percent": parse_float(parts[0]),
            "gpu_temp_c": parse_float(parts[1]),
            "gpu_power_w": parse_float(parts[2]),
            "gpu_power_limit_w": parse_float(parts[3]),
            "gpu_mem_used_gb_smi": (mem_used_mb / 1024.0) if mem_used_mb is not None else None,
            "gpu_mem_total_gb_smi": (mem_total_mb / 1024.0) if mem_total_mb is not None else None,
        }
    except Exception:
        return {}


def disk_usage_gb(path: str | Path | None = None) -> Dict[str, Optional[float]]:
    target = Path(path).resolve() if path is not None else Path.cwd().resolve()
    try:
        usage = shutil.disk_usage(target)
        return {
            "disk_used_gb": float(usage.used / (1024 ** 3)),
            "disk_free_gb": float(usage.free / (1024 ** 3)),
            "disk_total_gb": float(usage.total / (1024 ** 3)),
        }
    except Exception:
        return {
            "disk_used_gb": None,
            "disk_free_gb": None,
            "disk_total_gb": None,
        }


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


def system_snapshot(path: str | Path | None = None) -> Dict[str, float | str | None]:
    snapshot: Dict[str, float | str | None] = {field: None for field in SYSTEM_SNAPSHOT_FIELDS}
    snapshot["timestamp_utc"] = utc_timestamp()
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
    snapshot.update(disk_usage_gb(path))
    snapshot.update(nvidia_smi_snapshot())
    return snapshot


def runtime_health_report(
    *,
    snapshot: Dict[str, Optional[float]],
    verification_confidence: float,
    failure_budget_signal: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Build a compact SLA-style runtime health report."""
    cpu = float(snapshot.get("cpu_percent") or 0.0)
    ram_total = float(snapshot.get("ram_total_gb") or 0.0)
    ram_used = float(snapshot.get("ram_used_gb") or 0.0)
    ram_ratio = (ram_used / ram_total) if ram_total > 0 else 0.0
    fb = failure_budget_signal or {}
    hours_since_progress = float(fb.get("hours_since_progress", 0.0))
    should_pivot = float(fb.get("should_pivot", 0.0))
    return {
        "verification_confidence": float(verification_confidence),
        "cpu_percent": cpu,
        "ram_usage_ratio": ram_ratio,
        "hours_since_progress": hours_since_progress,
        "pivot_signal": should_pivot,
    }


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
