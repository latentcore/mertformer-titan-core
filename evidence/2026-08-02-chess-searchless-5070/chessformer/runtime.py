"""Device selection, determinism, precision policy and environment capture.

Also carries the Blackwell guard. The delivery target is an RTX 5070 Laptop
(Blackwell, compute capability 12.0). PyTorch did not ship ``sm_120`` kernels
until 2.7 + CUDA 12.8; a 2.6/cu124 build sees the GPU, reports it as available,
and then fails at the first kernel launch with "no kernel image is available for
execution on the device". :func:`check_gpu_compatibility` detects that
combination up front and says exactly what to install, instead of letting a
48-hour run die on its first step.
"""
from __future__ import annotations

import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

MIN_TORCH_FOR_SM120 = (2, 7)


@dataclass
class DeviceInfo:
    device: torch.device
    kind: str                      # "cuda" | "cpu" | "mps"
    name: str
    total_vram_bytes: int
    compute_capability: Optional[str]
    supports_bf16: bool
    torch_version: str
    cuda_version: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "kind": self.kind,
            "name": self.name,
            "total_vram_bytes": int(self.total_vram_bytes),
            "total_vram_gb": round(self.total_vram_bytes / (1024 ** 3), 3),
            "compute_capability": self.compute_capability,
            "supports_bf16": bool(self.supports_bf16),
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
        }


def pick_device(preference: str = "auto") -> DeviceInfo:
    pref = (preference or "auto").lower()
    use_cuda = torch.cuda.is_available() and pref in {"auto", "cuda"}

    if use_cuda:
        props = torch.cuda.get_device_properties(0)
        cc = f"{props.major}.{props.minor}"
        return DeviceInfo(
            device=torch.device("cuda"),
            kind="cuda",
            name=props.name,
            total_vram_bytes=int(props.total_memory),
            compute_capability=cc,
            supports_bf16=bool(torch.cuda.is_bf16_supported()),
            torch_version=torch.__version__,
            cuda_version=getattr(torch.version, "cuda", None),
        )

    if pref in {"auto", "mps"} and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return DeviceInfo(
            device=torch.device("mps"), kind="mps", name="Apple MPS",
            total_vram_bytes=0, compute_capability=None, supports_bf16=False,
            torch_version=torch.__version__, cuda_version=None,
        )

    import psutil  # optional but present in requirements

    try:
        ram = int(psutil.virtual_memory().total)
    except Exception:
        ram = 0
    return DeviceInfo(
        device=torch.device("cpu"), kind="cpu", name=platform.processor() or "cpu",
        total_vram_bytes=ram, compute_capability=None, supports_bf16=False,
        torch_version=torch.__version__, cuda_version=None,
    )


def check_gpu_compatibility(info: DeviceInfo) -> Dict[str, Any]:
    """Verify the installed torch build actually has kernels for this GPU."""
    result: Dict[str, Any] = {"ok": True, "problems": [], "advice": []}
    if info.kind != "cuda" or not info.compute_capability:
        return result

    try:
        major, minor = (int(x) for x in info.compute_capability.split("."))
    except ValueError:
        return result

    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        arch_list = []
    result["torch_arch_list"] = arch_list

    # CUDA cubin compatibility rule: a binary built for sm_XY runs on any
    # sm_XZ with the SAME major X and Z >= Y. So a torch built for sm_86
    # runs fine on sm_89 (Ada) even though "sm_89" never appears in the list.
    # Only a missing *major* is a hard failure.
    covered = False
    best_same_major: Optional[int] = None
    for tag in arch_list:
        if not tag.startswith("sm_"):
            continue
        digits = tag[3:]
        if not digits.isdigit():
            continue
        a_major, a_minor = int(digits[:-1]), int(digits[-1])
        if a_major != major:
            continue
        best_same_major = max(best_same_major or 0, a_minor)
        if a_minor <= minor:
            covered = True

    sm_tag = f"sm_{major}{minor}"
    hard_incompatible = False
    if arch_list and not covered:
        result["ok"] = False
        if best_same_major is None:
            detail = f"no kernels for compute-capability {major}.x at all"
            # A missing GPU *generation* cannot be rescued by cubin
            # forward-compatibility; this verdict is definitive.
            hard_incompatible = True
        else:
            detail = (
                f"the newest same-generation kernels are sm_{major}{best_same_major}, "
                "which is newer than this device"
            )
        result["problems"].append(
            f"installed torch {torch.__version__} cannot run {sm_tag}: {detail} "
            f"(built for: {', '.join(arch_list)})"
        )
    result["hard_incompatible"] = hard_incompatible

    torch_major_minor = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if major >= 12 and torch_major_minor < MIN_TORCH_FOR_SM120:
        result["ok"] = False
        result["problems"].append(
            f"{info.name} is compute capability {info.compute_capability} (Blackwell); "
            f"torch {torch.__version__} predates sm_120 support (needs >= 2.7 with CUDA 12.8)"
        )
        result["advice"].append(
            "pip install --upgrade --force-reinstall torch "
            "--index-url https://download.pytorch.org/whl/cu128"
        )

    if not result["ok"] and not result["advice"]:
        result["advice"].append(
            "reinstall torch from the CUDA build matching this GPU: "
            "https://pytorch.org/get-started/locally/"
        )

    # A real kernel launch is the authority; the static analysis above only
    # explains *why* when it fails.
    probe = verify_gpu_actually_computes(info)
    result["kernel_probe"] = probe
    if probe.get("ok") and not result["ok"] and not result["hard_incompatible"]:
        result["ok"] = True
        result["problems"].append(
            "static arch check was pessimistic (cubin minor-version forward compatibility), "
            "but a real kernel launch succeeded; treating the device as usable"
        )
    elif not probe.get("ok", True):
        result["ok"] = False
        result["problems"].append(f"kernel launch failed: {probe.get('error')}")
    return result


def verify_gpu_actually_computes(info: DeviceInfo) -> Dict[str, Any]:
    """Launch one real kernel. Catches 'available but unusable' GPUs."""
    if info.kind != "cuda":
        return {"ok": True, "skipped": "not a cuda device"}
    try:
        a = torch.randn(64, 64, device=info.device)
        b = torch.randn(64, 64, device=info.device)
        c = (a @ b).sum().item()
        return {"ok": bool(np.isfinite(c))}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def configure_precision(use_tf32: bool, deterministic: bool) -> Dict[str, Any]:
    """Set the matmul precision policy.

    The upstream onefile defaulted ``determinism_strict=True``, which sets
    ``allow_tf32 = not strict`` -- i.e. it turned TF32 **off** and enabled
    ``use_deterministic_algorithms`` for every run. On Ada/Blackwell that gives
    up a large fraction of matmul throughput by default. Here speed is the
    default and determinism is an explicit opt-in.
    """
    applied: Dict[str, Any] = {"requested_tf32": bool(use_tf32), "deterministic": bool(deterministic)}
    tf32 = bool(use_tf32) and not deterministic

    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = bool(deterministic)
    try:
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
    except Exception:
        pass

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:
            applied["deterministic_error"] = str(exc)
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    applied["tf32_enabled"] = tf32
    applied["cudnn_benchmark"] = not deterministic
    return applied


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> List[str]:
    """Restore RNG state. Returns the names of any streams that failed.

    Failures are surfaced rather than swallowed: a silently un-restored RNG
    makes a resumed run non-reproducible while still looking healthy.
    """
    failed: List[str] = []
    try:
        random.setstate(state["python"])
    except Exception:
        failed.append("python")
    try:
        np.random.set_state(state["numpy"])
    except Exception:
        failed.append("numpy")
    try:
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    except Exception:
        failed.append("torch")
    if torch.cuda.is_available() and "cuda" in state:
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:
            failed.append("cuda")
    return failed


def nvidia_smi_query() -> Dict[str, Any]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return {"available": False}
    try:
        out = subprocess.check_output(
            [binary, "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", timeout=10,
        ).strip()
    except Exception as exc:
        return {"available": True, "error": f"{type(exc).__name__}: {exc}"}
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append({
                "name": parts[0], "memory_total": parts[1],
                "driver_version": parts[2], "compute_cap": parts[3],
            })
    return {"available": True, "gpus": rows}


def environment_snapshot(info: Optional[DeviceInfo] = None) -> Dict[str, Any]:
    """Everything needed to interpret a measured number later."""
    info = info or pick_device("auto")
    import importlib.metadata as md

    def version(pkg: str) -> str:
        try:
            return md.version(pkg)
        except Exception:
            return "missing"

    snapshot: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": {
            name: version(name)
            for name in ("torch", "numpy", "zstandard", "chess", "psutil")
        },
        "device": info.to_dict(),
        "gpu_compatibility": check_gpu_compatibility(info),
        "nvidia_smi": nvidia_smi_query(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        snapshot["ram_total_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        snapshot["ram_total_gb"] = None
    return snapshot


def default_num_workers(requested: int) -> int:
    """Auto worker count. Windows spawn is expensive, so stay modest."""
    if requested >= 0:
        return int(requested)
    cpus = os.cpu_count() or 4
    cap = 8 if platform.system() != "Windows" else 6
    return max(2, min(cap, cpus // 2))


def cuda_memory_report(device: torch.device) -> Dict[str, Any]:
    if device.type != "cuda":
        return {}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 ** 2), 1),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / (1024 ** 2), 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1),
    }
