"""
scripts/ddp_smoke.py -- a short, real, pre-spend 2-GPU DDP smoke test.

Context (see BACKLOG.md, B8, "PROPOSED (2026-07-25)"): today the only DDP-rank-sync
assertion in the pipeline (`train/train.py`'s "[Gate 3]" check) fires *inside* the real
training run, at the step-10000-class Liquid-unfreeze event -- i.e. after real budget has
already been spent getting there. This module provides a cheap, pre-spend equivalent: it
launches a short `accelerate launch --num_processes 2 ... scripts/preflight_run.py` and
polls real GPU utilization WHILE the subprocess is alive, only reporting genuine dual-GPU
activity confirmed if at least one in-flight sample showed BOTH GPUs active. Ambiguous or
inconclusive always resolves to "not confirmed" rather than a false positive -- a false
"DDP works" claim is worse than an honest "could not confirm" here, since nothing downstream
should ever treat this as a substitute for the real Gate 3 assertion during the actual run.

This is an independent implementation, not a refactor of
`scripts/kaggle_batch_runner.py::ddp_smoke_test()` (which has the same design and was built
and bug-fixed earlier in this project, entirely for the Kaggle batch orchestrator's own use).
Deliberately NOT sharing code with that file to avoid regression risk to its existing test
suite this close to a real launch -- the same accepted-duplication class already documented
elsewhere in BACKLOG.md (e.g. the `RMSNorm`/`_QKRMSNorm` note).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_POLL_INTERVAL_SECONDS = 10.0
DEFAULT_GRACE_SECONDS = 60.0


def gpu_utilization_snapshot() -> list:
    """Best-effort per-GPU utilization %, via nvidia-smi. Empty list if unavailable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=15,
        )
        if out.returncode != 0:
            return []
        vals = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                vals.append(int(parts[1]))
        return vals
    except Exception:
        return []


def detect_gpu_count() -> int:
    """Best-effort GPU count: torch.cuda if importable, else `nvidia-smi -L`, else 0."""
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
        if out.returncode == 0:
            return len([ln for ln in out.stdout.splitlines() if ln.strip()])
    except Exception:
        pass
    return 0


def run_ddp_smoke_test(
    *,
    gpu_count: Optional[int] = None,
    target_script: str = "scripts/preflight_run.py",
    cwd: Optional[Path] = None,
    budget_seconds: Optional[float] = None,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
    log_path: Optional[Path] = None,
) -> dict:
    """Runs the short real DDP smoke subprocess and returns a decisive structured result.

    Returns a dict with at least: gpu_count, skipped, attempted, status
    ("skipped_not_2_gpu" | "completed" | "exited_error" | "timed_out" | "exception"),
    both_active_any_sample, samples, wall_seconds, ok, error.

    `ok` is True only when exactly 2 GPUs were detected, the subprocess did not exit
    with a nonzero code, and at least one poll sample showed both GPUs genuinely active.
    """
    if gpu_count is None:
        gpu_count = detect_gpu_count()
    if budget_seconds is None:
        budget_seconds = float(os.environ.get("MERTFORMER_DDP_SMOKE_SECONDS", "240"))
    if cwd is None:
        cwd = ROOT

    if gpu_count != 2:
        return {
            "gpu_count": gpu_count,
            "skipped": True,
            "attempted": False,
            "status": "skipped_not_2_gpu",
            "both_active_any_sample": False,
            "samples": [],
            "wall_seconds": 0.0,
            "ok": False,
            "error": None,
        }

    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        "2",
        "--num_machines",
        "1",
        "--mixed_precision",
        "bf16",
        target_script,
    ]

    t0 = time.time()
    samples: list = []
    exited_on_own = False
    returncode = None

    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            lf = log_path.open("w", encoding="utf-8", errors="replace")
        else:
            lf = subprocess.DEVNULL

        try:
            if log_path is not None:
                lf.write(f"# cmd: {' '.join(cmd)}\n# cwd: {cwd}\n\n")
                lf.flush()
            proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=lf, stderr=subprocess.STDOUT)

            while True:
                elapsed = time.time() - t0
                if proc.poll() is not None:
                    exited_on_own = True
                    break
                if elapsed >= budget_seconds:
                    break
                util = gpu_utilization_snapshot()
                if util:
                    samples.append(util)
                time.sleep(max(1.0, min(poll_interval_seconds, budget_seconds - elapsed)))

            if not exited_on_own:
                try:
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                        proc.wait(timeout=30)
                    except Exception:
                        pass
            returncode = proc.returncode
        finally:
            if log_path is not None:
                lf.close()

        wall = time.time() - t0
        both_active_any_sample = any(
            len(u) == 2 and all(v > 0 for v in u) for u in samples
        )
        status = "exited_error" if (exited_on_own and returncode != 0) else (
            "completed" if exited_on_own else "timed_out"
        )
        ok = status != "exited_error" and both_active_any_sample
        return {
            "gpu_count": gpu_count,
            "skipped": False,
            "attempted": True,
            "status": status,
            "both_active_any_sample": both_active_any_sample,
            "samples": samples,
            "wall_seconds": wall,
            "ok": ok,
            "error": None,
        }
    except Exception as exc:
        return {
            "gpu_count": gpu_count,
            "skipped": False,
            "attempted": True,
            "status": "exception",
            "both_active_any_sample": False,
            "samples": samples,
            "wall_seconds": time.time() - t0,
            "ok": False,
            "error": repr(exc),
        }


def _main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Standalone 2-GPU DDP smoke test.")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--budget-seconds", type=float, default=None)
    args = parser.parse_args()

    result = run_ddp_smoke_test(budget_seconds=args.budget_seconds)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if (result["skipped"] or result["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
