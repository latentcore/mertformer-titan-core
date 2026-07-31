#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MertFormer Titan — Kaggle Batch Runner
=======================================

Unattended, sequential 5-job runner for a single Kaggle "Save & Run All
(Commit)" session. Originally built and run OUTSIDE the mertformer-titan-core
repo (a Claude Code session that had read-only repo access for that task);
added here on 2026-07-25 after a real Kaggle run produced verified results
(see BACKLOG.md's N3/N4 entries and the "Third real-hardware confirmation"
note). It drives unmodified copies of the repo's own scripts and does not
change the canonical 45K training path.

Jobs, in shortest-expected-duration-first order, chess deliberately LAST
(lowest priority — not part of the tracked pre-45K/post-45K backlog, so it
is the job sacrificed first if time runs short, not the LM safety-fix
verification):
  1. Nutrition5k N3  (Liquid-OFF ablation)   ~1.5h budget
  2. Nutrition5k N4  (MoE-OFF ablation)      ~1.5h budget
  3. 36M LM re-verify (preflight_run.py)     ~2.0h budget, DDP-attempted if 2 GPUs
  4. 171M LM re-verify (preflight_run_pilot171m.py) protected 3.0h budget, DDP-attempted
  5. Chess 5080 PoC  (chess_5080_onefile.py) remaining budget (self-time-boxing)

Design principles (see the accompanying README.md for the full rationale):
  - Every job runs as an isolated subprocess against its OWN copy of the repo
    snapshot, wall-clock time-boxed. A hang/crash/divergence in one job can
    never take down the orchestrator or starve a later job of its own budget.
  - Every job's outcome is caught and logged; the loop always continues.
  - Outputs are written incrementally, not just at the very end, so a
    mid-session forced kill still leaves partial results in place.
  - Only a job that actually COMPLETES within its time-box gets its full
    result folder in the final zip. A job that timed out gets one manifest
    line ("attempted, did not finish") and nothing else — no half-finished
    folder is ever presented as a real result.
  - Multi-GPU is attempted ONLY for the two LM scripts (the only ones with
    any DDP-capable code underneath, via HuggingFace Accelerate), and only
    after a cheap up-front smoke test proves it actually works in this
    environment. Chess and Nutrition5k always run single-GPU — there is no
    multi-GPU code path in either script, and writing one from scratch for a
    one-shot unsupervised run is not worth the risk.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile
import hashlib
from datetime import datetime, timezone
from pathlib import Path

# =====================================================================
# Configuration (env-overridable; sane defaults for a real Kaggle run)
# =====================================================================
TOTAL_BUDGET_HOURS = float(os.environ.get("MERTFORMER_TOTAL_BUDGET_HOURS", "8.5"))
SAFETY_MARGIN_MINUTES = float(os.environ.get("MERTFORMER_SAFETY_MARGIN_MINUTES", "20"))
MIN_LAST_JOB_MINUTES = 15.0  # if less than this remains for job 5, skip it entirely
SIGTERM_GRACE_SECONDS = int(os.environ.get("MERTFORMER_SIGTERM_GRACE_SECONDS", "300"))
DDP_SMOKE_TEST_SECONDS = int(os.environ.get("MERTFORMER_DDP_SMOKE_SECONDS", "240"))

HERE = Path(__file__).resolve().parent  # .../orchestrator
BUNDLE_ROOT = HERE.parent  # the unzipped Kaggle Dataset root
REPO_SNAPSHOT_SRC = BUNDLE_ROOT / "repo_snapshot"

_KAGGLE_WORKING = Path("/kaggle/working")
WORKING_DIR = Path(os.environ.get("MERTFORMER_WORKING_DIR", "")) or (
    _KAGGLE_WORKING if _KAGGLE_WORKING.exists() else (BUNDLE_ROOT / "working")
)
JOBS_DIR = WORKING_DIR / "mertformer_batch_output"
FINAL_ZIP_BASENAME = "MertFormer_Kaggle_Batch_Output"
LOCK_PATH = WORKING_DIR / "batch_runner.lock"

RUN_STARTED_AT = time.time()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def acquire_lock() -> bool:
    """Hard-prevents two orchestrator instances from ever running concurrently
    in the same container -- added 2026-07-25 after exactly this happened on a
    real Kaggle run (a duplicate notebook cell caused two full instances to
    race on the same JOBS_DIR/output zip/GPUs). Exclusive file creation
    (O_CREAT | O_EXCL) is atomic at the OS level, so this is race-free even
    if two processes call it within milliseconds of each other."""
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(f"pid={os.getpid()} started_utc={now_utc()}\n")
        return True
    except FileExistsError:
        return False


def release_lock() -> None:
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def elapsed_seconds() -> float:
    return time.time() - RUN_STARTED_AT


def remaining_budget_seconds() -> float:
    total = TOTAL_BUDGET_HOURS * 3600.0
    margin = SAFETY_MARGIN_MINUTES * 60.0
    return max(0.0, total - margin - elapsed_seconds())


def log(msg: str) -> None:
    print(f"[batch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =====================================================================
# Job specs
# =====================================================================
JOBS = [
    {
        "name": "01_nutrition5k_liquid_off",
        "kind": "nutrition5k_ablation",
        "ablation": "liquid_off",
        "budget_hours": 1.5,
    },
    {
        "name": "02_nutrition5k_moe_off",
        "kind": "nutrition5k_ablation",
        "ablation": "moe_off",
        "budget_hours": 1.5,
    },
    {
        "name": "03_lm_36m_reverify",
        "kind": "lm_preflight",
        "script_rel": "scripts/preflight_run.py",
        "budget_hours": 2.0,
        "allow_ddp": True,
    },
    {
        # Reordered 2026-07-25 per Mert: chess is NOT part of the tracked
        # pre-45K/post-45K backlog and isn't the main goal -- it should be
        # the one sacrificed if time runs short, not the LM re-verification
        # (which is directly checking safety-critical fixes). So it now gets
        # a protected, fixed budget in position 4 instead of "whatever
        # remains" -- more likely to actually finish and produce a real
        # verification signal.
        "name": "04_lm_171m_reverify",
        "kind": "lm_preflight",
        "script_rel": "scripts/preflight_run_pilot171m.py",
        "budget_hours": 3.0,
        "allow_ddp": True,
    },
    {
        "name": "05_chess_5080_poc",
        "kind": "chess",
        "budget_hours": None,  # whatever remains -- lowest priority, last in line on purpose
    },
]

# =====================================================================
# Environment detection
# =====================================================================
def detect_gpu_count() -> int:
    try:
        import torch  # noqa: WPS433

        return int(torch.cuda.device_count())
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15
        )
        if out.returncode == 0:
            return len([ln for ln in out.stdout.splitlines() if ln.strip()])
    except Exception:
        pass
    return 0


def detect_environment() -> dict:
    is_kaggle = bool(
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or os.environ.get("KAGGLE_URL_BASE")
        or os.environ.get("KAGGLE_KERNEL_RUN_ID")
    )
    gpu_count = detect_gpu_count()
    info = {
        "generated_utc": now_utc(),
        "is_kaggle": is_kaggle,
        "gpu_count": gpu_count,
        "working_dir": str(WORKING_DIR),
        "repo_snapshot_src": str(REPO_SNAPSHOT_SRC),
        "total_budget_hours": TOTAL_BUDGET_HOURS,
        "safety_margin_minutes": SAFETY_MARGIN_MINUTES,
    }
    log(f"environment: kaggle={is_kaggle} gpu_count={gpu_count} budget={TOTAL_BUDGET_HOURS}h")
    return info


# =====================================================================
# Subprocess execution with wall-clock time-boxing (SIGTERM -> grace -> KILL)
# =====================================================================
def run_timeboxed(
    cmd: list,
    cwd: Path,
    log_path: Path,
    budget_seconds: float,
    env: dict | None = None,
) -> dict:
    """Runs `cmd`, capped at `budget_seconds` wall-clock. Returns a result dict
    with status in {"completed", "timed_out", "failed"} and wall_seconds.

    On timeout: SIGTERM first (train/train.py installs a graceful-checkpoint
    handler for exactly this — see its own comment about Kaggle preemption),
    wait SIGTERM_GRACE_SECONDS, then SIGKILL as a last resort. Scripts with no
    SIGTERM handler (Nutrition5k) still get a cleaner shutdown than a bare
    SIGKILL would give, even though they won't checkpoint gracefully.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    with log_path.open("w", encoding="utf-8", errors="replace") as lf:
        lf.write(f"# cmd: {' '.join(cmd)}\n# cwd: {cwd}\n# budget_seconds: {budget_seconds}\n\n")
        lf.flush()
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(cwd), stdout=lf, stderr=subprocess.STDOUT, env=full_env
            )
        except Exception as exc:
            return {
                "status": "failed",
                "wall_seconds": 0.0,
                "returncode": None,
                "error": f"failed to launch: {exc!r}",
            }

        try:
            returncode = proc.wait(timeout=budget_seconds)
            wall = time.time() - t0
            return {
                "status": "completed" if returncode == 0 else "failed",
                "wall_seconds": wall,
                "returncode": returncode,
                "error": None if returncode == 0 else f"nonzero exit {returncode}",
            }
        except subprocess.TimeoutExpired:
            log(f"  budget of {budget_seconds:.0f}s exceeded, sending SIGTERM ...")
            try:
                proc.send_signal(signal.SIGTERM)
            except Exception:
                pass
            try:
                proc.wait(timeout=SIGTERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                log("  still alive after grace period, sending SIGKILL")
                try:
                    proc.kill()
                    proc.wait(timeout=30)
                except Exception:
                    pass
            wall = time.time() - t0
            return {
                "status": "timed_out",
                "wall_seconds": wall,
                "returncode": proc.returncode,
                "error": f"exceeded {budget_seconds:.0f}s budget",
            }


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


# =====================================================================
# Repo-snapshot staging per job (each job gets its OWN independent copy —
# simpler and safer than trying to share/redirect paths between jobs)
# =====================================================================
def stage_job_repo(job_name: str) -> Path:
    job_root = JOBS_DIR / job_name
    repo_dir = job_root / "repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    job_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_SNAPSHOT_SRC, repo_dir)
    return repo_dir


def patch_constant(file_path: Path, pattern: str, replacement: str) -> None:
    """Single, targeted, verified text substitution on a JOB'S OWN COPY of a
    script (never the shared repo_snapshot source) -- e.g. disabling Liquid
    or MoE for a Nutrition5k ablation. Raises if the pattern isn't found
    exactly once, so a silent no-op patch can never happen."""
    text = file_path.read_text(encoding="utf-8")
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly 1 match for {pattern!r} in {file_path}, found {len(matches)}"
        )
    new_text = re.sub(pattern, replacement, text, count=1)
    file_path.write_text(new_text, encoding="utf-8")


# =====================================================================
# DDP smoke test (only attempted for the 2 LM jobs, only if 2 GPUs present)
# =====================================================================
def ddp_smoke_test(env_info: dict) -> bool:
    """Decide whether to trust `accelerate launch --num_processes 2` for the
    two LM re-verify jobs. Rewritten 2026-07-25 after a real Kaggle run showed
    the original version could say "USE DDP" with `gpu_util=[0, 0]`: it only
    sampled GPU utilization ONCE, right after the subprocess had already been
    sent SIGTERM (i.e. while it was shutting down, not while it was actually
    running), and treated a bare `made_progress` flag (state.json merely
    existing -- true even from non-GPU setup work) as sufficient on its own.
    This version polls GPU utilization repeatedly WHILE the subprocess is
    still alive and only returns True if at least one of those in-flight
    samples showed both GPUs genuinely active. Ambiguous or inconclusive
    always falls back to single-GPU -- the well-tested path -- since a false
    "USE DDP" costs hours of a job's budget for zero result, while a false
    "fall back" only costs the DDP speedup, never correctness.
    """
    if env_info["gpu_count"] != 2:
        return False
    log("2 GPUs detected -- running a cheap DDP smoke test before committing real budget ...")
    try:
        smoke_repo = stage_job_repo("_ddp_smoke_test")
        cmd = [
            "accelerate",
            "launch",
            "--num_processes",
            "2",
            "--num_machines",
            "1",
            "--mixed_precision",
            "bf16",
            "scripts/preflight_run.py",
        ]
        log_dir = JOBS_DIR / "_ddp_smoke_test"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "smoke.log"

        poll_interval_seconds = 10.0
        smoke_grace_seconds = 60.0  # short on purpose: disposable smoke test, no checkpoint to protect
        samples: list = []
        t0 = time.time()

        with log_path.open("w", encoding="utf-8", errors="replace") as lf:
            lf.write(f"# cmd: {' '.join(cmd)}\n# cwd: {smoke_repo}\n\n")
            lf.flush()
            proc = subprocess.Popen(cmd, cwd=str(smoke_repo), stdout=lf, stderr=subprocess.STDOUT)

            exited_on_own = False
            while True:
                elapsed = time.time() - t0
                if proc.poll() is not None:
                    exited_on_own = True
                    break
                if elapsed >= DDP_SMOKE_TEST_SECONDS:
                    break
                util = gpu_utilization_snapshot()
                if util:
                    samples.append(util)
                time.sleep(max(1.0, min(poll_interval_seconds, DDP_SMOKE_TEST_SECONDS - elapsed)))

            if not exited_on_own:
                log(f"  smoke test budget of {DDP_SMOKE_TEST_SECONDS:.0f}s reached, sending SIGTERM ...")
                try:
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=smoke_grace_seconds)
                except subprocess.TimeoutExpired:
                    log("  smoke test still alive after grace period, sending SIGKILL")
                    try:
                        proc.kill()
                        proc.wait(timeout=30)
                    except Exception:
                        pass
            returncode = proc.returncode

        wall = time.time() - t0
        both_active_any_sample = any(
            len(u) == 2 and all(v > 0 for v in u) for u in samples
        )
        status = "exited_error" if (exited_on_own and returncode != 0) else (
            "completed" if exited_on_own else "timed_out"
        )
        ok = status != "exited_error" and both_active_any_sample
        log(
            f"  DDP smoke test: status={status} wall={wall:.0f}s samples={samples} "
            f"both_active_any_sample={both_active_any_sample} -> "
            f"{'USE DDP' if ok else 'FALL BACK TO SINGLE-GPU'}"
        )
        return ok
    except Exception as exc:
        log(f"  DDP smoke test raised an exception, falling back to single-GPU: {exc!r}")
        return False
    finally:
        shutil.rmtree(JOBS_DIR / "_ddp_smoke_test", ignore_errors=True)


# =====================================================================
# Per-job-kind runners. Each returns a result dict with at least:
#   status ("completed" | "timed_out" | "failed" | "skipped")
#   wall_seconds, output_dir (Path or None), notes (str)
# =====================================================================
def run_nutrition5k_ablation(job: dict, budget_seconds: float) -> dict:
    repo_dir = stage_job_repo(job["name"])
    script_path = repo_dir / "scripts" / "train_nutrition5k.py"

    if job["ablation"] == "liquid_off":
        patch_constant(script_path, r"LIQUID_LAYER_IDS = \(5,\)", "LIQUID_LAYER_IDS = ()")
    elif job["ablation"] == "moe_off":
        patch_constant(script_path, r"MOE_LAYER_IDS = \(3, 6\)", "MOE_LAYER_IDS = ()")
    else:
        raise ValueError(f"unknown ablation {job['ablation']!r}")

    log_path = JOBS_DIR / job["name"] / "run.log"
    result = run_timeboxed(
        ["python3", "scripts/train_nutrition5k.py"],
        cwd=repo_dir,
        log_path=log_path,
        budget_seconds=budget_seconds,
    )

    metrics_path = repo_dir / "scripts" / "nutrition5k_work" / "metrics.json"
    key_metrics = None
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            val = data.get("train_result", {}).get("val_metrics", {}).get("calories", {})
            key_metrics = {
                "calorie_mae": val.get("mae"),
                "calorie_mae_pct": val.get("mae_pct"),
            }
        except Exception:
            pass

    result["output_source"] = repo_dir / "scripts" / "nutrition5k_work"
    result["extra_copy"] = [
        repo_dir / "scripts" / "nutrition5k_work" / "REPORT.md",
    ]
    result["key_metrics"] = key_metrics
    result["repo_dir"] = repo_dir
    return result


def run_lm_preflight(job: dict, budget_seconds: float, use_ddp: bool) -> dict:
    repo_dir = stage_job_repo(job["name"])
    if use_ddp:
        cmd = [
            "accelerate",
            "launch",
            "--num_processes",
            "2",
            "--num_machines",
            "1",
            "--mixed_precision",
            "bf16",
            job["script_rel"],
        ]
    else:
        cmd = ["python3", job["script_rel"]]

    log_path = JOBS_DIR / job["name"] / "run.log"
    result = run_timeboxed(cmd, cwd=repo_dir, log_path=log_path, budget_seconds=budget_seconds)
    result["used_ddp"] = use_ddp

    script_dir = repo_dir / Path(job["script_rel"]).parent
    output_zip = script_dir / "preflight_run_output.zip"
    work_dir = script_dir / "preflight_work"
    result["output_source"] = output_zip if output_zip.exists() else work_dir
    result["repo_dir"] = repo_dir

    metrics_path = work_dir / "metrics.json"
    if metrics_path.exists():
        try:
            result["key_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            result["key_metrics"] = None
    else:
        result["key_metrics"] = None
    return result


def run_chess(job: dict, budget_seconds: float) -> dict:
    repo_dir = stage_job_repo(job["name"])
    artifact_root = JOBS_DIR / job["name"] / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    # Reserve a little headroom below our own outer timeout so the script's
    # OWN internal max-wall-hours check (chess_5080_onefile.py, ~line 5306)
    # fires first in the normal case -- our SIGTERM is the backstop, not the
    # primary stop mechanism.
    inner_budget_hours = max(0.05, (budget_seconds / 3600.0) - (10.0 / 60.0))
    # Fixed 2026-07-25 after a real Kaggle run: `python3 scripts/chess_5080_onefile.py`
    # puts the script's OWN directory (repo_dir/scripts/) on sys.path[0], not repo_dir
    # itself -- so the script's own `from scripts.chess_onefile_contract import ...`
    # raised `ModuleNotFoundError: No module named 'scripts'` before any training
    # started (0.365s wall time on the real run; not a chess/training result at all).
    # `-m scripts.chess_5080_onefile` with cwd=repo_dir puts repo_dir on sys.path
    # instead, so the `scripts` package (scripts/__init__.py exists) resolves.
    cmd = [
        "python3",
        "-m",
        "scripts.chess_5080_onefile",
        "--profile",
        "production_5080",
        "--max-wall-hours",
        f"{inner_budget_hours:.3f}",
        "--artifact-root",
        str(artifact_root),
        "--allow-install",
    ]
    log_path = JOBS_DIR / job["name"] / "run.log"
    result = run_timeboxed(cmd, cwd=repo_dir, log_path=log_path, budget_seconds=budget_seconds)
    result["output_source"] = artifact_root
    result["repo_dir"] = repo_dir
    result["key_metrics"] = None
    return result


# =====================================================================
# Output collection (incremental -- called right after each job, not just
# at the very end, so a mid-session forced kill still preserves progress)
# =====================================================================
def collect_job_output(job: dict, result: dict) -> Path | None:
    dest = JOBS_DIR / job["name"] / "collected"
    src = result.get("output_source")
    if result["status"] != "completed":
        # Per Mert's instruction: a job that didn't finish gets a manifest
        # note only, never a folder presented as if it were a real result.
        return None
    if not src or not Path(src).exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    if Path(src).is_file():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / Path(src).name)
    else:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    return dest


def cleanup_job_repo(job: dict, result: dict) -> None:
    """Free disk space once a job's useful output has been collected -- the
    per-job repo copy (with its downloaded datasets/checkpoints) is no
    longer needed once `collected/` (or the timeout note) exists."""
    repo_dir = result.get("repo_dir")
    if repo_dir and Path(repo_dir).exists():
        shutil.rmtree(repo_dir, ignore_errors=True)


# =====================================================================
# Final zip assembly -- mirrors scripts/build_training_outputs_bundle.py's
# established pattern: temp file + atomic os.replace, ZIP_DEFLATED +
# allowZip64, sha256 sidecar, JSON+MD manifest pair.
# =====================================================================
def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_final_zip(env_info: dict, job_results: list) -> None:
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    final_zip = WORKING_DIR / f"{FINAL_ZIP_BASENAME}.zip"
    tmp_zip = WORKING_DIR / f".{FINAL_ZIP_BASENAME}.zip.tmp"
    if tmp_zip.exists():
        tmp_zip.unlink()

    manifest = {
        "schema": "mertformer_kaggle_batch_output_manifest_v1",
        "generated_utc": now_utc(),
        "environment": env_info,
        "total_wall_seconds": elapsed_seconds(),
        "jobs": [],
    }

    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for job, result in job_results:
            entry = {
                "name": job["name"],
                "kind": job["kind"],
                "status": result.get("status"),
                "wall_seconds": result.get("wall_seconds"),
                "used_ddp": result.get("used_ddp", False),
                "key_metrics": result.get("key_metrics"),
                "error": result.get("error"),
                "included_in_zip": False,
            }
            collected = result.get("collected_dir")
            if result.get("status") == "completed" and collected and Path(collected).exists():
                for f in Path(collected).rglob("*"):
                    if f.is_file():
                        arcname = f"{job['name']}/{f.relative_to(collected)}"
                        zf.write(f, arcname)
                entry["included_in_zip"] = True
            manifest["jobs"].append(entry)

        manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
        zf.writestr("manifest.json", manifest_json)

        md_lines = [
            "# MertFormer Titan — Kaggle Batch Output",
            "",
            f"Generated: {manifest['generated_utc']}",
            f"Total wall time: {manifest['total_wall_seconds'] / 3600:.2f}h",
            "",
            "| Job | Status | Wall time | DDP | In zip |",
            "|---|---|---:|---|---|",
        ]
        for entry in manifest["jobs"]:
            wall = entry["wall_seconds"]
            wall_str = f"{wall / 3600:.2f}h" if wall else "n/a"
            md_lines.append(
                f"| {entry['name']} | {entry['status']} | {wall_str} | "
                f"{'yes' if entry['used_ddp'] else 'no'} | {'yes' if entry['included_in_zip'] else 'no'} |"
            )
        zf.writestr("manifest.md", "\n".join(md_lines) + "\n")

    # Verify before publishing.
    with zipfile.ZipFile(tmp_zip) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"output zip failed integrity check on member: {bad}")

    os.replace(tmp_zip, final_zip)
    digest = sha256_file(final_zip)
    (WORKING_DIR / f"{FINAL_ZIP_BASENAME}.zip.sha256").write_text(f"{digest}  {final_zip.name}\n")
    (WORKING_DIR / f"{FINAL_ZIP_BASENAME}_manifest.json").write_text(manifest_json)
    (WORKING_DIR / f"{FINAL_ZIP_BASENAME}_manifest.md").write_text("\n".join(md_lines) + "\n")

    log(f"FINAL ZIP written: {final_zip} ({final_zip.stat().st_size / 1e6:.1f} MB), sha256={digest[:16]}...")


# =====================================================================
# Main
# =====================================================================
def main() -> int:
    if not acquire_lock():
        existing = "(unreadable)"
        try:
            existing = LOCK_PATH.read_text().strip()
        except Exception:
            pass
        log(
            "FATAL: another kaggle_batch_runner.py instance appears to already be running "
            f"in this container (lock held: {existing}). Refusing to start a second, "
            "concurrent instance -- this guard exists specifically because two duplicate "
            "notebook cells once caused exactly this, racing on the same output dir and "
            f"GPUs. If you are certain no other instance is actually running, delete "
            f"{LOCK_PATH} yourself and retry."
        )
        return 3
    try:
        return _run_batch()
    finally:
        release_lock()


def _run_batch() -> int:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    env_info = detect_environment()
    (WORKING_DIR / "batch_preflight.json").write_text(json.dumps(env_info, indent=2))

    ddp_ok = ddp_smoke_test(env_info) if env_info["gpu_count"] == 2 else False

    job_results = []
    for i, job in enumerate(JOBS):
        log(f"=== job {i + 1}/{len(JOBS)}: {job['name']} ({job['kind']}) ===")

        # Decide this job's budget (or whether it must be skipped outright)
        # WITHOUT an early `continue` -- every job, skipped or not, must fall
        # through to the same tail (collect/cleanup/append/incremental-zip)
        # below, or a skipped LAST job would silently never make it into the
        # final manifest at all (caught by the local dry-run test suite).
        skip_reason = None
        if job["budget_hours"] is None:
            budget_seconds = remaining_budget_seconds()
            if budget_seconds < MIN_LAST_JOB_MINUTES * 60:
                skip_reason = (
                    f"insufficient time remaining ({budget_seconds / 60:.1f} min left, "
                    f"below the {MIN_LAST_JOB_MINUTES:.0f} min floor)"
                )
        else:
            budget_seconds = min(job["budget_hours"] * 3600.0, remaining_budget_seconds())
            if budget_seconds <= 0:
                skip_reason = "no budget remaining"

        if skip_reason is not None:
            log(f"  skipping: {skip_reason}")
            result = {
                "status": "skipped",
                "wall_seconds": 0.0,
                "error": skip_reason,
                "key_metrics": None,
            }
        else:
            try:
                if job["kind"] == "nutrition5k_ablation":
                    result = run_nutrition5k_ablation(job, budget_seconds)
                elif job["kind"] == "lm_preflight":
                    use_ddp = ddp_ok and job.get("allow_ddp", False)
                    result = run_lm_preflight(job, budget_seconds, use_ddp)
                elif job["kind"] == "chess":
                    result = run_chess(job, budget_seconds)
                else:
                    result = {
                        "status": "failed",
                        "wall_seconds": 0.0,
                        "error": f"unknown job kind {job['kind']!r}",
                        "key_metrics": None,
                    }
            except Exception as exc:  # noqa: BLE001 -- a job must NEVER take the orchestrator down
                log(f"  EXCEPTION in job {job['name']}: {exc!r}")
                result = {
                    "status": "failed",
                    "wall_seconds": 0.0,
                    "error": f"orchestrator-level exception: {exc!r}\n{traceback.format_exc()}",
                    "key_metrics": None,
                }

        log(f"  -> status={result.get('status')} wall={result.get('wall_seconds', 0) / 3600:.2f}h")

        try:
            collected = collect_job_output(job, result)
            result["collected_dir"] = collected
        except Exception as exc:
            log(f"  output-collection exception (non-fatal): {exc!r}")
            result["collected_dir"] = None

        try:
            cleanup_job_repo(job, result)
        except Exception as exc:
            log(f"  cleanup exception (non-fatal): {exc!r}")

        job_results.append((job, result))

        # Incremental manifest write after EVERY job, not just at the end.
        try:
            build_final_zip(env_info, job_results)
        except Exception as exc:
            log(f"  incremental zip-write exception (non-fatal, will retry after next job): {exc!r}")

    log("=== all jobs attempted, batch complete ===")
    for job, result in job_results:
        log(f"  {job['name']}: {result.get('status')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
