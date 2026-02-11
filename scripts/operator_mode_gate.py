"""
Operator Mode gate runner for the Specialized Edge Coding Launch.
Single-entry script to execute safety drills and readiness checks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import cfg
from utils.logger import RunLogger, sha256_file
from orchestrator.telemetry import system_snapshot


def dataset_manifest(root: Path, hash_all: bool, max_hash_mb: int) -> Dict[str, Dict[str, Any]]:
    manifest: Dict[str, Dict[str, Any]] = {}
    if not root.exists():
        return manifest

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        entry: Dict[str, Any] = {"bytes": size}
        if hash_all or size <= max_hash_mb * 1024 * 1024:
            try:
                entry["sha256"] = sha256_file(path)
            except Exception as exc:
                entry["sha256_error"] = str(exc)
        else:
            entry["sha256"] = None
        manifest[str(path)] = entry
    return manifest


def run_subprocess(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def run_nan_kill_test() -> None:
    result = subprocess.run([sys.executable, "scripts/nan_kill_test.py"], check=False)
    if result.returncode != 42:
        raise RuntimeError(f"NaN kill test failed: expected exit 42, got {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--no-pytest", action="store_true")
    parser.add_argument("--pytest-target", type=str, default="tests")
    parser.add_argument("--hash-all-datasets", action="store_true")
    parser.add_argument("--max-hash-mb", type=int, default=50)
    parser.add_argument("--overfit-dataset", type=str, default="datasets/stage1/stage1_data.jsonl")
    args = parser.parse_args()

    start = time.time()
    run_id = time.strftime("operator_%Y-%m-%d_%H-%M-%S", time.localtime())
    log_dir = PROJECT_ROOT / "logs" / "operator_mode"

    manifest = dataset_manifest(PROJECT_ROOT / "datasets", args.hash_all_datasets, args.max_hash_mb)
    results = []

    logger = RunLogger(
        cfg,
        log_dir=log_dir,
        run_name=run_id,
        project_root=PROJECT_ROOT,
        train_path=PROJECT_ROOT / "train" / "train.py",
        config_path=PROJECT_ROOT / "config" / "config.py",
        extra_files=[PROJECT_ROOT / "TASK.md", PROJECT_ROOT / "IMPLEMENTATION_PLAN.md"],
    )

    try:
        logger.log_meta({"seed": cfg.seed, "datasets": manifest, "mode": "full" if args.full else "safe"})

        run_nan_kill_test()
        results.append({"step": "nan_kill_switch", "status": "pass"})

        run_subprocess([sys.executable, "scripts/checkpoint_restore_drill.py"])
        results.append({"step": "checkpoint_restore_drill", "status": "pass"})

        run_subprocess([sys.executable, "scripts/failure_budget_drill.py"])
        results.append({"step": "failure_budget_drill", "status": "pass"})

        if args.full:
            run_subprocess([
                sys.executable,
                "scripts/overfit_gate.py",
                "--dataset",
                args.overfit_dataset,
                "--bytes",
                "1000000",
            ])
            results.append({"step": "overfit_gate", "status": "pass"})
        else:
            run_subprocess([
                sys.executable,
                "scripts/overfit_gate.py",
                "--dataset",
                args.overfit_dataset,
                "--fast",
            ])
            results.append({"step": "overfit_gate", "status": "pass_fast"})

        run_subprocess([sys.executable, "scripts/golden_eval.py"])
        results.append({"step": "golden_samples", "status": "pass"})

        snapshot = system_snapshot()
        results.append({"step": "telemetry_snapshot", "status": "pass", "snapshot": snapshot})

        if args.full:
            run_subprocess([sys.executable, "scripts/benchmarks_internal.py", "--run", "--samples", "5"])
            results.append({"step": "benchmarks", "status": "pass"})
        else:
            run_subprocess([sys.executable, "scripts/benchmarks_internal.py"])
            results.append({"step": "benchmarks", "status": "ready"})

        if not args.no_pytest:
            try:
                import pytest
                code = pytest.main([args.pytest_target])
                if code != 0:
                    raise RuntimeError(f"pytest failed with exit code {code}")
                results.append({"step": "pytest", "status": "pass"})
            except ModuleNotFoundError:
                results.append({"step": "pytest", "status": "skipped_missing_pytest"})
            except Exception as exc:
                raise RuntimeError(f"pytest failed: {exc}") from exc

        elapsed = time.time() - start
        summary = {"status": "completed", "elapsed_sec": elapsed, "results": results}
        logger.finalize(status="completed", extra=summary)
        print(json.dumps(summary, indent=2))

    except Exception as exc:
        elapsed = time.time() - start
        summary = {"status": "failed", "elapsed_sec": elapsed, "error": str(exc), "results": results}
        logger.finalize(status="failed", extra=summary)
        print(json.dumps(summary, indent=2))
        raise


if __name__ == "__main__":
    main()
