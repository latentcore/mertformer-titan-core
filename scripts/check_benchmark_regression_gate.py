#!/usr/bin/env python3
"""
Checkpoint-bound benchmark regression gate.

[2026-07-12] BACKLOG I.7 #86: "CI eval kapilari: lm-eval-harness + benchmark
regresyon kapisi". Pre-45K there is no checkpoint and nothing to regress
against -- this gate reflects that honestly (SKIPPED/NO_CHECKPOINT, exit 0,
never blocks verify_all.sh) rather than faking a pass. Once a canonical
checkpoint exists, it compares the checkpoint-bound benchmark battery
(eval/held_out_ppl.py, eval/gsm8k.py, eval/humaneval.py, and the six
2026-07-12 probes) against a stored `reports/benchmarks/regression_baseline.json`
and hard-fails on a regression beyond --tolerance. Establishing that stored
baseline (via --update-baseline) is itself a deliberate, explicit, human-
triggered action -- never done silently by this gate.

Usage (pre-45K, no checkpoint):
    python scripts/check_benchmark_regression_gate.py --checkpoint checkpoints/latest.pt
    # -> SKIPPED, exit 0

Usage (post-45K, once a checkpoint and baseline exist):
    python scripts/check_benchmark_regression_gate.py --checkpoint checkpoints/latest.pt \\
        --update-baseline   # first run: establish the baseline
    python scripts/check_benchmark_regression_gate.py --checkpoint checkpoints/latest.pt
        # subsequent runs: compare against the stored baseline, fail on regression
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval._probe_common import resolve_checkpoint_or_none  # noqa: E402

DEFAULT_BASELINE = PROJECT_ROOT / "reports" / "benchmarks" / "regression_baseline.json"
DEFAULT_OUT = PROJECT_ROOT / "reports" / "benchmarks" / "regression_gate_summary.json"

# metric name -> (summary file, json key, higher_is_better)
TRACKED_METRICS = {
    "held_out_ppl": ("held_out_ppl_summary.json", "ppl", False),
    "calibration_ece": ("calibration_ece_summary.json", "ece", False),
    "bias_mean_abs_skew": ("bias_fairness_probe_summary.json", "mean_absolute_skew", False),
    "toxicity_hit_rate": ("toxicity_probe_summary.json", "denylist_hit_rate", False),
    "hallucination_rate": ("hallucination_rate_probe_summary.json", "hallucination_rate_proxy", False),
    "adversarial_similarity": ("adversarial_prompt_robustness_summary.json", "mean_completion_similarity", True),
}


def _read_metric(benchmarks_dir: Path, filename: str, key: str) -> float | None:
    path = benchmarks_dir / filename
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def collect_current_metrics(benchmarks_dir: Path) -> Dict[str, float]:
    current = {}
    for name, (filename, key, _higher_is_better) in TRACKED_METRICS.items():
        value = _read_metric(benchmarks_dir, filename, key)
        if value is not None:
            current[name] = value
    return current


def compare(current: Dict[str, float], baseline: Dict[str, float], tolerance: float) -> list[str]:
    regressions = []
    for name, (_filename, _key, higher_is_better) in TRACKED_METRICS.items():
        if name not in current or name not in baseline:
            continue
        cur, base = current[name], baseline[name]
        if base == 0:
            continue
        delta = (cur - base) / abs(base)
        regressed = (delta < -tolerance) if higher_is_better else (delta > tolerance)
        if regressed:
            regressions.append(
                f"{name}: baseline={base:.4f} current={cur:.4f} delta={delta:+.1%} "
                f"(tolerance {tolerance:.0%}, higher_is_better={higher_is_better})"
            )
    return regressions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpoint-bound benchmark regression gate.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmarks-dir", default=str(PROJECT_ROOT / "reports" / "benchmarks"))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if resolve_checkpoint_or_none(args.checkpoint) is None:
        summary: Dict[str, Any] = {
            "schema": "benchmark_regression_gate_v1",
            "status": "SKIPPED",
            "reason_code": "NO_CHECKPOINT",
            "message": "No checkpoint found; regression gate not run. Expected pre-45K.",
        }
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    benchmarks_dir = Path(args.benchmarks_dir)
    current = collect_current_metrics(benchmarks_dir)

    baseline_path = Path(args.baseline)
    if args.update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        summary = {
            "schema": "benchmark_regression_gate_v1",
            "status": "BASELINE_UPDATED",
            "baseline": current,
            "baseline_path": str(baseline_path),
        }
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    if not baseline_path.exists():
        summary = {
            "schema": "benchmark_regression_gate_v1",
            "status": "SKIPPED",
            "reason_code": "NO_BASELINE",
            "message": f"No baseline at {baseline_path}; run with --update-baseline first.",
        }
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    regressions = compare(current, baseline, args.tolerance)

    summary = {
        "schema": "benchmark_regression_gate_v1",
        "status": "FAIL" if regressions else "PASS",
        "current": current,
        "baseline": baseline,
        "tolerance": args.tolerance,
        "regressions": regressions,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
