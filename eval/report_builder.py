"""Evaluation report builder (pre-training safe).

Collects benchmark outputs (if present) and produces a concise summary.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _metric_entry(
    metric: str,
    baseline: float,
    current: float,
    confidence: float,
    evidence_ref: str,
) -> Dict[str, Any]:
    return {
        "metric": metric,
        "baseline": float(baseline),
        "current": float(current),
        "delta": float(current - baseline),
        "confidence": float(max(0.0, min(1.0, confidence))),
        "evidence_ref": evidence_ref,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    humaneval = REPORT_DIR / "humaneval_outputs.jsonl"
    mbpp = REPORT_DIR / "mbpp_outputs.jsonl"
    gsm8k = REPORT_DIR / "gsm8k_outputs.jsonl"
    gsm8k_summary = REPORT_DIR / "gsm8k_summary.json"
    generalization_report = REPORT_DIR / "generalization_suite_build30.json"
    agentic_report = REPORT_DIR / "agentic_suite_build30.json"

    counts = {
        "humaneval_outputs": _count_jsonl(humaneval),
        "mbpp_outputs": _count_jsonl(mbpp),
        "gsm8k_outputs": _count_jsonl(gsm8k),
    }

    any_outputs = any(v > 0 for v in counts.values())
    generalization_payload = _load_json(generalization_report)
    agentic_payload = _load_json(agentic_report)

    has_structured = bool(generalization_payload) or bool(agentic_payload)
    status = "ready" if has_structured else ("partial" if any_outputs else "pending")

    generalization_pass_rate = float(generalization_payload.get("pass_rate", 0.0))
    agentic_completion_rate = float(agentic_payload.get("completion_rate", 0.0))
    safety_regression = float(agentic_payload.get("safety_violation_rate", 0.0))

    metrics = [
        _metric_entry(
            metric="generalization.pass_rate",
            baseline=0.80,
            current=generalization_pass_rate,
            confidence=0.85 if generalization_payload else 0.40,
            evidence_ref=str(generalization_report) if generalization_payload else "missing",
        ),
        _metric_entry(
            metric="agentic.completion_rate",
            baseline=0.50,
            current=agentic_completion_rate,
            confidence=0.80 if agentic_payload else 0.40,
            evidence_ref=str(agentic_report) if agentic_payload else "missing",
        ),
        _metric_entry(
            metric="agentic.safety_regression",
            baseline=0.0,
            current=safety_regression,
            confidence=0.90 if agentic_payload else 0.40,
            evidence_ref=str(agentic_report) if agentic_payload else "missing",
        ),
    ]

    summary = {
        "schema": "benchmark_summary_v2",
        "status": status,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": counts,
        "notes": "Outputs are pre-training unless generated after a full training run.",
        "gsm8k_summary_path": str(gsm8k_summary) if gsm8k_summary.exists() else None,
        "generalization_report_path": str(generalization_report) if generalization_payload else None,
        "agentic_report_path": str(agentic_report) if agentic_payload else None,
        "metrics": metrics,
        "gate_thresholds": {
            "generalization_pass_rate_min": 0.80,
            "agentic_completion_gain_target": 0.20,
            "safety_regression_max": 0.0,
        },
    }
    out_path = REPORT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote benchmark summary to {out_path}")


if __name__ == "__main__":
    main()
