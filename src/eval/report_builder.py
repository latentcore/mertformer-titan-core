"""Evaluation report builder (pre-training safe).

Collects benchmark outputs (if present) and produces a concise summary.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

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


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    humaneval = REPORT_DIR / "humaneval_outputs.jsonl"
    mbpp = REPORT_DIR / "mbpp_outputs.jsonl"
    gsm8k = REPORT_DIR / "gsm8k_outputs.jsonl"
    gsm8k_summary = REPORT_DIR / "gsm8k_summary.json"

    counts = {
        "humaneval_outputs": _count_jsonl(humaneval),
        "mbpp_outputs": _count_jsonl(mbpp),
        "gsm8k_outputs": _count_jsonl(gsm8k),
    }

    any_outputs = any(v > 0 for v in counts.values())
    status = "partial" if any_outputs else "pending"

    summary = {
        "status": status,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": counts,
        "notes": "Outputs are pre-training unless generated after a full training run.",
        "gsm8k_summary_path": str(gsm8k_summary) if gsm8k_summary.exists() else None,
    }
    out_path = REPORT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote benchmark summary to {out_path}")


if __name__ == "__main__":
    main()
