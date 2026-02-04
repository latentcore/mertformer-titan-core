"""Evaluation report builder (template).

Collects benchmark outputs and produces a concise summary.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "template",
        "notes": "Populate with benchmark outputs after training.",
    }
    out_path = REPORT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote template summary to {out_path}")


if __name__ == "__main__":
    main()
