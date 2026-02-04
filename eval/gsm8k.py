"""GSM8K evaluator (stub).

This file provides a clean entrypoint for GSM8K evaluation once the dataset
is wired into the pipeline. For now, use `scripts/benchmarks_internal.py`
for HumanEval/MBPP and add GSM8K integration here.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    print("GSM8K evaluator is a stub. Integrate dataset and scoring here.")


if __name__ == "__main__":
    main()
