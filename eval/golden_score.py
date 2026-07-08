"""Golden assertion SCORER (wrapper).

[2026-07-08] `eval/golden.py` wraps `scripts/golden_eval.py`, which is the dry-run /
manifest-check tool — it does NOT score. The actual scorer, `scripts/golden_score.py`,
writes `reports/benchmarks/golden_summary.json`, which train/train.py's saturation gate
reads. It was the only real evaluator with no `eval/` entry point, so it was invisible
from the package that holds all its siblings (eval/gsm8k.py, eval/humaneval.py, eval/golden.py).

This wrapper mirrors the existing eval/golden.py -> scripts/golden_eval.py pattern exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.golden_score import main as golden_score_main


if __name__ == "__main__":
    golden_score_main()
