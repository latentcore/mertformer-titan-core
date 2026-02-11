"""Golden sample evaluator (wrapper)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.golden_eval import main as golden_main


if __name__ == "__main__":
    golden_main()
