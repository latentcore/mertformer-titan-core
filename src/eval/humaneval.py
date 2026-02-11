"""HumanEval evaluator (wrapper).

This is a thin wrapper around `scripts/benchmarks_internal.py`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cmd = [sys.executable, "scripts/benchmarks_internal.py", "--run"]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)


if __name__ == "__main__":
    main()
