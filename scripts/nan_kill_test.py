"""
Synthetic NaN kill-switch test.
Exits with code 42 when the kill switch triggers.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.safety import kill_if_non_finite


def main() -> None:
    value = float("nan")
    kill_if_non_finite(value, name="synthetic_nan", action="exit", exit_code=42)


if __name__ == "__main__":
    main()
