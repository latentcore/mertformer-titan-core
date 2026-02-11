"""
Failure budget drill to verify pivot trigger logic.
"""
from __future__ import annotations

import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.failure_budget import FailureBudget, FailureBudgetConfig


def main() -> None:
    config = FailureBudgetConfig(
        max_hours_no_progress=0.0003,  # ~1 second
        min_loss_slope_per_hour=-0.0001,
        min_router_entropy=0.2,
        min_expert_balance=0.2,
        window=5,
    )
    budget = FailureBudget(config=config)

    now = time.time()
    # Simulate flat loss and poor balance
    for i in range(6):
        budget.update(1.0, router_entropy=0.05, expert_balance=0.05, timestamp=now + i * 0.2)

    report = budget.update(1.0, router_entropy=0.05, expert_balance=0.05, timestamp=now + 2.0)
    if report.get("should_pivot", 0.0) < 1.0:
        raise RuntimeError("Failure budget drill failed: pivot not triggered")

    print("Failure budget drill: PASS")


if __name__ == "__main__":
    main()
