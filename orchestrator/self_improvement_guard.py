"""Policy-bound recursive self-improvement scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ImprovementProposal:
    title: str
    rationale: str
    risk: str
    requires_human_approval: bool = True


class SelfImprovementGuard:
    """
    Produces proposals but never auto-applies changes.
    This keeps the path policy-safe and reviewable.
    """

    def __init__(self, max_proposals: int = 5) -> None:
        self.max_proposals = int(max(1, max_proposals))

    def propose(self, telemetry: Dict[str, object]) -> List[ImprovementProposal]:
        proposals: List[ImprovementProposal] = []
        health = float(telemetry.get("health_score", 1.0))
        if health < 0.7:
            proposals.append(
                ImprovementProposal(
                    title="Improve verification calibration",
                    rationale="Low health score indicates unstable confidence calibration.",
                    risk="medium",
                )
            )
        if float(telemetry.get("failure_budget_signal", 0.0)) > 0.2:
            proposals.append(
                ImprovementProposal(
                    title="Tighten failure budget thresholds",
                    rationale="Frequent budget pressure suggests runtime risk accumulation.",
                    risk="low",
                )
            )
        if not proposals:
            proposals.append(
                ImprovementProposal(
                    title="No-op stability hold",
                    rationale="System health is stable; keep configuration unchanged.",
                    risk="low",
                )
            )
        return proposals[: self.max_proposals]

