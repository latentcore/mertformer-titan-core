"""Policy-bound recursive self-improvement scaffold with controlled auto-apply."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ImprovementProposal:
    title: str
    rationale: str
    risk: str  # "low", "medium", "high"
    requires_human_approval: bool = True
    category: str = "general"  # "performance", "stability", "strategy", "general"


@dataclass
class ApplyResult:
    """Application result."""
    applied: bool
    proposal_title: str = ""
    reason: str = ""
    rollback_id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)


class SelfImprovementGuard:
    """
    Controlled self-improvement guard.

    - Low-risk proposals may be auto-applied (only when allow_auto_apply=True)
    - High-risk proposals always require human approval
    - Every applied change is reversible (rollback stack)
    """

    def __init__(
        self,
        max_proposals: int = 5,
        allow_auto_apply: bool = False,
    ) -> None:
        self.max_proposals = int(max(1, max_proposals))
        self.allow_auto_apply = bool(allow_auto_apply)
        self.applied_history: List[ApplyResult] = []
        self._rollback_stack: List[Dict[str, Any]] = []
        self._rollback_counter = 0

    def propose(self, telemetry: Dict[str, object]) -> List[ImprovementProposal]:
        proposals: List[ImprovementProposal] = []
        health = float(telemetry.get("health_score", 1.0))
        failure_signal = float(telemetry.get("failure_budget_signal", 0.0))

        if health < 0.5:
            proposals.append(
                ImprovementProposal(
                    title="Emergency stability recovery",
                    rationale="Critical health score — system stability at risk.",
                    risk="high",
                    category="stability",
                )
            )
        elif health < 0.7:
            proposals.append(
                ImprovementProposal(
                    title="Improve verification calibration",
                    rationale="Low health score indicates unstable confidence calibration.",
                    risk="medium",
                    category="performance",
                )
            )

        if failure_signal > 0.5:
            proposals.append(
                ImprovementProposal(
                    title="Aggressive failure budget tightening",
                    rationale="Severe budget pressure — system may need architectural pivot.",
                    risk="high",
                    category="stability",
                )
            )
        elif failure_signal > 0.2:
            proposals.append(
                ImprovementProposal(
                    title="Tighten failure budget thresholds",
                    rationale="Frequent budget pressure suggests runtime risk accumulation.",
                    risk="low",
                    category="stability",
                    requires_human_approval=False,
                )
            )

        # Strategy proposal
        if health > 0.8 and failure_signal < 0.1:
            proposals.append(
                ImprovementProposal(
                    title="Explore advanced reasoning strategies",
                    rationale="System is stable — safe to experiment with ToT or deeper CoT.",
                    risk="low",
                    category="strategy",
                    requires_human_approval=False,
                )
            )

        if not proposals:
            proposals.append(
                ImprovementProposal(
                    title="No-op stability hold",
                    rationale="System health is stable; keep configuration unchanged.",
                    risk="low",
                    category="general",
                )
            )
        return proposals[: self.max_proposals]

    def apply_if_safe(
        self,
        proposal: ImprovementProposal,
        current_state: Optional[Dict[str, Any]] = None,
        evaluation: Optional[Dict[str, Any]] = None,
    ) -> ApplyResult:
        """
        Auto-apply low-risk proposals, reject high-risk ones.
        """
        gate_ok, gate_reason = self._acceptance_gate(evaluation)
        if not gate_ok:
            return ApplyResult(
                applied=False,
                proposal_title=proposal.title,
                reason=gate_reason,
            )

        if proposal.risk == "high" or proposal.requires_human_approval:
            return ApplyResult(
                applied=False,
                proposal_title=proposal.title,
                reason="requires_human_approval",
            )

        if not self.allow_auto_apply:
            return ApplyResult(
                applied=False,
                proposal_title=proposal.title,
                reason="auto_apply_disabled",
            )

        # Save rollback point
        self._rollback_counter += 1
        rollback_id = self._rollback_counter
        self._rollback_stack.append({
            "rollback_id": rollback_id,
            "state": current_state or {},
            "proposal": proposal.title,
            "timestamp": time.time(),
        })

        result = ApplyResult(
            applied=True,
            proposal_title=proposal.title,
            reason="auto_applied_low_risk_gate_pass",
            rollback_id=rollback_id,
        )
        self.applied_history.append(result)
        return result

    @staticmethod
    def _acceptance_gate(evaluation: Optional[Dict[str, Any]]) -> tuple[bool, str]:
        """
        Improvement acceptance rule:
        delta_benchmark > 0 && delta_safety >= 0 && cost_within_budget
        """
        if evaluation is None:
            return True, "gate_not_provided"

        delta_benchmark = float(evaluation.get("delta_benchmark", 0.0))
        delta_safety = float(evaluation.get("delta_safety", 0.0))
        cost_within_budget = bool(evaluation.get("cost_within_budget", False))

        if delta_benchmark <= 0.0:
            return False, "delta_benchmark_non_positive"
        if delta_safety < 0.0:
            return False, "delta_safety_regression"
        if not cost_within_budget:
            return False, "cost_budget_exceeded"
        return True, "gate_pass"

    def rollback_last(self) -> Optional[Dict[str, Any]]:
        """Undo last auto-applied improvement."""
        if not self._rollback_stack:
            return None
        return self._rollback_stack.pop()

    def applied_count(self) -> int:
        return sum(1 for r in self.applied_history if r.applied)

    def summary(self) -> Dict[str, Any]:
        return {
            "auto_apply_enabled": self.allow_auto_apply,
            "total_applied": self.applied_count(),
            "rollback_stack_size": len(self._rollback_stack),
            "history": [
                {"title": r.proposal_title, "applied": r.applied, "reason": r.reason}
                for r in self.applied_history[-5:]
            ],
        }
