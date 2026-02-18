"""Policy boundary checks for swarm orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass(frozen=True)
class GovernancePolicy:
    offline_only: bool = True
    allow_network: bool = False
    allow_shell: bool = False
    blocked_keywords: tuple[str, ...] = (
        "http://",
        "https://",
        "socket",
        "subprocess",
        "curl",
        "wget",
    )
    allowed_actions: tuple[str, ...] = (
        "analyze",
        "plan",
        "verify",
        "report",
        "summarize",
    )


@dataclass
class GovernanceDecision:
    allowed: bool
    reasons: List[str] = field(default_factory=list)


class GovernanceGate:
    def __init__(self, policy: GovernancePolicy | None = None) -> None:
        self.policy = policy or GovernancePolicy()

    def evaluate(self, task: str, *, requested_actions: Iterable[str] | None = None) -> GovernanceDecision:
        reasons: List[str] = []
        task_text = (task or "").strip().lower()
        if not task_text:
            reasons.append("empty_task")

        if self.policy.offline_only and not self.policy.allow_network:
            for keyword in self.policy.blocked_keywords:
                if keyword in task_text:
                    reasons.append(f"network_keyword_blocked:{keyword}")

        actions = [a.strip().lower() for a in (requested_actions or []) if a and a.strip()]
        for action in actions:
            if action not in self.policy.allowed_actions:
                reasons.append(f"action_not_allowed:{action}")

        return GovernanceDecision(allowed=len(reasons) == 0, reasons=reasons)
