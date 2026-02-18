"""Deterministic 3/15/45-agent swarm runtime."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

from .agent_registry import AgentSpec, get_profile_specs
from .failure_budget import FailureBudget
from .governance import GovernanceDecision, GovernanceGate, GovernancePolicy
from .planner import SwarmPlanner
from .telemetry import runtime_health_report, system_snapshot
from .verifier import SwarmVerifier


@dataclass
class SwarmRunReport:
    mode: str
    task: str
    governance: Dict[str, object]
    selected_agents: List[str]
    outputs: List[str]
    verification: Dict[str, object]
    telemetry: Dict[str, object]


class SwarmRuntime:
    def __init__(
        self,
        *,
        planner: Optional[SwarmPlanner] = None,
        verifier: Optional[SwarmVerifier] = None,
        governance: Optional[GovernanceGate] = None,
        failure_budget: Optional[FailureBudget] = None,
        generate_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.planner = planner or SwarmPlanner()
        self.verifier = verifier or SwarmVerifier()
        self.governance = governance or GovernanceGate(GovernancePolicy())
        self.failure_budget = failure_budget or FailureBudget()
        self.generate_fn = generate_fn

    def _emit_agent_output(self, action_text: str, agent: AgentSpec) -> str:
        if self.generate_fn is None:
            return f"[{agent.agent_id}:{agent.role}] {action_text}"
        try:
            result = self.generate_fn(action_text)
        except Exception as exc:  # pragma: no cover
            return f"[{agent.agent_id}:{agent.role}] generation_error={exc}"
        return f"[{agent.agent_id}:{agent.role}] {result}".strip()

    def run(self, task: str, *, mode: str = "nano") -> SwarmRunReport:
        mode_key = str(mode).lower().strip()
        profile = get_profile_specs(mode_key)

        gov: GovernanceDecision = self.governance.evaluate(task, requested_actions=["plan", "verify", "report"])
        if not gov.allowed:
            report = SwarmRunReport(
                mode=mode_key,
                task=task,
                governance={"allowed": False, "reasons": gov.reasons},
                selected_agents=[],
                outputs=[],
                verification={"pass_check": False, "confidence": 0.0, "uncertainty": 1.0, "notes": ["governance_block"]},
                telemetry={"snapshot": system_snapshot(), "failure_budget": {}},
            )
            return report

        plan = self.planner.plan(task, profile, mode_key)
        profile_by_id = {spec.agent_id: spec for spec in profile}

        outputs: List[str] = []
        for action in plan:
            spec = profile_by_id[action.agent_id]
            objective = action.objective
            if action.tool_id:
                objective = f"{objective} [tool={action.tool_id}]"
            outputs.append(self._emit_agent_output(objective, spec))

        verification = self.verifier.verify(task, outputs)
        fb = self.failure_budget.update(
            max(0.0, 1.0 - verification.confidence),
            router_entropy=verification.confidence,
            expert_balance=verification.confidence,
        )

        snapshot = system_snapshot()
        telemetry = {
            "snapshot": snapshot,
            "failure_budget": fb,
            "health_report": runtime_health_report(
                snapshot=snapshot,
                verification_confidence=verification.confidence,
                failure_budget_signal=fb,
            ),
        }

        return SwarmRunReport(
            mode=mode_key,
            task=task,
            governance={"allowed": True, "reasons": gov.reasons},
            selected_agents=[item.agent_id for item in plan],
            outputs=outputs,
            verification=asdict(verification),
            telemetry=telemetry,
        )
