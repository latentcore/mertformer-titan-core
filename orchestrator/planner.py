"""Deterministic swarm planning and role assignment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .agent_registry import AgentSpec


@dataclass(frozen=True)
class PlannedAction:
    agent_id: str
    role: str
    objective: str


class SwarmPlanner:
    """Simple keyword-capability scorer with deterministic tie-break."""

    def __init__(self, max_actions_by_mode: dict[str, int] | None = None) -> None:
        self.max_actions_by_mode = max_actions_by_mode or {
            "nano": 3,
            "mid": 10,
            "omega": 45,
        }

    def plan(self, task: str, agents: Sequence[AgentSpec], mode: str) -> List[PlannedAction]:
        normalized_task = (task or "").lower()
        scored: list[tuple[int, int, AgentSpec]] = []

        for spec in agents:
            score = 0
            for cap in spec.capabilities:
                if cap in normalized_task:
                    score += 3
                for token in cap.split("_"):
                    if token in normalized_task:
                        score += 1
            scored.append((score, -spec.priority, spec))

        scored.sort(key=lambda item: (item[0], item[1], item[2].agent_id), reverse=True)
        limit = max(1, int(self.max_actions_by_mode.get(mode, len(agents))))

        selected = [item[2] for item in scored[: min(limit, len(scored))]]
        # Deterministic execution order by priority then id.
        selected.sort(key=lambda spec: (spec.priority, spec.agent_id))

        return [
            PlannedAction(
                agent_id=spec.agent_id,
                role=spec.role,
                objective=f"{spec.role} executes policy-bound analysis for task: {task}",
            )
            for spec in selected
        ]
