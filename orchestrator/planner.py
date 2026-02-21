"""Deterministic swarm planning and role assignment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .agent_registry import AgentSpec
from .tool_registry import ToolSpec, default_tool_registry


@dataclass(frozen=True)
class PlannedAction:
    agent_id: str
    role: str
    objective: str
    tool_id: str | None = None


class SwarmPlanner:
    """Simple keyword-capability scorer with deterministic tie-break."""

    def __init__(
        self,
        max_actions_by_mode: dict[str, int] | None = None,
        tool_registry: Dict[str, ToolSpec] | None = None,
    ) -> None:
        self.max_actions_by_mode = max_actions_by_mode or {
            "nano": 3,
            "mid": 10,
            "omega": 45,
        }
        self.tool_registry = tool_registry or default_tool_registry()
        # Keep planner and runtime registry aligned by default.
        self.tool_allowlist = set(self.tool_registry.keys())
        self.tool_denylist_tokens = {"network", "download", "exfiltrate", "stealth", "covert"}

    def _select_tool(self, task: str, role: str) -> str | None:
        task_l = (task or "").lower()
        role_l = (role or "").lower()

        if any(t in task_l for t in self.tool_denylist_tokens):
            return None

        candidates: List[tuple[int, str]] = []
        for tool_id, spec in self.tool_registry.items():
            if tool_id not in self.tool_allowlist:
                continue
            score = 0
            if role_l in tool_id:
                score += 2
            for cap in spec.capabilities:
                if cap in task_l:
                    score += 2
                for tok in cap.split("_"):
                    if tok in task_l:
                        score += 1
            candidates.append((score, tool_id))

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][1] if candidates and candidates[0][0] > 0 else None

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
                tool_id=self._select_tool(task, spec.role),
            )
            for spec in selected
        ]
