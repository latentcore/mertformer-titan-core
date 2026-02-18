"""Compute and energy orchestration (local simulation backend)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ComputeNode:
    node_id: str
    backend: str
    capacity_score: float
    energy_score: float


class ComputeOrchestrator:
    """
    Local simulation for global orchestration logic.
    """

    def __init__(self) -> None:
        self.nodes = [
            ComputeNode("local-cpu", "cpu", 0.4, 0.9),
            ComputeNode("local-mps", "mps", 0.7, 0.8),
            ComputeNode("remote-sim-gpu", "sim_gpu", 0.9, 0.5),
        ]

    def schedule(self, workload: Dict[str, float]) -> Dict[str, object]:
        perf_weight = float(workload.get("performance_priority", 0.5))
        energy_weight = float(workload.get("energy_priority", 0.5))
        scored: List[tuple[float, ComputeNode]] = []
        for node in self.nodes:
            score = perf_weight * node.capacity_score + energy_weight * node.energy_score
            scored.append((score, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        return {
            "selected_node": best.node_id,
            "backend": best.backend,
            "score": scored[0][0],
            "candidates": [{"node_id": n.node_id, "score": s} for s, n in scored],
        }

