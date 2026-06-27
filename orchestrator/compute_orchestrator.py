"""Compute and energy orchestration (local simulation backend).

INERT / OUT-OF-SCOPE: bu modul 45K egitim yolunda kapali (feature-flag);
yalnizca yerel orkestrasyon mantigini simule eder, gercek backend secimi /
enerji telemetrisi YOKTUR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ComputeNode:
    node_id: str
    backend: str
    # capacity_score / energy_score: elle girilmis HEURISTIK agirliklardir;
    # gercek kapasite olcumu veya enerji telemetrisi DEGILDIR. 'energy_score'
    # adi enerji telemetrisi imasi verse de simulasyon icin sabit bir agirliktir.
    capacity_score: float
    energy_score: float


class ComputeOrchestrator:
    """
    Local simulation for global orchestration logic.
    """

    def __init__(self) -> None:
        # SIMULASYON node'lari: 'remote-sim-gpu'/'sim_gpu' gercek bir backend
        # degildir; skorlar (0.4/0.9, 0.7/0.8, 0.9/0.5) elle yazilmis sabitlerdir.
        self.nodes = [
            ComputeNode("local-cpu", "cpu", 0.4, 0.9),
            ComputeNode("local-mps", "mps", 0.7, 0.8),
            ComputeNode("remote-sim-gpu", "sim_gpu", 0.9, 0.5),
        ]

    def schedule(self, workload: Dict[str, float]) -> Dict[str, object]:
        # SIMULE EDILMIS oneri: skorlar elle yazilmis heuristik agirliklardan
        # turetilir; gercek olculmus kapasite/enerji degil. Cikti 'simulated'
        # alaniyla isaretlenir ki cagri yuzeyi gercek backend secimi sanmasin.
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
            # Heuristik simulasyon oneresi; gercek backend secimi / enerji
            # olcumu DEGIL. Cikti tuketicileri bunu gecme-kapisi saymamalidir.
            "simulated": True,
            "note": "heuristic_simulation_not_measured",
        }

