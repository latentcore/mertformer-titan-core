from __future__ import annotations
"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - AGI EXPERIENCE STORE
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================

TR: Deneyim deposu — episodik öğrenme, strateji adaptasyonu, performans izleme.
EN: Experience store — episodic learning, strategy adaptation, performance tracking.
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert"

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# TR: VERİ YAPILARI / EN: DATA STRUCTURES
# -----------------------------------------------------------------------------

@dataclass
class Episode:
    """TR: Tek bir deneyim bölümü. / EN: A single experience episode."""
    task: str
    strategy_used: str  # "direct", "cot", "tot"
    thoughts: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    outcome_score: float = 0.0  # 0.0 - 1.0
    audit_score: float = 0.0
    reflection: str = ""
    iterations: int = 1
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """TR: Strateji performans metrikleri. / EN: Strategy performance metrics."""
    strategy: str
    total_uses: int = 0
    avg_score: float = 0.0
    success_rate: float = 0.0
    avg_iterations: float = 0.0
    avg_tool_count: float = 0.0
    best_score: float = 0.0
    worst_score: float = 1.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy,
            "total_uses": self.total_uses,
            "avg_score": round(self.avg_score, 4),
            "success_rate": round(self.success_rate, 4),
            "avg_iterations": round(self.avg_iterations, 2),
            "avg_tool_count": round(self.avg_tool_count, 2),
            "best_score": round(self.best_score, 4),
            "worst_score": round(self.worst_score, 4),
        }


# -----------------------------------------------------------------------------
# TR: DENEYİM DEPOSU / EN: EXPERIENCE STORE
# -----------------------------------------------------------------------------

class ExperienceStore:
    """
    TR: Deneyimlerden öğrenen kalıcı depo.
    EN: Persistent store that learns from experiences.

    - JSONL tabanlı kalıcılık (GodMemory ile tutarlı)
    - Semantik benzerlik ile geri çağırma (SenseEngine opsiyonel)
    - Strateji performans izleme → MetaLearner'a veri sağlar
    - FIFO eviction (max kapasiteye ulaşıldığında)
    """

    MAX_EPISODES = 10_000
    SUCCESS_THRESHOLD = 0.6

    def __init__(
        self,
        store_path: Optional[Path] = None,
        sense_engine: Optional[Any] = None,
    ) -> None:
        self.store_path = Path(store_path) if store_path else None
        self.sense_engine = sense_engine
        self.episodes: List[Episode] = []
        self._strategy_stats: Dict[str, StrategyPerformance] = {}
        self._load()

    def _load(self) -> None:
        """TR: Kalıcı depodan yükle. / EN: Load from persistent store."""
        if self.store_path is None or not self.store_path.exists():
            return

        try:
            with self.store_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        episode = Episode(
                            task=data.get("task", ""),
                            strategy_used=data.get("strategy_used", "direct"),
                            thoughts=data.get("thoughts", []),
                            actions=data.get("actions", []),
                            tools_used=data.get("tools_used", []),
                            outcome_score=float(data.get("outcome_score", 0.0)),
                            audit_score=float(data.get("audit_score", 0.0)),
                            reflection=data.get("reflection", ""),
                            iterations=int(data.get("iterations", 1)),
                            timestamp=float(data.get("timestamp", 0.0)),
                            metadata=data.get("metadata", {}),
                        )
                        self.episodes.append(episode)
                        self._update_stats(episode)
                    except Exception:
                        continue
            print(f"📚 Experience Store: {len(self.episodes)} episodes loaded")
        except Exception as e:
            print(f"⚠️ Experience Store load error: {e}")

    def _save_episode(self, episode: Episode) -> None:
        """TR: Tek bir bölümü diske yaz. / EN: Write a single episode to disk."""
        if self.store_path is None:
            return
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with self.store_path.open("a", encoding="utf-8") as f:
                data = {
                    "task": episode.task,
                    "strategy_used": episode.strategy_used,
                    "thoughts": episode.thoughts,
                    "actions": episode.actions,
                    "tools_used": episode.tools_used,
                    "outcome_score": episode.outcome_score,
                    "audit_score": episode.audit_score,
                    "reflection": episode.reflection,
                    "iterations": episode.iterations,
                    "timestamp": episode.timestamp,
                    "metadata": {
                        k: v for k, v in episode.metadata.items()
                        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                    },
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️ Experience save error: {e}")

    def record_episode(self, episode: Episode) -> None:
        """TR: Yeni deneyim kaydet. / EN: Record new experience."""
        self.episodes.append(episode)
        self._update_stats(episode)
        self._save_episode(episode)

        # TR: FIFO eviction / EN: FIFO eviction
        if len(self.episodes) > self.MAX_EPISODES:
            self.episodes = self.episodes[-self.MAX_EPISODES:]

    def recall_similar(
        self,
        task: str,
        top_k: int = 5,
    ) -> List[Episode]:
        """
        TR: Benzer görevlerdeki deneyimleri hatırla.
        EN: Recall experiences from similar tasks.
        """
        if not self.episodes:
            return []

        # TR: Semantic recall varsa kullan / EN: Use semantic recall if available
        if self.sense_engine is not None:
            return self._semantic_recall(task, top_k)

        # TR: Keyword-based fallback / EN: Keyword-based fallback
        return self._keyword_recall(task, top_k)

    def _semantic_recall(self, task: str, top_k: int) -> List[Episode]:
        """TR: Semantik benzerlik ile geri çağırma. / EN: Recall by semantic similarity."""
        try:
            q_vec = torch.tensor(
                self.sense_engine.encode_text(task), dtype=torch.float32
            )

            scored: List[tuple[float, Episode]] = []
            for episode in self.episodes:
                e_vec = torch.tensor(
                    self.sense_engine.encode_text(episode.task), dtype=torch.float32
                )
                if q_vec.numel() != e_vec.numel():
                    continue
                with torch.no_grad():
                    sim = float(
                        F.cosine_similarity(q_vec.unsqueeze(0), e_vec.unsqueeze(0)).item()
                    )
                scored.append((sim, episode))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [ep for _, ep in scored[:top_k]]
        except Exception:
            return self._keyword_recall(task, top_k)

    def _keyword_recall(self, task: str, top_k: int) -> List[Episode]:
        """TR: Anahtar kelime tabanlı geri çağırma. / EN: Keyword-based recall."""
        task_words = set(task.lower().split())
        scored: List[tuple[int, Episode]] = []

        for episode in self.episodes:
            ep_words = set(episode.task.lower().split())
            overlap = len(task_words & ep_words)
            if overlap > 0:
                scored.append((overlap, episode))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def _update_stats(self, episode: Episode) -> None:
        """TR: Strateji istatistiklerini güncelle. / EN: Update strategy statistics."""
        strategy = episode.strategy_used
        if strategy not in self._strategy_stats:
            self._strategy_stats[strategy] = StrategyPerformance(strategy=strategy)

        stats = self._strategy_stats[strategy]
        n = stats.total_uses
        stats.total_uses += 1

        # TR: Çevrimiçi ortalama güncelleme / EN: Online average update
        stats.avg_score = (stats.avg_score * n + episode.outcome_score) / (n + 1)
        stats.avg_iterations = (stats.avg_iterations * n + episode.iterations) / (n + 1)
        stats.avg_tool_count = (
            (stats.avg_tool_count * n + len(episode.tools_used)) / (n + 1)
        )

        # TR: Başarı oranı güncelleme / EN: Success rate update
        is_success = 1.0 if episode.outcome_score >= self.SUCCESS_THRESHOLD else 0.0
        stats.success_rate = (stats.success_rate * n + is_success) / (n + 1)

        # TR: En iyi / en kötü / EN: Best / worst
        stats.best_score = max(stats.best_score, episode.outcome_score)
        stats.worst_score = min(stats.worst_score, episode.outcome_score)

    def strategy_stats(self) -> Dict[str, StrategyPerformance]:
        """TR: Strateji performans istatistikleri. / EN: Strategy performance statistics."""
        return dict(self._strategy_stats)

    def best_strategy_for(self, task_type: str = "") -> str:
        """
        TR: Verilen görev tipi için en iyi stratejiyi belirler.
        EN: Determines the best strategy for the given task type.
        """
        if not self._strategy_stats:
            return "cot"  # TR: Varsayılan strateji / EN: Default strategy

        # TR: Benzer görevlerdeki geçmiş stratejileri kontrol et
        # EN: Check past strategies for similar tasks
        if task_type:
            similar = self.recall_similar(task_type, top_k=10)
            if similar:
                # TR: En başarılı stratejiyi bul / EN: Find most successful strategy
                strategy_scores: Dict[str, List[float]] = {}
                for ep in similar:
                    if ep.strategy_used not in strategy_scores:
                        strategy_scores[ep.strategy_used] = []
                    strategy_scores[ep.strategy_used].append(ep.outcome_score)

                if strategy_scores:
                    best = max(
                        strategy_scores.items(),
                        key=lambda x: sum(x[1]) / len(x[1]),
                    )
                    return best[0]

        # TR: Genel en iyi strateji / EN: Overall best strategy
        if self._strategy_stats:
            best_stat = max(
                self._strategy_stats.values(),
                key=lambda s: s.avg_score if s.total_uses >= 3 else 0.0,
            )
            if best_stat.total_uses >= 3:
                return best_stat.strategy

        return "cot"

    def summary(self) -> Dict[str, object]:
        """TR: Depo özet raporu. / EN: Store summary report."""
        return {
            "total_episodes": len(self.episodes),
            "strategies": {
                k: v.to_dict() for k, v in self._strategy_stats.items()
            },
            "best_overall_strategy": self.best_strategy_for(),
            "most_recent_score": (
                self.episodes[-1].outcome_score if self.episodes else None
            ),
        }
