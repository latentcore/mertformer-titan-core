"""
Failure budget monitoring for long-running training runs.

NOTE: This module lives under orchestrator/ and is INERT / OUT-OF-SCOPE for the
45K training path (orchestrator is closed/feature-flagged off in that path). Nothing
imports FailureBudget; it is retained as a design reference, not wired into any run.

[2026-07-08] The 45K path DOES now have a loss-divergence circuit breaker, but it is a
separate, deliberately lighter implementation in `utils/divergence_guard.py`, ported from
this file's slope-tracking idea rather than promoted from it: `FailureBudget` keys off
wall-clock slope-per-hour (`time.time()`), which is neither reproducible nor testable
without freezing the clock. This file stays exactly where it is, unused — consistent with
the sealed repo rule that inert code gets real bugs fixed and honest labels, never promotion.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple

from .telemetry import LossSlopeTracker


@dataclass
class FailureBudgetConfig:
    max_hours_no_progress: float = 72.0
    min_loss_slope_per_hour: float = -0.001
    min_router_entropy: float = 0.2
    min_expert_balance: float = 0.2
    window: int = 20


@dataclass
class FailureBudgetState:
    last_progress_ts: float = field(default_factory=lambda: time.time())
    loss_history: List[Tuple[float, float]] = field(default_factory=list)
    last_router_entropy: Optional[float] = None
    last_expert_balance: Optional[float] = None


class FailureBudget:
    def __init__(self, config: Optional[FailureBudgetConfig] = None, state: Optional[FailureBudgetState] = None):
        self.config = config or FailureBudgetConfig()
        self.state = state or FailureBudgetState()
        self._slope = LossSlopeTracker(window=self.config.window)
        for ts, loss in self.state.loss_history:
            self._slope.update(loss, timestamp=ts)

    def update(
        self,
        loss: float,
        *,
        router_entropy: Optional[float] = None,
        expert_balance: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, float]:
        ts = float(timestamp if timestamp is not None else time.time())
        self._slope.update(loss, timestamp=ts)
        self.state.loss_history.append((ts, float(loss)))
        if len(self.state.loss_history) > self.config.window:
            self.state.loss_history = self.state.loss_history[-self.config.window :]

        if router_entropy is not None:
            self.state.last_router_entropy = float(router_entropy)
        if expert_balance is not None:
            self.state.last_expert_balance = float(expert_balance)

        slope = self._slope.slope_per_hour()
        progress = slope is not None and slope <= self.config.min_loss_slope_per_hour
        balance_ok = True
        if self.state.last_router_entropy is not None:
            balance_ok = balance_ok and (self.state.last_router_entropy >= self.config.min_router_entropy)
        if self.state.last_expert_balance is not None:
            balance_ok = balance_ok and (self.state.last_expert_balance >= self.config.min_expert_balance)

        if progress and balance_ok:
            self.state.last_progress_ts = ts

        hours_since_progress = (ts - self.state.last_progress_ts) / 3600.0
        should_pivot = hours_since_progress >= self.config.max_hours_no_progress

        return {
            "loss_slope_per_hour": slope if slope is not None else 0.0,
            "hours_since_progress": hours_since_progress,
            "should_pivot": 1.0 if should_pivot else 0.0,
        }

    def to_json(self) -> str:
        payload = {
            "config": asdict(self.config),
            "state": asdict(self.state),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @staticmethod
    def from_json(raw: str) -> "FailureBudget":
        data = json.loads(raw)
        config = FailureBudgetConfig(**data.get("config", {}))
        state = FailureBudgetState(**data.get("state", {}))
        return FailureBudget(config=config, state=state)
