"""
Continual learning adapter (lightweight, offline-safe).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class ContinualAdapterState:
    step: int
    replay_size: int
    running_loss_ema: float
    drift_alert: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = int(max(1, capacity))
        # deque(maxlen=...) bounds the buffer in O(1) and drops the oldest item
        # automatically once full — no per-add reslicing of the whole list.
        self._items: "deque[torch.Tensor]" = deque(maxlen=self.capacity)

    def add(self, sample: torch.Tensor) -> None:
        self._items.append(sample.detach().cpu())

    def sample(self, k: int = 16) -> List[torch.Tensor]:
        # Deterministic tail window: the k most-recently-added samples, in
        # insertion order (oldest -> newest). Not a random sample.
        if not self._items:
            return []
        k = min(int(max(1, k)), len(self._items))
        return list(self._items)[-k:]

    def __len__(self) -> int:
        return len(self._items)


class ContinualLearningAdapter:
    """
    Tracks drift and maintains replay buffer metadata.
    This is intentionally lightweight for local/mac environments.
    """

    def __init__(
        self,
        replay_capacity: int = 2048,
        loss_ema_decay: float = 0.98,
        drift_threshold: float = 0.2,
    ) -> None:
        self.replay = ReplayBuffer(replay_capacity)
        self.loss_ema_decay = float(loss_ema_decay)
        self.drift_threshold = float(drift_threshold)
        self.step = 0
        self.running_loss_ema = 0.0
        self.prev_loss_ema = 0.0
        self.drift_alert = False

    def update(self, *, loss: float, sample: torch.Tensor | None = None) -> ContinualAdapterState:
        self.step += 1
        self.prev_loss_ema = self.running_loss_ema
        if self.step == 1:
            # Cold-start: seed the EMA with the first observed loss rather than
            # blending against the fictitious 0.0 init (which biased early steps
            # toward zero and produced a spurious first-step "drift").
            self.running_loss_ema = float(loss)
            self.prev_loss_ema = float(loss)
        else:
            self.running_loss_ema = (
                self.running_loss_ema * self.loss_ema_decay + float(loss) * (1.0 - self.loss_ema_decay)
            )
        drift = abs(self.running_loss_ema - self.prev_loss_ema)
        self.drift_alert = drift > self.drift_threshold

        if sample is not None:
            self.replay.add(sample)

        return self.state()

    def state(self) -> ContinualAdapterState:
        return ContinualAdapterState(
            step=self.step,
            replay_size=len(self.replay),
            running_loss_ema=float(self.running_loss_ema),
            drift_alert=bool(self.drift_alert),
        )

    def to_dict(self) -> Dict[str, float | int | bool]:
        s = self.state()
        return {
            "step": s.step,
            "replay_size": s.replay_size,
            "running_loss_ema": s.running_loss_ema,
            "drift_alert": s.drift_alert,
            "drift_threshold": self.drift_threshold,
        }

