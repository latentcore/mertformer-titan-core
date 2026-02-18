"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Lifelong safety/adaptation guard (feature-flag, non-breaking)
==============================================================================
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class LifelongSafetyLayer(nn.Module):
    """
    Policy-constrained adaptation layer.

    - Tracks running activation profile.
    - Dampens abrupt distribution shifts.
    - Keeps adaptation bounded for safety-critical stability.
    """

    def __init__(
        self,
        hidden_size: int,
        ema_decay: float = 0.99,
        max_adaptation_gain: float = 0.05,
        drift_threshold: float = 0.35,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.ema_decay = float(ema_decay)
        self.max_adaptation_gain = float(max_adaptation_gain)
        self.drift_threshold = float(drift_threshold)

        self.register_buffer("running_mean", torch.zeros(self.hidden_size), persistent=False)
        self.register_buffer("running_var", torch.ones(self.hidden_size), persistent=False)
        self.register_buffer("last_drift", torch.zeros(()), persistent=False)
        self.gain = nn.Parameter(torch.zeros(self.hidden_size))

    def _update_stats(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            mean = x.detach().mean(dim=(0, 1))
            var = x.detach().var(dim=(0, 1), unbiased=False)
            self.running_mean.mul_(self.ema_decay).add_(mean * (1.0 - self.ema_decay))
            self.running_var.mul_(self.ema_decay).add_(var * (1.0 - self.ema_decay))

            drift = (mean - self.running_mean).abs().mean()
            self.last_drift.copy_(drift.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"LifelongSafetyLayer expects [B,T,H], got {tuple(x.shape)}")

        self._update_stats(x)

        # Keep adaptation bounded and deterministic.
        bounded = torch.tanh(self.gain).to(device=x.device, dtype=x.dtype)
        scale = torch.clamp(torch.tensor(self.max_adaptation_gain, device=x.device, dtype=x.dtype), min=0.0)

        if float(self.last_drift.item()) > self.drift_threshold:
            # If drift is high, reduce adaptation (stability-first).
            scale = scale * 0.5

        return x * (1.0 + bounded.view(1, 1, -1) * scale)

    def safety_metrics(self) -> Dict[str, float]:
        return {
            "last_drift": float(self.last_drift.item()),
            "ema_decay": self.ema_decay,
            "max_adaptation_gain": self.max_adaptation_gain,
            "drift_threshold": self.drift_threshold,
        }

