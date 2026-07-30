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
        # Adaptation scale is a non-negative constant; precompute it once instead of
        # rebuilding a clamped tensor on every forward (identical math, less per-step work).
        self._scale = max(0.0, self.max_adaptation_gain)

        self.register_buffer("running_mean", torch.zeros(self.hidden_size), persistent=False)
        self.register_buffer("last_drift", torch.zeros(()), persistent=False)
        self.gain = nn.Parameter(torch.zeros(self.hidden_size))

    def _update_stats(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            mean = x.detach().mean(dim=(0, 1))

            # [2026-07-29] Drift MUST be measured against the PREVIOUS running mean, i.e.
            # before the EMA absorbs this batch. It used to be computed after the update:
            #   running' = d*running + (1-d)*mean
            #   mean - running' = (1-d) * (mean - running)
            # so the reported drift was (1-ema_decay) x the real drift -- at the default
            # ema_decay=0.99 that is 1% of it. `drift_threshold=0.35` was therefore
            # unreachable in practice and the stability-first damping branch in forward()
            # (scale *= 0.5) was dead code. The layer is feature-flagged off by default,
            # so this changes no canonical-run behaviour -- but a component named
            # "safety layer" whose safety trigger can never fire is worse than no layer.
            drift = (mean - self.running_mean).abs().mean()
            self.last_drift.copy_(drift.detach())

            self.running_mean.mul_(self.ema_decay).add_(mean * (1.0 - self.ema_decay))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"LifelongSafetyLayer expects [B,T,H], got {tuple(x.shape)}")

        # [P12] EMA running stats should drift only during training, not at eval/inference.
        if self.training:
            self._update_stats(x)

        # Keep adaptation bounded and deterministic.
        bounded = torch.tanh(self.gain).to(device=x.device, dtype=x.dtype)
        scale = self._scale

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

