"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
World model dynamics head (feature-flag driven, non-breaking)
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class WorldModelOutput:
    """Compact container for world-model side outputs."""

    dynamics_logits: torch.Tensor
    latent_state: torch.Tensor
    uncertainty: torch.Tensor
    counterfactual_logits: torch.Tensor
    risk_score: torch.Tensor

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "world_dynamics_logits": self.dynamics_logits,
            "world_latent_state": self.latent_state,
            "world_uncertainty": self.uncertainty,
            "world_counterfactual_logits": self.counterfactual_logits,
            "world_risk_score": self.risk_score,
        }


class CausalWorldModelHead(nn.Module):
    """
    Lightweight causal dynamics head.

    This module does not change the model forward signature. It computes
    side-channel outputs for diagnostics/planning pathways.
    """

    def __init__(self, hidden_size: int, horizon: int = 1) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.horizon = int(max(1, horizon))
        self.pre = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dynamics = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.counterfactual = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.uncertainty = nn.Linear(self.hidden_size, 1, bias=True)
        self.risk = nn.Linear(self.hidden_size, 1, bias=True)

    def forward(self, x: torch.Tensor) -> WorldModelOutput:
        # x: [B, T, H]
        summary = x.mean(dim=1)  # [B, H]
        latent = torch.tanh(self.pre(summary))

        dyn_steps = []
        state = latent
        for _ in range(self.horizon):
            state = torch.tanh(self.dynamics(state))
            dyn_steps.append(state)
        stacked = torch.stack(dyn_steps, dim=1)  # [B, horizon, H]

        cf_steps = []
        cf_state = torch.tanh(self.counterfactual(-latent))
        for _ in range(self.horizon):
            cf_state = torch.tanh(self.dynamics(cf_state))
            cf_steps.append(cf_state)
        counterfactual = torch.stack(cf_steps, dim=1)  # [B, horizon, H]

        unc = torch.sigmoid(self.uncertainty(latent)).squeeze(-1)  # [B]
        risk = torch.sigmoid(self.risk(latent - cf_state)).squeeze(-1)  # [B]
        return WorldModelOutput(
            dynamics_logits=stacked,
            latent_state=latent,
            uncertainty=unc,
            counterfactual_logits=counterfactual,
            risk_score=risk,
        )
