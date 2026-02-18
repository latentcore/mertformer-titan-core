"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Research extension layers (feature-flag driven, non-breaking defaults)
==============================================================================
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalWorkspaceBroadcast(nn.Module):
    """Broadcast a global workspace vector back to token states."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, workspace: Optional[torch.Tensor]) -> torch.Tensor:
        if workspace is None:
            return x
        workspace = workspace.to(device=x.device, dtype=x.dtype)
        signal = torch.tanh(self.proj(workspace)).unsqueeze(1)
        return x + signal * torch.sigmoid(self.gate)


class ContinuousLatentODEStateChannel(nn.Module):
    """Simple continuous-time latent ODE channel with persistent runtime state."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.state_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.input_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.register_buffer("latent_state", torch.zeros(1, hidden_size), persistent=False)

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        with torch.no_grad():
            self.latent_state = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def _ensure_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self.latent_state.numel() == 0
            or self.latent_state.shape[0] != batch_size
            or self.latent_state.device != device
            or self.latent_state.dtype != dtype
        ):
            self.reset_state(batch_size, device, dtype)

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        bsz = x.size(0)
        self._ensure_state(bsz, x.device, x.dtype)
        summary = x.mean(dim=1)
        z = self.latent_state
        dz = torch.tanh(self.state_proj(z) + self.input_proj(summary))
        z_next = z + float(dt) * dz
        with torch.no_grad():
            self.latent_state.copy_(z_next.detach())
        return x + self.out_proj(z_next).unsqueeze(1).to(dtype=x.dtype)


class NeuromodulatoryGainLayer(nn.Module):
    """Global gain modulation driven by workspace summary."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gain_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.bias_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.gain_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, workspace: Optional[torch.Tensor]) -> torch.Tensor:
        if workspace is None:
            return x
        workspace = workspace.to(device=x.device, dtype=x.dtype)
        gain = torch.sigmoid(self.gain_proj(workspace)).unsqueeze(1)
        bias = torch.tanh(self.bias_proj(workspace)).unsqueeze(1)
        return x * (1.0 + gain * self.gain_scale) + bias * self.gain_scale


class HebbianPlasticityLayer(nn.Module):
    """Lightweight Hebbian trace (diagonal local plasticity)."""

    def __init__(self, hidden_size: int, eta: float = 0.01, decay: float = 0.99) -> None:
        super().__init__()
        self.eta = float(eta)
        self.decay = float(decay)
        self.register_buffer("trace", torch.zeros(hidden_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                activity = x.detach().pow(2).mean(dim=(0, 1)).to(self.trace.dtype)
                self.trace.mul_(self.decay).add_(activity * (1.0 - self.decay))
        gain = 1.0 + self.eta * torch.tanh(self.trace.to(device=x.device, dtype=x.dtype))
        return x * gain.view(1, 1, -1)


class NeuroSymbolicLayer(nn.Module):
    """
    Neural-symbolic bridge:
    neural state -> rule selection -> rule-conditioned residual.
    """

    def __init__(self, hidden_size: int, num_rules: int = 8) -> None:
        super().__init__()
        self.num_rules = int(max(1, num_rules))
        self.rule_keys = nn.Parameter(torch.randn(self.num_rules, hidden_size) * 0.02)
        self.rule_values = nn.Parameter(torch.randn(self.num_rules, hidden_size) * 0.02)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rule_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summary = x.mean(dim=1)  # [B, H]
        logits = torch.matmul(summary, self.rule_keys.t())  # [B, R]
        weights = F.softmax(logits, dim=-1)
        rule_context = torch.matmul(weights, self.rule_values)  # [B, H]
        residual = torch.tanh(self.out_proj(rule_context)).unsqueeze(1)
        return x + residual * torch.sigmoid(self.rule_gain)
