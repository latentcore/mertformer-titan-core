"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert"

import torch
import torch.nn as nn
from typing import Optional, Tuple

from config.config import cfg

# Import layer components
from layers.ffn import MertFormerFFN
from layers.cognitive_extensions import (
    GlobalWorkspaceBroadcast,
    HebbianPlasticityLayer,
    NeuroSymbolicLayer,
)
from layers.lifelong_safety import LifelongSafetyLayer
from layers.liquid import LiquidMixer
from layers.mla import MLA
from layers.moe import MoE
from layers.qinn import UnitaryQINN


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization - Faster alternative to LayerNorm.

    V27.0: torch.compile removed for safer export/inference control.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """
        RMSNorm initializer.

        Args:
            dim (int): Dimension to normalize
            eps (float): Epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # @torch.compile (Removed for safer export/inference control)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - Applies RMS normalization (fused kernel).

        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Normalized tensor
        """
        # Fused RMS calculation (torch.compile optimizes this)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class MertFormerBlock(nn.Module):
    """
    MertFormer Transformer Block - Combination of Attention, FFN/MoE, Liquid, QINN.

    Features:
    - Attention: MLA (LLaMA-3 Compatible)
    - FeedForward: MoE (Z-Loss fix) or SwiGLU Dense FFN
    - Stability: RMSNorm + optional UnitaryQINN
    - LiquidMixer: Optional, residual + LayerNorm
    - Aux Loss: Always returns a scalar
    """

    def __init__(self, layer_id: int) -> None:
        """
        MertFormerBlock initializer.

        Args:
            layer_id (int): Layer ID (0-indexed)
        """
        super().__init__()
        self.layer_id = int(layer_id)
        H = int(cfg.hidden_size)

        # Normalization layers
        eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
        self.norm1 = RMSNorm(H, eps=eps)
        self.norm2 = RMSNorm(H, eps=eps)
        
        # V23.0: Residual Scaling (deep network stabilization)
        # Scale residual connections based on layer depth
        # Inspired by DeepNorm: 1/sqrt(2*N) formula
        num_layers = int(getattr(cfg, "num_layers", 24))
        self.residual_scale = (2 * num_layers) ** -0.5

        # Attention mechanism
        self.attn = MLA()

        # FeedForward / MoE selection
        self.use_moe = bool(getattr(cfg, "use_moe", False))
        every_n = int(getattr(cfg, "moe_every_n_layers", 0))
        self.ff: nn.Module
        if self.use_moe and every_n > 0 and ((self.layer_id + 1) % every_n == 0):
            self.is_moe_layer = True
            self.ff = MoE()
        else:
            self.is_moe_layer = False
            self.ff = MertFormerFFN()

        # QINN optional (quantum-inspired unitary layer)
        self.qinn = None
        if getattr(cfg, "use_qinn", False) and UnitaryQINN is not None:
            q_every = int(getattr(cfg, "qinn_every_n_layers", 1))
            if q_every > 0 and ((self.layer_id + 1) % q_every == 0):
                self.qinn = UnitaryQINN(H)

        # LiquidMixer optional
        # Use explicit liquid_layers_idx (fallback to every_n if not available)
        self.liquid = None
        if getattr(cfg, "use_liquid", False):
            liquid_layers_idx = getattr(cfg, "liquid_layers_idx", None)
            if liquid_layers_idx and self.layer_id in liquid_layers_idx:
                self.liquid = LiquidMixer(
                    H,
                    fast_path=bool(getattr(cfg, "liquid_fast_path", True)),
                    train_impl=str(getattr(cfg, "liquid_train_impl", "baseline")),
                )
            else:
                # Fallback to every_n logic if indices not provided
                liq_every = int(getattr(cfg, "liquid_every_n_layers", 0))
                if liq_every > 0 and ((self.layer_id + 1) % liq_every == 0):
                    self.liquid = LiquidMixer(
                        H,
                        fast_path=bool(getattr(cfg, "liquid_fast_path", True)),
                        train_impl=str(getattr(cfg, "liquid_train_impl", "baseline")),
                    )

        # Optional global workspace broadcast
        self.workspace_layer = (
            GlobalWorkspaceBroadcast(H)
            if bool(getattr(cfg, "use_global_workspace_broadcast", False))
            else None
        )

        # Optional Hebbian plasticity
        self.hebbian_layer = (
            HebbianPlasticityLayer(
                H,
                eta=float(getattr(cfg, "hebbian_eta", 0.01)),
                decay=float(getattr(cfg, "hebbian_decay", 0.99)),
            )
            if bool(getattr(cfg, "use_hebbian_plasticity", False))
            else None
        )

        # Optional neuro-symbolic bridge
        self.neuro_symbolic_layer = (
            NeuroSymbolicLayer(H, num_rules=int(getattr(cfg, "neuro_symbolic_rules", 8)))
            if bool(getattr(cfg, "use_neuro_symbolic_layer", False))
            else None
        )
        self.lifelong_safety_layer = (
            LifelongSafetyLayer(
                H,
                ema_decay=float(getattr(cfg, "lifelong_ema_decay", 0.99)),
                max_adaptation_gain=float(getattr(cfg, "lifelong_max_adaptation_gain", 0.05)),
                drift_threshold=float(getattr(cfg, "lifelong_drift_threshold", 0.35)),
            )
            if bool(getattr(cfg, "use_lifelong_safety_layer", False))
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        workspace: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward propagation - Manages Liquid state and MoE routing.

        Args:
            x (torch.Tensor): Input tensor [Batch, Seq, Hidden]
            past_key_value (Optional[Tuple]): Previous KV cache
            use_cache (bool): Return cache?
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
                (Output, Aux loss, KV Cache)
        """
        # 1. Attention (with KV Cache)
        h = self.norm1(x)
        attn_out, present_key_value = self.attn(h, past_key_value=past_key_value, use_cache=use_cache)
        # V23.0: Residual Scaling
        x = x + attn_out * self.residual_scale

        # 2. [ARCH UPDATE] LiquidMixer (Liquid-Guided Flow)
        # Liquid layer runs before MoE, giving tokens "time perception".
        # This way Router decides with tokens that know the past.
        if self.liquid is not None:
            # LiquidMixer contains Residual + Norm internally
            x = self.liquid(x)
        if self.workspace_layer is not None:
            x = self.workspace_layer(x, workspace)

        # 3. FeedForward / MoE
        h = self.norm2(x)
        aux_loss = None
        if self.is_moe_layer:
            ff_out, aux_loss = self.ff(h)  # Includes Z-Loss
        else:
            ff_out = self.ff(h)
            aux_loss = h.new_zeros(())

        # V23.0: Residual Scaling
        x = x + ff_out * self.residual_scale
        if self.hebbian_layer is not None:
            x = self.hebbian_layer(x)
        if self.neuro_symbolic_layer is not None:
            x = self.neuro_symbolic_layer(x)
        if self.lifelong_safety_layer is not None:
            x = self.lifelong_safety_layer(x)

        # 4. QINN optional
        if self.qinn is not None:
            x = self.qinn(x)

        # Aux loss must always be scalar
        if aux_loss is None:
            aux_loss = x.new_zeros(())
        if aux_loss.ndim > 0:
            aux_loss = aux_loss.sum()

        return x, aux_loss, present_key_value
