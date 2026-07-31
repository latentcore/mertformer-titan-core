"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright 2026 Mert Yünlü
Licensed under the Apache License, Version 2.0 (see LICENSE).

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

from config.build_label import BUILD_LABEL as __version__
__author__ = "Mert Yünlü"

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, List, Set, Tuple

from config.config import cfg
from layers.bitlinear import BitLinear, activation_quant, weight_quant


_MOE_PACK_ENABLED = os.environ.get("TITAN_MOE_PACK", "0") == "1"


def set_moe_pack_enabled(enabled: bool) -> None:
    """Runtime toggle for BitSwiGLU gate+up packed projection."""
    global _MOE_PACK_ENABLED
    _MOE_PACK_ENABLED = bool(enabled)


def _moe_packed_bitlinear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply BitLinear fallback math to a packed expert output weight matrix."""
    x_q = activation_quant(x)
    w_q = weight_quant(weight)
    return F.linear(x_q, w_q, None)


def moe_capacity(n_tokens: int, top_k: int, num_experts: int, capacity_factor: float) -> int:
    """Switch-style per-expert capacity: ceil(factor * total_assignments / experts), min 1.

    [2026-07-29] Extracted from ``MoE.forward`` so it can be imported and tested for
    real. Both ``tests/test_moe_capacity.py`` and ``tests/test_property_moe_capacity.py``
    previously re-implemented this formula as a local copy, and each said so explicitly
    ("Mirrors the inline formula in layers/moe.py; if that logic is extracted into a
    helper, import it here instead of mirroring"). That meant neither test actually
    exercised the shipped code -- a change to the real capacity path could break it while
    both tests stayed green. This is that extraction; the tests now import this function.

    Args:
        n_tokens: flattened token count (batch x seq).
        top_k: experts selected per token.
        num_experts: size of the routed expert pool.
        capacity_factor: slack multiplier over a perfectly balanced share (1.0 = exact
            share; the canonical config uses 1.25).
    Returns:
        Maximum assignments a single expert may accept, floored at 1.
    """
    return max(1, int(math.ceil(capacity_factor * (n_tokens * top_k) / max(1, num_experts))))


class BitSwiGLU(nn.Module):
    """
    BitNet SwiGLU block - Optimization for MoE experts.

    Features:
    - Memory savings via BitLinear projections
    - SwiGLU activation function
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        """
        BitSwiGLU initializer.

        Args:
            hidden_size (int): Hidden dimension
            intermediate_size (int): Intermediate dimension
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # BitLinear projections: gate, up, down
        self.gate_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = BitLinear(intermediate_size, hidden_size, bias=False)

    def _forward_packed(self, x: torch.Tensor) -> torch.Tensor:
        packed_weight = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
        gate_up = _moe_packed_bitlinear(x, packed_weight)
        gate, up = gate_up.chunk(2, dim=-1)
        x_inter = F.silu(gate) * up
        return self.down_proj(x_inter)

    def _forward_baseline(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(gate) * up -> down
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x_inter = F.silu(gate) * up
        return self.down_proj(x_inter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - Applies SwiGLU activation.

        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        if _MOE_PACK_ENABLED and os.environ.get("MERTFORMER_LOWBIT_KERNEL", "0") != "1":
            return self._forward_packed(x)
        return self._forward_baseline(x)




class LiquidRouter(nn.Module):
    """
    Liquid Router - Expert selector utilizing temporal context.

    Tech:
    - Main Path: looks at the current token (Standard Routing).
    - Fluid Path: looks at the 'momentum' of previous tokens (Context Momentum).
    - Mechanism: Causal Depthwise Conv1d and Rolling Buffer.

    NOTE (naming clarification): the "Liquid"/"Fluid Dynamics"/"Context Momentum"
    labels here are descriptive only. This module is NOT a continuous-time
    liquid/CfC cell. The actual mechanism is a causal depthwise Conv1d over a short
    history window plus a BitLinear gate (main_proj + fluid_gate). For the real
    liquid-time / CfC implementation see liquid.py (LiquidCell). The class name and
    param paths (main_proj/fluid_gate) are kept as-is because they are bound to the
    sealed checkpoint/state_dict contract.
    """
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        # Router Quantization: Use BitLinear for consistent 1.58-bit
        self.main_proj = BitLinear(hidden_size, num_experts, bias=False)
        
        # Fluid Dynamics: Lightweight history (History/Momentum)
        self.history_window = 4
        self.fluid_mixer = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=self.history_window,
            groups=hidden_size, # Depthwise
            padding=0, # Left padding is applied manually
            bias=False
        )
        # Router Quantization
        self.fluid_gate = BitLinear(hidden_size, num_experts, bias=False)
        nn.init.zeros_(self.fluid_gate.weight)
        
        # Stateful Inference Buffer
        # Holds the state of the last (history_window - 1) tokens during inference
        # Inference cache, excluded from checkpoints (runtime state)
        self.register_buffer(
            "inference_state",
            torch.zeros(1, hidden_size, self.history_window - 1),
            persistent=False,
        )

    def _update_inference_state(self, state: torch.Tensor) -> None:
        """
        Safely updates router state (dtype/device + shape alignment).
        """
        with torch.no_grad():
            target = state.detach()
            if self.inference_state.device != target.device:
                raise RuntimeError(
                    "inference_state device mismatch: "
                    f"state={self.inference_state.device}/{self.inference_state.dtype}, "
                    f"target={target.device}/{target.dtype}. "
                    "Move the whole model via model.to(...)."
                )
            if self.inference_state.dtype != target.dtype:
                self.inference_state = self.inference_state.to(dtype=target.dtype)
            self.inference_state.resize_(target.shape[0], target.shape[1], target.shape[2])
            self.inference_state.copy_(target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq, Hidden] or [Batch*Seq, Hidden] (must be made 3D if flattened)

        # Dimension Handling
        is_flat = False
        if x.dim() == 2:
            is_flat = True
            # If input arrived flattened: (Batch*Seq, Hidden) -> (Batch*Seq, 1, Hidden)
            # For safety:
            x = x.unsqueeze(1) # (N, 1, H)

        B, S, H = x.shape
        
        # 1. Instant Decision (Main Path)
        logits_main = self.main_proj(x)
        
        # 2. Fluid Decision (Fluid Path)
        if self.training or S > 1:
            # Training / Prefill mode (Parallel)
            x_t = x.transpose(1, 2) # [B, H, S]
            
            # True-causal conv: only left padding (no right-side leakage)
            x_t_padded = F.pad(x_t, (self.history_window - 1, 0))
            fluid_mem = self.fluid_mixer(x_t_padded)
            
            fluid_mem = fluid_mem.transpose(1, 2) # [B, S, H]
            logits_fluid = self.fluid_gate(F.silu(fluid_mem))
            
            # Update inference state with last tokens for next step
            if not self.training:
                last_tokens = x_t[..., -(self.history_window-1):]
                # Pad if sequence is too short
                if last_tokens.size(2) < (self.history_window - 1):
                    pad = torch.zeros(B, H, (self.history_window - 1) - last_tokens.size(2), device=x.device, dtype=x.dtype)
                    last_tokens = torch.cat([pad, last_tokens], dim=2)
                self._update_inference_state(last_tokens)
                
            out = logits_main + logits_fluid

        else:
            # Inference step (Single Token)
            # Batch-Safe Cache Handling
            # Check for batch size mismatch (e.g. beam search expansion)
            if self.inference_state.size(0) != B:
                if self.inference_state.size(0) == 1:
                     # Expand single state to match batch (Broadcasting)
                     expanded = self.inference_state.expand(B, -1, -1).contiguous()
                     self._update_inference_state(expanded)
                else:
                     # Reset state if mismatch is unresolvable (Safety Fallback)
                     reset_state = torch.zeros(
                         B,
                         H,
                         self.history_window - 1,
                         device=x.device,
                         dtype=x.dtype,
                     )
                     self._update_inference_state(reset_state)
            
            # Use buffer for history
            current_token = x.transpose(1, 2) # [B, H, 1]
            
            # Rolling buffer update: [History, Current]
            # State: [B, H, Window-1]
            # Context: [B, H, Window]
            context = torch.cat([self.inference_state, current_token], dim=2)
            
            # True-causal conv with explicit left padding (inference parity)
            context_padded = F.pad(context, (self.history_window - 1, 0))
            fluid_mem = self.fluid_mixer(context_padded) # [B, H, Window]
            
            # Take only the last step (current token output)
            fluid_mem = fluid_mem[..., -1:] # [B, H, 1]
            
            fluid_mem = fluid_mem.transpose(1, 2) # [B, 1, H]
            logits_fluid = self.fluid_gate(F.silu(fluid_mem)) # [B, 1, E]
            
            # Update state (Shift left)
            self._update_inference_state(context[..., 1:])
            
            out = logits_main + logits_fluid

        if is_flat:
             out = out.reshape(-1, self.num_experts)
             
        return out

    def get_state(self) -> torch.Tensor:
        """
        Returns the current memory (state) of the router.
        Returns:
            torch.Tensor: State buffer [Batch, Hidden, Window-1]
        """
        return self.inference_state.clone()

    def set_state(self, state: torch.Tensor) -> None:
        """
        Loads router memory from external source.
        Args:
            state (torch.Tensor): New state buffer
        """
        if state.dim() != 3:
            raise ValueError(f"State must be 3D [Batch, Hidden, Window-1], got {state.shape}")
        
        # Shape check (optional but recommended)
        expected_window = self.history_window - 1
        if state.size(2) != expected_window:
             raise ValueError(f"State window size mismatch. Expected {expected_window}, got {state.size(2)}")
        
        self._update_inference_state(state)


class MoE(nn.Module):
    """
    Sparse Mixture of Experts - Efficient expert selection via top-k routing.

    Features:
    - Optimized dispatch: only the selected experts are executed
    - Memory efficient: load balancing via bincount
    - Learnable shared expert: shared expert contribution is learnable
    - Dtype safe: runs cleanly under mixed precision training
    """

    def __init__(self) -> None:
        """MoE initializer."""
        super().__init__()

        self.hidden_size: int = int(cfg.hidden_size)
        self.num_experts: int = int(cfg.num_experts)
        # Use num_experts_per_tok (fallback for backwards compatibility)
        self.active_experts: int = int(
            getattr(cfg, "num_experts_per_tok", getattr(cfg, "active_experts", 1))
        )

        # Safety checks (explicit raises survive `python -O`, which strips asserts).
        # TR: num_experts en az 1 olmalı. / EN: num_experts must be >= 1.
        if self.num_experts < 1:
            raise ValueError("num_experts must be >= 1.")
        # TR: active_experts 1 <= k <= N olmalı. / EN: active_experts must satisfy 1 <= k <= num_experts.
        if not (1 <= self.active_experts <= self.num_experts):
            raise ValueError("active_experts must satisfy 1 <= k <= num_experts.")

        moe_intermediate = int(getattr(cfg, "moe_intermediate", self.hidden_size * 4))

        # Router: routes tokens to experts
        # [TITAN INNOVATION]: Liquid Router
        self.router = LiquidRouter(self.hidden_size, self.num_experts)

        # Experts: each expert is a BitSwiGLU block
        self.experts = nn.ModuleList(
            BitSwiGLU(self.hidden_size, moe_intermediate)
            for _ in range(self.num_experts)
        )

        # Shared Expert: common expert for all tokens (with learnable gate)
        self.shared_expert = BitSwiGLU(self.hidden_size, moe_intermediate)
        shared_gate_init = float(getattr(cfg, "shared_expert_gate", 0.0))
        # Must be an nn.Parameter - the model optimizes it
        self.shared_gate = nn.Parameter(
            torch.tensor([shared_gate_init], dtype=torch.float32)
        )

        # Router hyperparameters
        self.router_temperature: float = float(
            getattr(cfg, "router_temperature", 1.0)
        )
        self.router_jitter: float = float(getattr(cfg, "router_jitter", 0.01))
        self.router_z_loss_coef: float = float(getattr(cfg, "z_loss_coef", 0.0))
        self.router_alarm_threshold: float = float(getattr(cfg, "router_alarm_threshold", 0.40))
        self.use_cross_expert_sync_bus: bool = bool(getattr(cfg, "use_cross_expert_sync_bus", False))
        self.cross_expert_sync_gain: float = float(getattr(cfg, "cross_expert_sync_gain", 0.05))
        self.use_structural_plasticity: bool = bool(getattr(cfg, "use_structural_plasticity", False))
        self.structural_ema_decay: float = float(getattr(cfg, "structural_ema_decay", 0.98))
        self.structural_prune_threshold: float = float(getattr(cfg, "structural_prune_threshold", 0.02))
        self.structural_grow_threshold: float = float(getattr(cfg, "structural_grow_threshold", 0.60))
        self.structural_update_interval: int = int(getattr(cfg, "structural_update_interval", 100))
        # Inference-first expert paging (on-demand expert residency).
        self.use_expert_paging: bool = bool(getattr(cfg, "use_expert_paging", False))
        self.expert_paging_inference_only: bool = bool(
            getattr(cfg, "expert_paging_inference_only", True)
        )
        self.expert_paging_lazy_init: bool = bool(
            getattr(cfg, "expert_paging_lazy_init", True)
        )
        self.expert_paging_cache_size: int = max(
            1, int(getattr(cfg, "expert_paging_cache_size", self.active_experts))
        )
        self.expert_paging_offload_device: str = str(
            getattr(cfg, "expert_paging_offload_device", "cpu")
        )
        self.expert_paging_verbose: bool = bool(getattr(cfg, "expert_paging_verbose", False))
        self._expert_lru: List[int] = []
        self._expert_resident: Set[int] = set()
        self._paging_bootstrapped: bool = False
        
        # Switch Loss and Collapse Recovery
        self.use_switch_loss: bool = bool(getattr(cfg, "use_switch_loss", False))
        self.router_jitter_boost: float = float(getattr(cfg, "router_jitter_boost", 0.1))
        self.collapse_threshold: float = 0.85  # Max load threshold for collapse detection
        self.moe_capacity_enforce: bool = bool(getattr(cfg, "moe_capacity_enforce", True))
        self.moe_capacity_factor: float = float(getattr(cfg, "moe_capacity_factor", 1.25))
        self.dispatch_mode: str = str(getattr(cfg, "moe_dispatch_mode", "sequential")).lower()
        
        # Telemetry & Collapse State.
        # persistent=False on all of these: they are runtime telemetry / derived
        # recovery state, NOT learned parameters. Writing them into a checkpoint
        # would leak a stale `collapse_detected` flag or a structural-plasticity
        # `expert_activity_mask` across a resume boundary (e.g. resume with jitter
        # already latched on). They are recomputed every forward, so dropping them
        # from the checkpoint is both correct and cleaner. Mirrors the paging
        # counters below, which were already persistent=False.
        self.register_buffer("last_expert_load", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("last_router_entropy", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_router_max_load", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_capacity_overflow_ratio", torch.tensor(0.0), persistent=False)
        self.register_buffer("collapse_detected", torch.tensor(False), persistent=False)
        self.register_buffer("expert_activity_mask", torch.ones(self.num_experts, dtype=torch.bool), persistent=False)
        self.register_buffer("expert_usage_ema", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("plasticity_step", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer(
            "expert_paging_swaps_in",
            torch.zeros((), dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "expert_paging_swaps_out",
            torch.zeros((), dtype=torch.int64),
            persistent=False,
        )
        # Cross-expert sync bus projections are only materialized when the
        # feature is enabled. The forward pass already guards their use behind
        # the same flag, so on the canonical flag-off path these modules are
        # never read; instantiating them unconditionally just carried ~25M idle
        # parameters per build. Guarding here keeps the measured runtime total
        # honest (idle params excluded) without changing flag-off behavior.
        if self.use_cross_expert_sync_bus:
            self.sync_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.sync_load_proj = nn.Linear(self.num_experts, self.hidden_size, bias=False)
            self.sync_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _should_skip_expert_apply(self) -> bool:
        """
        Skip expert migration during parent model.to(...) only when paging is enabled.
        This is the key to avoiding first-load expert VRAM residency.
        """
        if not self.use_expert_paging:
            return False
        if not self.expert_paging_lazy_init:
            return False
        return True

    def _offload_all_experts(self, target_device: torch.device, target_dtype: torch.dtype | None = None) -> None:
        with torch.no_grad():
            for expert in self.experts:
                if target_dtype is None:
                    expert.to(device=target_device)
                else:
                    expert.to(device=target_device, dtype=target_dtype)
        self._expert_resident.clear()
        self._expert_lru.clear()
        self._paging_bootstrapped = True

    def _apply(self, fn):
        """
        Preserve expert offload residency across parent model.to(...).
        We let MoE non-expert tensors follow fn, then keep experts on offload device.
        """
        if not self._should_skip_expert_apply():
            return super()._apply(fn)

        experts = self._modules.pop("experts")
        try:
            out = super()._apply(fn)
        finally:
            self._modules["experts"] = experts

        offload_device = self._offload_device()
        ref_param = next(self.shared_expert.parameters(), None)
        target_dtype = ref_param.dtype if ref_param is not None and torch.is_floating_point(ref_param) else None
        self._offload_all_experts(offload_device, target_dtype=target_dtype)
        return out

    def train(self, mode: bool = True):
        """
        Keep behavior safe across train/eval toggles when expert paging is enabled.
        - train(): all experts on compute device (no paging path).
        - eval(): optionally offload all experts for lazy residency.
        """
        out = super().train(mode)
        if not self.use_expert_paging:
            return out

        ref_param = next(self.shared_expert.parameters(), None)
        if ref_param is None:
            return out
        ref_device = ref_param.device
        ref_dtype = ref_param.dtype if torch.is_floating_point(ref_param) else None

        if mode:
            with torch.no_grad():
                for expert in self.experts:
                    if ref_dtype is None:
                        expert.to(device=ref_device)
                    else:
                        expert.to(device=ref_device, dtype=ref_dtype)
            self._expert_resident = set(range(self.num_experts))
            self._expert_lru = list(range(self.num_experts))
            self._paging_bootstrapped = False
            return out

        if self.expert_paging_lazy_init:
            offload_device = self._offload_device()
            if offload_device != ref_device:
                self._offload_all_experts(offload_device, target_dtype=ref_dtype)
        return out

    def get_router_state(self) -> torch.Tensor:
        """External API: Get Liquid Router state."""
        return self.router.get_state()
        
    def set_router_state(self, state: torch.Tensor) -> None:
        """External API: Set Liquid Router state."""
        self.router.set_state(state)
        
    def get_expert_load(self) -> torch.Tensor:
        """External API: Get last expert load distribution."""
        return self.last_expert_load
    
    def get_router_entropy(self) -> torch.Tensor:
        """External API: Normalized router load entropy [0,1]."""
        return self.last_router_entropy

    def get_router_max_load(self) -> torch.Tensor:
        """External API: Maximum expert load."""
        return self.last_router_max_load

    def get_expert_paging_stats(self) -> Dict[str, Any]:
        """External API: Returns runtime paging counters and mode."""
        return {
            "enabled": bool(self.use_expert_paging),
            "inference_only": bool(self.expert_paging_inference_only),
            "lazy_init": bool(self.expert_paging_lazy_init),
            "cache_size": int(self.expert_paging_cache_size),
            "offload_device": str(self.expert_paging_offload_device),
            "bootstrapped": bool(self._paging_bootstrapped),
            "swaps_in": int(self.expert_paging_swaps_in.item()),
            "swaps_out": int(self.expert_paging_swaps_out.item()),
            "resident_count": int(len(self._expert_resident)),
        }

    def _paging_active_for_step(self) -> bool:
        """
        Inference-first paging gate.
        Training path intentionally remains unchanged for gradient safety.
        """
        if not self.use_expert_paging:
            return False
        if self.training and self.expert_paging_inference_only:
            return False
        if self.training:
            return False
        return True

    def _expert_device(self, expert_id: int) -> torch.device:
        p = next(self.experts[expert_id].parameters(), None)
        return p.device if p is not None else torch.device("cpu")

    def _offload_device(self) -> torch.device:
        try:
            return torch.device(self.expert_paging_offload_device)
        except Exception:
            return torch.device("cpu")

    def _touch_lru(self, expert_id: int) -> None:
        if expert_id in self._expert_lru:
            self._expert_lru.remove(expert_id)
        self._expert_lru.append(expert_id)

    def _refresh_resident(self, compute_device: torch.device) -> None:
        self._expert_resident = {
            idx
            for idx in range(self.num_experts)
            if self._expert_device(idx) == compute_device
        }
        self._expert_lru = [idx for idx in self._expert_lru if idx in self._expert_resident]

    def _bootstrap_expert_paging(self, compute_device: torch.device) -> None:
        if self._paging_bootstrapped:
            return
        if compute_device.type == "cpu":
            self._paging_bootstrapped = True
            return
        offload_device = self._offload_device()
        if offload_device == compute_device:
            self._paging_bootstrapped = True
            return
        ref_param = next(self.shared_expert.parameters(), None)
        target_dtype = ref_param.dtype if ref_param is not None and torch.is_floating_point(ref_param) else None
        self._offload_all_experts(offload_device, target_dtype=target_dtype)
        if self.expert_paging_verbose:
            print(
                f"[moe:paging] bootstrapped offload={offload_device} "
                f"cache_size={self.expert_paging_cache_size}"
            )

    def _page_in_active_experts(
        self,
        active_expert_ids: List[int],
        compute_device: torch.device,
        compute_dtype: torch.dtype,
    ) -> None:
        if not active_expert_ids:
            return
        self._bootstrap_expert_paging(compute_device)
        if compute_device.type == "cpu":
            return
        offload_device = self._offload_device()
        if offload_device == compute_device:
            return

        self._refresh_resident(compute_device)
        with torch.no_grad():
            for expert_id in active_expert_ids:
                if expert_id not in self._expert_resident:
                    self.experts[expert_id].to(device=compute_device, dtype=compute_dtype)
                    self._expert_resident.add(expert_id)
                    self.expert_paging_swaps_in.add_(1)
                self._touch_lru(expert_id)

            keep = set(active_expert_ids)
            max_resident = max(self.expert_paging_cache_size, len(keep))
            while len(self._expert_resident) > max_resident:
                evict_id = None
                for candidate in self._expert_lru:
                    if candidate not in keep:
                        evict_id = candidate
                        break
                if evict_id is None:
                    break
                self.experts[evict_id].to(device=offload_device, dtype=compute_dtype)
                self._expert_resident.discard(evict_id)
                self.expert_paging_swaps_out.add_(1)
                self._expert_lru = [idx for idx in self._expert_lru if idx != evict_id]

    def _apply_structural_plasticity(self, load: torch.Tensor) -> None:
        """
        Structural plasticity v0:
        - Prune low-usage experts by deactivating mask
        - Grow by re-activating one inactive expert on heavy collapse pressure
        """
        if not self.use_structural_plasticity:
            return
        if not self.training:
            return

        with torch.no_grad():
            self.expert_usage_ema.mul_(self.structural_ema_decay).add_(
                load.detach() * (1.0 - self.structural_ema_decay)
            )
            self.plasticity_step.add_(1)
            if int(self.plasticity_step.item()) % max(1, self.structural_update_interval) != 0:
                return

            active_count = int(self.expert_activity_mask.sum().item())
            min_active = max(1, self.active_experts)

            # Prune one weakest active expert if safely above minimum.
            if active_count > min_active:
                active_idx = torch.where(self.expert_activity_mask)[0]
                active_ema = self.expert_usage_ema[active_idx]
                prune_pos = torch.argmin(active_ema)
                prune_idx = active_idx[prune_pos]
                if active_ema[prune_pos].item() < self.structural_prune_threshold:
                    self.expert_activity_mask[prune_idx] = False
                    active_count -= 1

            # Grow by re-enabling one inactive expert under heavy pressure.
            inactive_idx = torch.where(~self.expert_activity_mask)[0]
            if inactive_idx.numel() > 0 and self.last_router_max_load.item() > self.structural_grow_threshold:
                grow_idx = inactive_idx[0]
                self.expert_activity_mask[grow_idx] = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - MoE routing and expert dispatch.

        Args:
            x (torch.Tensor): Input tensor [Batch, Seq, Hidden]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Output, Aux loss)
        """
        B, T, H = x.shape
        
        # -----------------------------
        # 1) Router & Gating (LIQUID FIX)
        # -----------------------------
        # Pass 3D tensor to LiquidRouter to preserve temporal context
        logits = self.router(x) # (B, T, E)
        
        # Flatten after routing for dispatch logic
        logits = logits.reshape(-1, self.num_experts) # (N, E)
        x_flat = x.reshape(-1, H)
        N = x_flat.size(0)
        E = self.num_experts

        # Computations are more stable in FP32
        logits_f = logits.float()
        active_mask = self.expert_activity_mask.to(device=logits_f.device)
        if active_mask.any():
            logits_f = logits_f.masked_fill(~active_mask.unsqueeze(0), float("-inf"))

        # Temperature scaling
        if self.router_temperature != 1.0:
            logits_f = logits_f / self.router_temperature

        # Jitter: add noise only during training (for exploration)
        applied_jitter = self.router_jitter
        
        # [DESIGN] Collapse Recovery logic moved here to avoid permanent generic mutation
        if self.training and self.collapse_detected.item():
             applied_jitter = self.router_jitter_boost
        
        if self.training and applied_jitter > 0.0:
            logits_f = logits_f + torch.randn_like(logits_f) * applied_jitter

        # Top-K First, then Softmax (NPU Optimization)
        # Old method: Softmax(all) -> TopK
        # New method: TopK -> Softmax(selected)
        # This method saves FLOPs and yields sharper decisions.
        k = min(self.active_experts, E)
        topk_logits, topk_idx = torch.topk(logits_f, k=k, dim=-1) # (N, k)
        
        topk_vals = F.softmax(topk_logits, dim=-1) # (N, k) - Softmax over only the K elements
        capacity_mask = torch.ones_like(topk_idx, dtype=torch.bool)
        overflow_ratio = torch.tensor(0.0, device=x.device, dtype=torch.float32)

        # Switch-style capacity control: cap per-expert assignments and renormalize gates.
        #
        # [2026-07-29] Fully vectorized. The previous implementation looped over experts
        # and called `(topk_idx == e).nonzero()` for each -- and `torch.nonzero` needs a
        # device->host sync to size its output. With E=8 experts x 6 MoE layers, plus the
        # `empty_rows.any()` and `int(overflow.size(0))` syncs, that was ~50+ pipeline
        # stalls per micro-batch on top of an O(E) full-tensor scan of N*k elements
        # (N = micro_batch x 4096 at the canonical config). This version is sync-free.
        #
        # SEMANTICS ARE PRESERVED EXACTLY, and that is the delicate part: `nonzero()`
        # returns hits in ROW-MAJOR order, so the old code kept each expert's first
        # `capacity` assignments ordered by (row, then column). `topk_idx.reshape(-1)`
        # produces flat index `row * k + col`, i.e. that same row-major order, and a
        # STABLE argsort groups by expert while preserving it inside each group. The
        # rank-within-group is therefore identical to the old slice boundary.
        # scripts/cfc_moe_tolerance_check.py is the gate on this (max_diff must stay 0).
        if self.moe_capacity_enforce and self.moe_capacity_factor > 0.0:
            capacity = moe_capacity(N, k, E, self.moe_capacity_factor)

            flat_e = topk_idx.reshape(-1)                                   # (N*k,) row-major
            order = torch.argsort(flat_e, stable=True)                      # expert-grouped
            slot_counts = torch.zeros(E, device=flat_e.device, dtype=torch.long)
            slot_counts.scatter_add_(0, flat_e, torch.ones_like(flat_e))
            group_starts = torch.cumsum(slot_counts, 0) - slot_counts
            rank_in_sorted = torch.empty_like(order)
            rank_in_sorted[order] = torch.arange(order.numel(), device=order.device)
            within_group = rank_in_sorted - group_starts[flat_e]
            capacity_mask = (within_group < capacity).reshape(topk_idx.shape)

            # Count drops BEFORE the empty-row restoration, matching the old ordering
            # (the old `dropped` was accumulated before the restoration ran).
            dropped = (~capacity_mask).sum()

            topk_vals = topk_vals * capacity_mask.float()
            # Any token whose every choice was dropped falls back to its top-1 expert.
            # Done with torch.where instead of `if empty_rows.any(): ...` so no sync.
            empty_rows = topk_vals.sum(dim=-1) <= 0
            first_slot = torch.zeros_like(capacity_mask)
            first_slot[:, 0] = True
            force_first = empty_rows.unsqueeze(-1) & first_slot
            topk_vals = torch.where(force_first, torch.ones_like(topk_vals), topk_vals)
            capacity_mask = capacity_mask | force_first

            topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            overflow_ratio = dropped.to(torch.float32) / float(max(1, N * k))

        # [TELEMETRY] Load Calculation (Always compute for monitoring)
        # MPS Safe Bincount: Use scatter_add_ instead of bincount for compatibility.
        # [2026-07-29] Counts are now accumulated with the capacity mask as the scatter
        # WEIGHT rather than via `topk_idx[capacity_mask]`. Boolean-mask indexing has a
        # data-dependent output size and therefore also forces a device->host sync every
        # forward; weighting the scatter is mathematically identical (dropped slots
        # contribute 0) and stays on-device.
        mask_flat = capacity_mask.reshape(-1).to(torch.float32)
        counts = torch.zeros(E, device=topk_idx.device, dtype=torch.float32)
        counts.scatter_add_(0, topk_idx.reshape(-1), mask_flat)

        # Kept-assignment count as a tensor (was `float(int(flat_idx.numel()))`, a sync).
        denom = mask_flat.sum().clamp(min=1.0)
        load = counts / denom
        self.last_expert_load.copy_(load.detach()) # Store for logging
        self.last_router_max_load.copy_(load.max().detach())
        norm = math.log(float(E)) if E > 1 else 1.0
        entropy = -(load.clamp(min=1e-8) * load.clamp(min=1e-8).log()).sum() / norm
        self.last_router_entropy.copy_(entropy.detach())
        self.last_capacity_overflow_ratio.copy_(overflow_ratio.detach())
        self._apply_structural_plasticity(load)
        if self._paging_active_for_step():
            active_expert_ids = (
                torch.nonzero(counts > 0.0, as_tuple=False).flatten().tolist()
            )
            self._page_in_active_experts(
                [int(idx) for idx in active_expert_ids],
                x.device,
                x.dtype,
            )

        # -----------------------------
        # 2) Aux Loss (Load Balancing) - Switch/L2 Option
        # -----------------------------
        # Importance: cumulative probability of the gates (must be computed over all experts)
        if self.training:
             gates_full = F.softmax(logits_f, dim=-1) # (N, E)
             importance = gates_full.mean(dim=0) # (E,)
             
             if self.use_switch_loss:
                 # Switch Transformer formula: more aggressive, faster learning
                 load_balancing_loss = (importance * load).sum() * float(E)
             else:
                 # L2 Loss: more stable, recommended for the Liquid Router
                 load_balancing_loss = ((importance - load) ** 2).mean() * float(E)
             
             # Router Collapse Detection & Recovery
             max_load = load.max().item()
             if max_load > self.collapse_threshold:
                 self.collapse_detected.fill_(True)
                 # Jitter boost is handled via 'applied_jitter' during forward pass
             elif self.collapse_detected.item() and max_load < 0.5:
                 # Recovery: Reset state
                 self.collapse_detected.fill_(False)
             # [18] The per-rank collapse flag is intentionally NOT all_reduced here.
             # A collective inside MoE.forward re-fires during gradient-checkpointing
             # reentrant recompute and can interleave with DDP's own gradient-bucket
             # all_reduces (collective-ordering / NCCL-deadlock hazard on multi-GPU),
             # and it forces a per-step host-device sync. The collapse flag is a local
             # recovery heuristic (drives jitter on this rank); cross-rank consensus is
             # not required, so we keep it rank-local and collective-free.
        else:
             load_balancing_loss = torch.tensor(0.0, device=x.device, dtype=logits_f.dtype)

        aux_loss = load_balancing_loss

        # Z-Loss: prevents logits from growing excessively
        if self.router_z_loss_coef > 0.0:
            z = torch.logsumexp(logits_f, dim=-1)  # (N,)
            z_loss = (z * z).mean() * self.router_z_loss_coef
            aux_loss = aux_loss + z_loss

        # -----------------------------
        # 3) Dispatch (Distribution to Experts)
        # -----------------------------
        if self.dispatch_mode == "parallel":
            out_flat = self._dispatch_parallel(x_flat, topk_idx, topk_vals, capacity_mask)
        else:
            out_flat = self._dispatch_sequential(x_flat, topk_idx, topk_vals)

        # -----------------------------
        # 4) Shared Expert (Dtype Safe & Sigmoid Gate)
        # -----------------------------
        shared_out = self.shared_expert(x_flat)  # (N, H)

        # Sigmoid Gate: force the gate into the 0-1 range.
        # Prevents negative learning and provides stability.
        gate_val = torch.sigmoid(self.shared_gate) # (1,)
        
        gate_scale = gate_val.to(
            dtype=shared_out.dtype, device=shared_out.device
        )

        out_flat = out_flat + shared_out * gate_scale

        # Cross-expert synchronization bus (attention-independent global coordination).
        if self.use_cross_expert_sync_bus:
            token_sync = self.sync_proj(out_flat.mean(dim=0, keepdim=True))
            load_sync = self.sync_load_proj(load.unsqueeze(0).to(dtype=out_flat.dtype))
            sync = torch.tanh(token_sync + load_sync).expand_as(out_flat)
            out_flat = out_flat + sync * (torch.sigmoid(self.sync_gate) * self.cross_expert_sync_gain)

        # Restore the shape
        out = out_flat.reshape(B, T, H)

        # Keep aux_loss as float32 for precision stability
        return out, aux_loss

    def _dispatch_sequential(
        self, x_flat: torch.Tensor, topk_idx: torch.Tensor, topk_vals: torch.Tensor
    ) -> torch.Tensor:
        N, H = x_flat.shape
        out_flat = x_flat.new_zeros((N, H))
        for expert_id_int, expert in enumerate(self.experts):
            expert_mask = topk_idx == expert_id_int
            token_mask = expert_mask.any(dim=-1)
            if not token_mask.any():
                continue
            selected_x = x_flat[token_mask]
            expert_param = next(expert.parameters(), None)
            if expert_param is not None and selected_x.dtype != expert_param.dtype:
                selected_x = selected_x.to(dtype=expert_param.dtype)
            expert_out = expert(selected_x)
            if expert_out.dtype != out_flat.dtype:
                expert_out = expert_out.to(dtype=out_flat.dtype)
            weights = (topk_vals[token_mask] * expert_mask[token_mask].float()).sum(dim=-1, keepdim=True)
            out_flat[token_mask] += expert_out * weights
        return out_flat

    def _dispatch_parallel(
        self,
        x_flat: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_vals: torch.Tensor,
        capacity_mask: torch.Tensor,
    ) -> torch.Tensor:
        N, H = x_flat.shape
        k = topk_idx.size(-1)
        out_flat = x_flat.new_zeros((N, H))

        token_idx = torch.arange(N, device=topk_idx.device).repeat_interleave(k)
        expert_idx = topk_idx.reshape(-1)
        weights = topk_vals.reshape(-1)
        mask = capacity_mask.reshape(-1)
        if mask.numel() > 0:
            token_idx = token_idx[mask]
            expert_idx = expert_idx[mask]
            weights = weights[mask]

        if expert_idx.numel() == 0:
            return out_flat

        order = torch.argsort(expert_idx)
        expert_sorted = expert_idx[order]
        token_sorted = token_idx[order]
        weight_sorted = weights[order]

        # MPS Safe Bincount: scatter_add_ instead of bincount, matching the same
        # portability fix already applied to the telemetry counts above (line ~730).
        # torch.bincount is not guaranteed on MPS across all torch>=2.0 (requirements.txt
        # sets no upper bound); this keeps both count paths in this file consistent.
        counts = torch.zeros(self.num_experts, device=expert_sorted.device, dtype=torch.long)
        counts.scatter_add_(0, expert_sorted, torch.ones_like(expert_sorted))
        if counts.numel() == 0:
            return out_flat

        # [2026-07-29] One device->host transfer instead of E of them. This loop needs the
        # per-expert segment boundaries as Python ints to slice with, and it used to read
        # them one at a time via `int(counts[e].item())` -- E separate syncs per MoE layer
        # per micro-batch (x6 MoE layers x grad_accum). `.tolist()` moves the whole E-length
        # boundary vector in a single transfer. Identical segmentation, identical order.
        segment_ends = torch.cumsum(counts, 0).tolist()

        start = 0
        for expert_id_int, expert in enumerate(self.experts):
            end = segment_ends[expert_id_int]
            if end == start:
                continue
            idx = token_sorted[start:end]
            w = weight_sorted[start:end].unsqueeze(-1)
            selected_x = x_flat.index_select(0, idx)
            expert_param = next(expert.parameters(), None)
            if expert_param is not None and selected_x.dtype != expert_param.dtype:
                selected_x = selected_x.to(dtype=expert_param.dtype)
            expert_out = expert(selected_x)
            if expert_out.dtype != out_flat.dtype:
                expert_out = expert_out.to(dtype=out_flat.dtype)
            # [2026-07-29] `index_add_` REQUIRES source dtype == self dtype (unlike
            # `__setitem__`, which casts). `expert_out` is cast to out_flat.dtype just
            # above, but `w` comes from topk_vals, which is fp32 (the router deliberately
            # computes in fp32), so `expert_out * w` promoted BACK to fp32 and the call
            # only survived because the residual stream happens to stay fp32 under
            # autocast (nn.Embedding is not autocast-listed, and `fp32 + bf16` promotes to
            # fp32). That is an accident, not an invariant: anything that makes the
            # residual bf16 would turn this into a hard RuntimeError mid-training.
            out_flat.index_add_(0, idx, expert_out * w.to(out_flat.dtype))
            start = end
        return out_flat
