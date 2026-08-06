"""Sparse mixture-of-experts with the Liquid router.

Mirrors ``vendor/upstream/layers/moe.py`` (``moe_capacity``, ``BitSwiGLU``,
``LiquidRouter``, ``MoE``).

DRIFT FIX #2 -- capacity enforcement
------------------------------------
Canonical ``moe.py:742-781`` is fully vectorized and sync-free:

* per-expert rank computed with a *stable* argsort so the kept slots match
  ``nonzero()``'s row-major order exactly;
* ``force_first`` restores the top-1 expert for any token whose every choice was
  dropped, via ``torch.where`` (no ``.any()`` sync);
* load counts accumulate the capacity mask as a scatter **weight**, because
  boolean-mask indexing has a data-dependent output size and forces a
  device->host sync every forward.

The canonical file carries a dated comment saying the loop version was a bug it
had already fixed. ``scripts/chess_5080_onefile.py`` still ran the old shape:
a Python ``for`` loop over experts calling ``(topk_idx == e).nonzero()`` plus
``topk_idx[capacity_mask]`` counting -- i.e. it reintroduced both the wrong
counting and O(E) pipeline stalls per MoE layer per step. This module follows
the canonical vectorized version.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .bitlinear import make_linear


def moe_capacity(n_tokens: int, top_k: int, num_experts: int, capacity_factor: float) -> int:
    """Switch-style per-expert capacity: ceil(factor * total_assignments / experts), min 1."""
    return max(1, int(math.ceil(capacity_factor * (n_tokens * top_k) / max(1, num_experts))))


class BitSwiGLU(nn.Module):
    """SwiGLU expert body."""

    def __init__(self, hidden_size: int, intermediate_size: int, use_bitnet: bool = False) -> None:
        super().__init__()
        self.gate_proj = make_linear(use_bitnet, hidden_size, intermediate_size)
        self.up_proj = make_linear(use_bitnet, hidden_size, intermediate_size)
        self.down_proj = make_linear(use_bitnet, intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LiquidRouter(nn.Module):
    """Expert selector: instantaneous projection + short causal history term.

    Naming note carried over from the canonical file: "Liquid"/"Fluid" here is
    descriptive only. The mechanism is a causal depthwise ``Conv1d`` over a
    4-token window plus a gate -- it is *not* a CfC cell (that is
    ``liquid.LiquidCell``). Names are kept because they are bound to the
    checkpoint/state_dict contract.
    """

    def __init__(self, hidden_size: int, num_experts: int, use_bitnet: bool = False) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.history_window = 4

        self.main_proj = make_linear(use_bitnet, hidden_size, num_experts)
        self.fluid_mixer = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=self.history_window,
            groups=hidden_size,  # depthwise
            padding=0,           # left padding applied manually
            bias=False,
        )
        self.fluid_gate = make_linear(use_bitnet, hidden_size, num_experts)
        nn.init.zeros_(self.fluid_gate.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_flat = x.dim() == 2
        if is_flat:
            x = x.unsqueeze(1)
        B, S, H = x.shape

        logits_main = self.main_proj(x)

        x_t = x.transpose(1, 2)                                   # [B, H, S]
        x_t_padded = F.pad(x_t, (self.history_window - 1, 0))     # true-causal: left pad only
        fluid_mem = self.fluid_mixer(x_t_padded).transpose(1, 2)  # [B, S, H]
        logits_fluid = self.fluid_gate(F.silu(fluid_mem))

        out = logits_main + logits_fluid
        if is_flat:
            out = out.reshape(-1, self.num_experts)
        return out


class MoE(nn.Module):
    """Top-k MoE with Switch-style capacity, shared expert and z-loss."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = int(cfg.hidden_size)
        self.num_experts = int(cfg.num_experts)
        self.active_experts = int(cfg.num_experts_per_tok)
        inter = cfg.resolved_moe_intermediate()
        use_bn = bool(cfg.use_bitnet)

        self.router = LiquidRouter(self.hidden_size, self.num_experts, use_bitnet=use_bn)
        self.experts = nn.ModuleList(
            BitSwiGLU(self.hidden_size, inter, use_bitnet=use_bn) for _ in range(self.num_experts)
        )
        self.shared_expert = BitSwiGLU(self.hidden_size, inter, use_bitnet=use_bn)
        self.shared_gate = nn.Parameter(torch.tensor([float(cfg.shared_expert_gate)]))

        self.router_temperature = float(cfg.router_temperature)
        self.router_jitter = float(cfg.router_jitter)
        self.router_jitter_boost = float(cfg.router_jitter_boost)
        self.router_z_loss_coef = float(cfg.z_loss_coef)
        self.use_switch_loss = bool(cfg.use_switch_loss)
        self.moe_capacity_enforce = bool(cfg.moe_capacity_enforce)
        self.moe_capacity_factor = float(cfg.moe_capacity_factor)
        self.dispatch_mode = str(cfg.moe_dispatch_mode)
        self.collapse_threshold = 0.85

        self.register_buffer("last_expert_load", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("last_router_entropy", torch.zeros(()), persistent=False)
        self.register_buffer("last_router_max_load", torch.zeros(()), persistent=False)
        self.register_buffer("last_capacity_overflow_ratio", torch.zeros(()), persistent=False)
        self.register_buffer("collapse_detected", torch.zeros((), dtype=torch.bool), persistent=False)

    # -- dispatch -----------------------------------------------------------
    def _dispatch_parallel(
        self,
        x_flat: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_vals: torch.Tensor,
        capacity_mask: torch.Tensor,
    ) -> torch.Tensor:
        N, H = x_flat.shape
        k = topk_idx.size(-1)
        out_flat = torch.zeros_like(x_flat)

        token_idx = torch.arange(N, device=topk_idx.device).repeat_interleave(k)
        expert_idx = topk_idx.reshape(-1)
        weights = topk_vals.reshape(-1) * capacity_mask.reshape(-1).to(topk_vals.dtype)

        order = torch.argsort(expert_idx, stable=True)
        expert_sorted = expert_idx[order]
        token_sorted = token_idx[order]
        weight_sorted = weights[order]

        counts = torch.zeros(self.num_experts, device=expert_idx.device, dtype=torch.long)
        counts.scatter_add_(0, expert_sorted, torch.ones_like(expert_sorted))
        # One host sync per MoE layer per step is unavoidable here: expert group
        # boundaries drive Python-level slicing. Everything upstream of this is
        # sync-free.
        counts_list = counts.tolist()

        start = 0
        for expert_id, cnt in enumerate(counts_list):
            if cnt == 0:
                continue
            end = start + cnt
            idx = token_sorted[start:end]
            w = weight_sorted[start:end].unsqueeze(-1)
            expert_out = self.experts[expert_id](x_flat.index_select(0, idx))
            out_flat.index_add_(0, idx, (expert_out * w).to(out_flat.dtype))
            start = end
        return out_flat

    def _dispatch_sequential(
        self,
        x_flat: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_vals: torch.Tensor,
        capacity_mask: torch.Tensor,
    ) -> torch.Tensor:
        out_flat = torch.zeros_like(x_flat)
        masked_vals = topk_vals * capacity_mask.to(topk_vals.dtype)
        for expert_id in range(self.num_experts):
            sel = topk_idx == expert_id
            if not bool(sel.any()):
                continue
            token_mask = sel.any(dim=-1)
            selected = x_flat[token_mask]
            expert_out = self.experts[expert_id](selected)
            w = (masked_vals[token_mask] * sel[token_mask].to(masked_vals.dtype)).sum(dim=-1, keepdim=True)
            out_flat[token_mask] += (expert_out * w).to(out_flat.dtype)
        return out_flat

    # -- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H = x.shape
        logits = self.router(x).reshape(-1, self.num_experts)
        x_flat = x.reshape(-1, H)
        N = x_flat.size(0)
        E = self.num_experts

        logits_f = logits.float()
        if self.router_temperature != 1.0:
            logits_f = logits_f / self.router_temperature

        applied_jitter = self.router_jitter
        if self.training and bool(self.collapse_detected):
            applied_jitter = self.router_jitter_boost
        if self.training and applied_jitter > 0.0:
            logits_f = logits_f + torch.randn_like(logits_f) * applied_jitter

        k = min(self.active_experts, E)
        topk_logits, topk_idx = torch.topk(logits_f, k=k, dim=-1)
        topk_vals = F.softmax(topk_logits, dim=-1)
        capacity_mask = torch.ones_like(topk_idx, dtype=torch.bool)
        overflow_ratio = torch.zeros((), device=x.device, dtype=torch.float32)

        if self.moe_capacity_enforce and self.moe_capacity_factor > 0.0:
            capacity = moe_capacity(N, k, E, self.moe_capacity_factor)

            flat_e = topk_idx.reshape(-1)                       # row-major (row*k + col)
            order = torch.argsort(flat_e, stable=True)          # groups by expert, keeps row order
            slot_counts = torch.zeros(E, device=flat_e.device, dtype=torch.long)
            slot_counts.scatter_add_(0, flat_e, torch.ones_like(flat_e))
            group_starts = torch.cumsum(slot_counts, 0) - slot_counts
            rank_in_sorted = torch.empty_like(order)
            rank_in_sorted[order] = torch.arange(order.numel(), device=order.device)
            within_group = rank_in_sorted - group_starts[flat_e]
            capacity_mask = (within_group < capacity).reshape(topk_idx.shape)

            # Count drops BEFORE empty-row restoration (canonical ordering).
            dropped = (~capacity_mask).sum()

            topk_vals = topk_vals * capacity_mask.to(topk_vals.dtype)
            empty_rows = topk_vals.sum(dim=-1) <= 0
            first_slot = torch.zeros_like(capacity_mask)
            first_slot[:, 0] = True
            force_first = empty_rows.unsqueeze(-1) & first_slot
            topk_vals = torch.where(force_first, torch.ones_like(topk_vals), topk_vals)
            capacity_mask = capacity_mask | force_first

            topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            overflow_ratio = dropped.to(torch.float32) / float(max(1, N * k))

        # Load telemetry: scatter with the mask as WEIGHT (no data-dependent
        # indexing => no device->host sync).
        mask_flat = capacity_mask.reshape(-1).to(torch.float32)
        counts = torch.zeros(E, device=topk_idx.device, dtype=torch.float32)
        counts.scatter_add_(0, topk_idx.reshape(-1), mask_flat)
        denom = mask_flat.sum().clamp(min=1.0)
        load = counts / denom

        self.last_expert_load.copy_(load.detach())
        self.last_router_max_load.copy_(load.max().detach())
        norm = math.log(float(E)) if E > 1 else 1.0
        entropy = -(load.clamp(min=1e-8) * load.clamp(min=1e-8).log()).sum() / norm
        self.last_router_entropy.copy_(entropy.detach())
        self.last_capacity_overflow_ratio.copy_(overflow_ratio.detach())

        if self.training:
            gates_full = F.softmax(logits_f, dim=-1)
            importance = gates_full.mean(dim=0)
            if self.use_switch_loss:
                aux_loss = (importance * load).sum() * float(E)
            else:
                aux_loss = ((importance - load) ** 2).mean() * float(E)
            # Collapse flag update stays on-device (no .item() sync).
            max_load = load.max().detach()
            collapsed = max_load > self.collapse_threshold
            recovered = self.collapse_detected & (max_load < 0.5)
            self.collapse_detected.copy_((self.collapse_detected | collapsed) & ~recovered)
        else:
            aux_loss = torch.zeros((), device=x.device, dtype=logits_f.dtype)

        if self.router_z_loss_coef > 0.0:
            z = torch.logsumexp(logits_f, dim=-1)
            aux_loss = aux_loss + (z * z).mean() * self.router_z_loss_coef

        if self.dispatch_mode == "parallel":
            out_flat = self._dispatch_parallel(x_flat, topk_idx, topk_vals, capacity_mask)
        else:
            out_flat = self._dispatch_sequential(x_flat, topk_idx, topk_vals, capacity_mask)

        shared_out = self.shared_expert(x_flat)
        gate_scale = torch.sigmoid(self.shared_gate).to(dtype=shared_out.dtype)
        out_flat = out_flat + shared_out * gate_scale

        return out_flat.reshape(B, T, H), aux_loss

    def router_stats(self) -> dict:
        return {
            "router_entropy": self.last_router_entropy,
            "router_max_load": self.last_router_max_load,
            "capacity_overflow_ratio": self.last_capacity_overflow_ratio,
        }
