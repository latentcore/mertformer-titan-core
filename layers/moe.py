"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD30"
__author__ = "Mert"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, List, Set, Tuple

from config.config import cfg
from layers.bitlinear import BitLinear


class BitSwiGLU(nn.Module):
    """
    TR: BitNet SwiGLU bloğu - MoE uzmanları için optimizasyon.
    EN: BitNet SwiGLU block - Optimization for MoE experts.

    Özellikler / Features:
    - BitLinear projeksiyonlar ile hafıza tasarrufu / Memory savings via BitLinear
    - SwiGLU aktivasyon fonksiyonu / SwiGLU activation function
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        """
        TR: BitSwiGLU başlatıcı.
        EN: BitSwiGLU initializer.

        Args:
            hidden_size (int): Gizli boyut / Hidden dimension
            intermediate_size (int): Ara boyut / Intermediate dimension
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # TR: BitLinear projeksiyonlar: gate, up, down
        # EN: BitLinear projections: gate, up, down
        self.gate_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = BitLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TR: İleri yayılım - SwiGLU aktivasyonu uygular.
        EN: Forward pass - Applies SwiGLU activation.

        Args:
            x (torch.Tensor): Girdi tensörü / Input tensor
        Returns:
            torch.Tensor: Çıktı tensörü / Output tensor
        """
        # TR: SwiGLU: SiLU(gate) * up -> down / EN: SwiGLU: SiLU(gate) * up -> down
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x_inter = F.silu(gate) * up
        return self.down_proj(x_inter)




class LiquidRouter(nn.Module):
    """
    TR: Akışkan Yönlendirici (Fluid Router) - Zamansal bağlamı kullanan uzman seçici.
    EN: Liquid Router - Expert selector utilizing temporal context.
    
    Teknoloji / Tech:
    - Main Path: Anlık kelimeye bakar (Standard Routing).
    - Fluid Path: Önceki kelimelerin 'momentumuna' bakar (Context Momentum).
    - Mekanizma: Causal Depthwise Conv1d ve Rolling Buffer.
    """
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        # TR: [V26.0 FIX] Router Quantization: Tutarlı 1.58-bit için BitLinear kullan
        # EN: [V26.0 FIX] Router Quantization: Use BitLinear for consistent 1.58-bit
        self.main_proj = BitLinear(hidden_size, num_experts, bias=False)
        
        # TR: Fluid Dynamics: Hafif tarihçe (History/Momentum)
        # EN: Fluid Dynamics: Lightweight history (History/Momentum)
        self.history_window = 4
        self.fluid_mixer = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=self.history_window,
            groups=hidden_size, # TR: Depthwise / EN: Depthwise
            padding=0, # TR: Sol pad'i manuel uygularız / EN: Left padding is applied manually
            bias=False
        )
        # TR: [V26.0 FIX] Router Quantization / EN: [V26.0 FIX] Router Quantization
        self.fluid_gate = BitLinear(hidden_size, num_experts, bias=False)
        nn.init.zeros_(self.fluid_gate.weight)
        
        # TR: [FIX 2] Stateful Inference Buffer
        # EN: [FIX 2] Stateful Inference Buffer
        # TR: Inference sırasında son (history_window - 1) token'ın state'ini tutar
        # EN: Holds the state of last (history_window - 1) tokens during inference
        # TR: Inference cache, checkpoint'e yazılmaz (runtime state)
        # EN: Inference cache, excluded from checkpoints (runtime state)
        self.register_buffer(
            "inference_state",
            torch.zeros(1, hidden_size, self.history_window - 1),
            persistent=False,
        )

    def _update_inference_state(self, state: torch.Tensor) -> None:
        """
        TR: Router state'ini güvenli şekilde günceller (dtype/device + shape uyumu).
        EN: Safely updates router state (dtype/device + shape alignment).
        """
        with torch.no_grad():
            target = state.detach()
            if self.inference_state.device != target.device or self.inference_state.dtype != target.dtype:
                raise RuntimeError(
                    "inference_state device/dtype mismatch: "
                    f"state={self.inference_state.device}/{self.inference_state.dtype}, "
                    f"target={target.device}/{target.dtype}. "
                    "Move the whole model via model.to(...)."
                )
            self.inference_state.resize_(target.shape[0], target.shape[1], target.shape[2])
            self.inference_state.copy_(target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TR: x: [Batch, Seq, Hidden] veya [Batch*Seq, Hidden] (İlk gelene göre 3D yapılmalı)
        # EN: x: [Batch, Seq, Hidden] or [Batch*Seq, Hidden] (Must be made 3D if flatten)
        
        # TR: [FIX 1] Boyut Yönetimi / EN: [FIX 1] Dimension Handling
        is_flat = False
        if x.dim() == 2:
            is_flat = True
            # TR: Flatten gelmişse (Batch*Seq, Hidden) -> (Batch*Seq, 1, Hidden)
            # EN: If flatten received (Batch*Seq, Hidden) -> (Batch*Seq, 1, Hidden)
            # TR: Güvenlik için: / EN: For safety:
            x = x.unsqueeze(1) # (N, 1, H)

        B, S, H = x.shape
        
        # TR: 1. Anlık Karar (Main Path) / EN: 1. Instant Decision (Main Path)
        logits_main = self.main_proj(x)
        
        # TR: 2. Akışkan Karar (Fluid Path) / EN: 2. Fluid Decision (Fluid Path)
        if self.training or S > 1:
            # Training / Prefill mode (Parallel)
            x_t = x.transpose(1, 2) # [B, H, S]
            
            # TR: True-causal conv: only left padding (no right-side leakage)
            # EN: True-causal conv: only left padding (no right-side leakage)
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
            # [FIX 4] Batch-Safe Cache Handling
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
            
            # TR: True-causal conv with explicit left padding (inference parity)
            # EN: True-causal conv with explicit left padding (inference parity)
            context_padded = F.pad(context, (self.history_window - 1, 0))
            fluid_mem = self.fluid_mixer(context_padded) # [B, H, Window]
            
            # Sadece son adımı al (current token output)
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
        TR: Router'ın o anki hafızasını (state) döndürür.
        EN: Returns the current memory (state) of the router.
        Returns:
            torch.Tensor: State buffer [Batch, Hidden, Window-1]
        """
        return self.inference_state.clone()

    def set_state(self, state: torch.Tensor) -> None:
        """
        TR: Router hafızasını dışarıdan yükler.
        EN: Loads router memory from external source.
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
    TR: Sparse Mixture of Experts - Top-k routing ile verimli uzman seçimi.
    EN: Sparse Mixture of Experts - Efficient expert selection via top-k routing.
    
    Özellikler / Features:
    - Optimize edilmiş dispatch: Sadece seçilen uzmanlar çalıştırılır
    - Hafıza verimli: bincount ile load balancing
    - Öğrenilebilir shared expert: Ortak uzman katkısı öğrenilebilir
    - Dtype güvenli: Mixed precision eğitiminde sorunsuz çalışır
    """

    def __init__(self) -> None:
        """TR: MoE başlatıcı. EN: MoE initializer."""
        super().__init__()

        self.hidden_size: int = int(cfg.hidden_size)
        self.num_experts: int = int(cfg.num_experts)
        # num_experts_per_tok kullan (geriye dönük uyumluluk için fallback)
        self.active_experts: int = int(
            getattr(cfg, "num_experts_per_tok", getattr(cfg, "active_experts", 1))
        )

        # Güvenlik kontrolleri
        assert self.num_experts >= 1, "num_experts en az 1 olmalı."
        assert (
            1 <= self.active_experts <= self.num_experts
        ), "active_experts mantıksız (1 <= k <= N)."

        moe_intermediate = int(getattr(cfg, "moe_intermediate", self.hidden_size * 4))

        # Router: Token'ları uzmanlara yönlendirir
        # [TITAN INNOVATION]: Liquid Router (Akışkan Yönlendirici)
        self.router = LiquidRouter(self.hidden_size, self.num_experts)

        # Experts: Her uzman bir BitSwiGLU bloğu
        self.experts = nn.ModuleList(
            BitSwiGLU(self.hidden_size, moe_intermediate)
            for _ in range(self.num_experts)
        )

        # Shared Expert: Tüm token'lar için ortak uzman (öğrenilebilir gate ile)
        self.shared_expert = BitSwiGLU(self.hidden_size, moe_intermediate)
        shared_gate_init = float(getattr(cfg, "shared_expert_gate", 0.0))
        # nn.Parameter olması önemli - model bunu optimize eder
        self.shared_gate = nn.Parameter(
            torch.tensor([shared_gate_init], dtype=torch.float32)
        )

        # Router hiperparametreleri
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
        
        # V26.5: Switch Loss and Collapse Recovery
        self.use_switch_loss: bool = bool(getattr(cfg, "use_switch_loss", False))
        self.router_jitter_boost: float = float(getattr(cfg, "router_jitter_boost", 0.1))
        self.collapse_threshold: float = 0.85  # Max load threshold for collapse detection
        self.moe_capacity_enforce: bool = bool(getattr(cfg, "moe_capacity_enforce", True))
        self.moe_capacity_factor: float = float(getattr(cfg, "moe_capacity_factor", 1.25))
        self.dispatch_mode: str = str(getattr(cfg, "moe_dispatch_mode", "sequential")).lower()
        
        # Telemetry & Collapse State
        self.register_buffer("last_expert_load", torch.zeros(self.num_experts))
        self.register_buffer("last_router_entropy", torch.tensor(0.0))
        self.register_buffer("last_router_max_load", torch.tensor(0.0))
        self.register_buffer("last_capacity_overflow_ratio", torch.tensor(0.0))
        self.register_buffer("collapse_detected", torch.tensor(False))
        self.register_buffer("expert_activity_mask", torch.ones(self.num_experts, dtype=torch.bool))
        self.register_buffer("expert_usage_ema", torch.zeros(self.num_experts))
        self.register_buffer("plasticity_step", torch.zeros((), dtype=torch.int64))
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
        TR: İleri yayılım - MoE routing ve uzman dispatch.
        EN: Forward pass - MoE routing and expert dispatch.

        Args:
            x (torch.Tensor): Girdi tensörü / Input tensor [Batch, Seq, Hidden]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Çıktı, Aux loss) / (Output, Aux loss)
        """
        B, T, H = x.shape
        
        # -----------------------------
        # 1) Router & Gating (LIQUID FIX)
        # -----------------------------
        # [FIX 1] Pass 3D tensor to LiquidRouter to preserve temporal context
        logits = self.router(x) # (B, T, E)
        
        # Flatten after routing for dispatch logic
        logits = logits.reshape(-1, self.num_experts) # (N, E)
        x_flat = x.reshape(-1, H)
        N = x_flat.size(0)
        E = self.num_experts

        # Hesaplamalar FP32'de daha kararlı
        logits_f = logits.float()
        active_mask = self.expert_activity_mask.to(device=logits_f.device)
        if active_mask.any():
            logits_f = logits_f.masked_fill(~active_mask.unsqueeze(0), float("-inf"))

        # Temperature scaling
        if self.router_temperature != 1.0:
            logits_f = logits_f / self.router_temperature

        # Jitter: Sadece eğitimde noise ekle (exploration için)
        applied_jitter = self.router_jitter
        
        # [DESIGN] Collapse Recovery logic moved here to avoid permanent generic mutation
        if self.training and self.collapse_detected.item():
             applied_jitter = self.router_jitter_boost
        
        if self.training and applied_jitter > 0.0:
            logits_f = logits_f + torch.randn_like(logits_f) * applied_jitter

        # [OPT 1] Top-K First, then Softmax (NPU Optimization)
        # Eski yöntem: Softmax(all) -> TopK
        # Yeni yöntem: TopK -> Softmax(selected)
        # Bu yöntem FLOP tasarrufu sağlar ve daha keskin kararlar verir.
        k = min(self.active_experts, E)
        topk_logits, topk_idx = torch.topk(logits_f, k=k, dim=-1) # (N, k)
        
        topk_vals = F.softmax(topk_logits, dim=-1) # (N, k) - Sadece K elemana softmax
        capacity_mask = torch.ones_like(topk_idx, dtype=torch.bool)
        overflow_ratio = torch.tensor(0.0, device=x.device, dtype=torch.float32)

        # Switch-style capacity control: cap per-expert assignments and renormalize gates.
        if self.moe_capacity_enforce and self.moe_capacity_factor > 0.0:
            capacity = max(1, int(math.ceil(self.moe_capacity_factor * (N * k) / max(1, E))))
            dropped = 0
            for expert_id_int in range(E):
                hits = (topk_idx == expert_id_int).nonzero(as_tuple=False)
                if hits.size(0) > capacity:
                    overflow = hits[capacity:]
                    capacity_mask[overflow[:, 0], overflow[:, 1]] = False
                    dropped += int(overflow.size(0))

            topk_vals = topk_vals * capacity_mask.float()
            empty_rows = topk_vals.sum(dim=-1) <= 0
            if empty_rows.any():
                topk_vals[empty_rows, 0] = 1.0
                capacity_mask[empty_rows, 0] = True

            topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            overflow_ratio = torch.tensor(
                float(dropped) / float(max(1, N * k)),
                device=x.device,
                dtype=torch.float32,
            )

        # [TELEMETRY] Load Calculation (Always compute for monitoring)
        # [FIX 2] MPS Safe Bincount: Use scatter_add_ instead of bincount for compatibility
        flat_idx = topk_idx[capacity_mask].reshape(-1)
        counts = torch.zeros(E, device=flat_idx.device, dtype=torch.float32)
        if flat_idx.numel() > 0:
            counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        
        denom = float(max(1, int(flat_idx.numel())))
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
        # 2) Aux Loss (Load Balancing) - V26.5: Switch/L2 Option
        # -----------------------------
        # Importance: Gate'lerin kümülatif olasılığı (Tüm uzmanlar için hesaplamamız lazım)
        if self.training:
             gates_full = F.softmax(logits_f, dim=-1) # (N, E)
             importance = gates_full.mean(dim=0) # (E,)
             
             if self.use_switch_loss:
                 # Switch Transformer formülü: Daha agresif, hızlı öğrenme
                 load_balancing_loss = (importance * load).sum() * float(E)
             else:
                 # L2 Loss: Daha stabil, Liquid Router için önerilir
                 load_balancing_loss = ((importance - load) ** 2).mean() * float(E)
             
             # V26.5: Router Collapse Detection & Recovery
             max_load = load.max().item()
             if max_load > self.collapse_threshold:
                 self.collapse_detected.fill_(True)
                 # Jitter boost is handled via 'applied_jitter' during forward pass
             elif self.collapse_detected.item() and max_load < 0.5:
                 # Recovery: Reset state
                 self.collapse_detected.fill_(False)
        else:
             load_balancing_loss = torch.tensor(0.0, device=x.device, dtype=logits_f.dtype)

        aux_loss = load_balancing_loss

        # Z-Loss: Logitlerin aşırı büyümesini engeller
        if self.router_z_loss_coef > 0.0:
            z = torch.logsumexp(logits_f, dim=-1)  # (N,)
            z_loss = (z * z).mean() * self.router_z_loss_coef
            aux_loss = aux_loss + z_loss

        # -----------------------------
        # 3) Dispatch (Uzmanlara Dağıtım)
        # -----------------------------
        if self.dispatch_mode == "parallel":
            out_flat = self._dispatch_parallel(x_flat, topk_idx, topk_vals, capacity_mask)
        else:
            out_flat = self._dispatch_sequential(x_flat, topk_idx, topk_vals)

        # -----------------------------
        # 4) Shared Expert (Dtype Safe & [FIX 3] Sigmoid Gate)
        # -----------------------------
        shared_out = self.shared_expert(x_flat)  # (N, H)

        # [FIX 3] Sigmoid Gate: Gate'in 0-1 arasında olmasını zorla.
        # Negatif öğrenmeyi engeller ve stabilite sağlar.
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

        # Şekli geri yükle
        out = out_flat.reshape(B, T, H)

        # [FIX 3] Keep aux_loss as float32 for precision stability
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

        counts = torch.bincount(expert_sorted, minlength=self.num_experts)
        if counts.numel() == 0:
            return out_flat

        start = 0
        for expert_id_int, expert in enumerate(self.experts):
            cnt = int(counts[expert_id_int].item())
            if cnt == 0:
                continue
            end = start + cnt
            idx = token_sorted[start:end]
            w = weight_sorted[start:end].unsqueeze(-1)
            selected_x = x_flat.index_select(0, idx)
            expert_param = next(expert.parameters(), None)
            if expert_param is not None and selected_x.dtype != expert_param.dtype:
                selected_x = selected_x.to(dtype=expert_param.dtype)
            expert_out = expert(selected_x)
            if expert_out.dtype != out_flat.dtype:
                expert_out = expert_out.to(dtype=out_flat.dtype)
            out_flat.index_add_(0, idx, expert_out * w)
            start = end
        return out_flat
