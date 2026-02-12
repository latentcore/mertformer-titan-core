"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 27) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD27"
__author__ = "Mert"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

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
            padding=self.history_window - 1, # TR: Causal padding / EN: Causal padding 
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
            if (
                self.inference_state.device != target.device
                or self.inference_state.dtype != target.dtype
            ):
                self.inference_state = self.inference_state.to(
                    device=target.device, dtype=target.dtype
                )
            self.inference_state.resize_(*target.shape)
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
            
            # Causal Convolution
            fluid_mem = self.fluid_mixer(x_t)
            fluid_mem = fluid_mem[..., :S] # Causal Crop
            
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
            
            # Conv pass
            fluid_mem = self.fluid_mixer(context) # [B, H, Window]
            
            # [FIX 1] Causal Align: Crop to input context size (training consistency)
            fluid_mem = fluid_mem[..., :context.size(2)] 
            
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
        
        # V26.5: Switch Loss and Collapse Recovery
        self.use_switch_loss: bool = bool(getattr(cfg, "use_switch_loss", False))
        self.router_jitter_boost: float = float(getattr(cfg, "router_jitter_boost", 0.1))
        self.collapse_threshold: float = 0.85  # Max load threshold for collapse detection
        
        # Telemetry & Collapse State
        self.register_buffer("last_expert_load", torch.zeros(self.num_experts))
        self.register_buffer("collapse_detected", torch.tensor(False))
        
    def get_router_state(self) -> torch.Tensor:
        """External API: Get Liquid Router state."""
        return self.router.get_state()
        
    def set_router_state(self, state: torch.Tensor) -> None:
        """External API: Set Liquid Router state."""
        self.router.set_state(state)
        
    def get_expert_load(self) -> torch.Tensor:
        """External API: Get last expert load distribution."""
        return self.last_expert_load

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

        # [TELEMETRY] Load Calculation (Always compute for monitoring)
        # [FIX 2] MPS Safe Bincount: Use scatter_add_ instead of bincount for compatibility
        flat_idx = topk_idx.reshape(-1)
        counts = torch.zeros(E, device=flat_idx.device, dtype=torch.float32)
        counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        
        load = counts / float(N * k)
        self.last_expert_load = load.detach() # Store for logging

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
        out_flat = x_flat.new_zeros((N, H))

        # TR: GPU->CPU sync yapan unique(...).tolist() yerine sabit uzman döngüsü
        # EN: Avoid unique(...).tolist() CPU sync; use fixed expert loop
        for expert_id_int, expert in enumerate(self.experts):

            # Maskeleme: Hangi token'lar bu uzmanı seçti?
            expert_mask = topk_idx == expert_id_int

            # Bu token'ın herhangi bir slotunda bu uzman var mı?
            token_mask = expert_mask.any(dim=-1)

            # Sadece seçen token'ları al
            selected_x = x_flat[token_mask]  # (M, H)

            # Uzmanı çalıştır
            expert_out = expert(selected_x)  # (M, H)

            # Ağırlıklandırma: Bu token için bu uzmana ait weight'lerin toplamı
            weights = (
                topk_vals[token_mask] * expert_mask[token_mask].float()
            ).sum(dim=-1, keepdim=True)  # (M, 1)

            # Sonuca ekle (scatter add)
            out_flat[token_mask] += expert_out * weights

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

        # Şekli geri yükle
        out = out_flat.reshape(B, T, H)

        # [FIX 3] Keep aux_loss as float32 for precision stability
        return out, aux_loss
