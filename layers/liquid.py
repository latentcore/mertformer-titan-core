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
from typing import Tuple, List

from layers.bitlinear import BitLinear


class LiquidCell(nn.Module):
    """
    TR: True Liquid Cell (CfC) - Dynamic Tau & Decoupled Projections.
    EN: True Liquid Cell (CfC) - Dynamic Tau & Decoupled Projections.

    V25.0 UPGRADE:
    - Decoupled `tau` weights from `value` weights.
    - Tau is now input-dependent (True Liquid).
    """

    def __init__(self, h: int) -> None:
        super().__init__()
        # TR: 1. Durum Güncelleme Projeksiyonları (Whal = tanh(W*x + R*h))
        # EN: 1. State Update Projections (Whal = tanh(W*x + R*h))
        self.input_w = BitLinear(h, h, bias=False)
        self.hidden_w = BitLinear(h, h, bias=False)
        
        # TR: 2. Zaman-Sabiti Projeksiyonları (Tau = softplus(W_tau*x + R_tau*h + bias))
        # EN: 2. Time-Constant Projections (Tau = softplus(W_tau*x + R_tau*h + bias))
        # TR: V25.0: Zaman-sabiti için ayrı ağırlıklar / EN: V25.0: Separate weights for time-constant
        self.tau_input_w = BitLinear(h, h, bias=False)
        self.tau_hidden_w = BitLinear(h, h, bias=False)
        # TR: V26.5: Daha yavaş bozulma = daha uzun zamansal hafıza için 0.5 ile başlat
        # EN: V26.5: Initialize with 0.5 for slower decay = longer temporal memory
        self.tau_bias = nn.Parameter(torch.ones(1, h) * 0.5)

    def forward(
        self, x: torch.Tensor, h_prev: torch.Tensor, dt: float = 1.0
    ) -> torch.Tensor:
        """
        Closed-Form Continuous-time (CfC) Step.
        """
        # TR: --- Durum Güncellemesi ("A" terimi) --- / EN: --- State Update (The "A" term) ---
        val_in = self.input_w(x)
        val_rec = self.hidden_w(h_prev)
        A = torch.tanh(val_in + val_rec)
        
        # TR: --- Zaman Sabiti ("Tau" terimi) --- / EN: --- Time Constant (The "Tau" term) ---
        # TR: V25.0: Tamamen girdi-bağımlı dinamik tau / EN: V25.0: Fully input-dependent dynamic tau
        tau_in = self.tau_input_w(x)
        tau_rec = self.tau_hidden_w(h_prev)
        # TR: softplus tau > 0 olmasını sağlar.
        # EN: softplus ensures tau > 0.
        # TR: (input + hidden + bias) "bağlam-farkında" zaman algısı verir.
        # EN: (input + hidden + bias) gives "context-aware" time perception.
        time_decay = F.softplus(tau_in + tau_rec + self.tau_bias)
        
        # TR: --- CfC Güncellemesi --- / EN: --- CfC Update ---
        # TR: h(t) = A + (h_prev - A) * exp(-time_decay * dt) / EN: h(t) = A + (h_prev - A) * exp(-time_decay * dt)
        decay = torch.exp(-time_decay * dt)
        h_new = A + (h_prev - A) * decay
        
        return h_new


# -----------------------------------------------------------------------------
# TR: JIT DERLENMİŞ ÇEKİRDEK (NPU HIZLANDIRICI)
# EN: JIT COMPILED CORE (NPU ACCELERATOR)
# -----------------------------------------------------------------------------
@torch.jit.script
def jit_quant(w: torch.Tensor) -> torch.Tensor:
    """JIT-compatible weight quantization (1.58-bit BitNet)"""
    # [V26.0 FIX] RMS Scale for consistency with BitLinear
    scale = torch.sqrt((w ** 2).mean(dim=1, keepdim=True)).clamp(min=1e-5)
    w_q = torch.round(w / scale).clamp(-1.0, 1.0)
    return w_q * scale


@torch.jit.script
def jit_liquid_loop(
    input_seq: torch.Tensor,
    h_init: torch.Tensor,
    dt: float,
    input_w_weight: torch.Tensor,
    hidden_w_weight: torch.Tensor,
    tau_input_w_weight: torch.Tensor,
    tau_hidden_w_weight: torch.Tensor,
    tau_bias: torch.Tensor
) -> torch.Tensor:
    """
    TR: JIT-Compiled Recurrent Loop for NPU.
    EN: JIT-Compiled Recurrent Loop for NPU.
    
    This function is compiled to a static graph node, removing Python control flow overhead.
    V24.0: BitNet quantization integrated for consistent 1.58-bit inference.
    """
    B, T, H = input_seq.shape
    h = h_init
    # TR: Çıktı için ön-tahsisat (Pre-allocation) - VRAM optimizasyonu
    # EN: Output pre-allocation - VRAM optimization
    out_seq = torch.zeros(B, T, H, device=input_seq.device, dtype=input_seq.dtype)
    
    # TR: Derleyici için açılabilir döngü / EN: Unrollable loop for compiler
    for t in range(T):
        x_t = input_seq[:, t, :]
        
        # TR: --- JIT için manuel olarak açılmış LiquidCell.forward ---
        # EN: --- Manually unrolled LiquidCell.forward for JIT ---
        # TR: V24.0: Ağırlıkları quantize ederek çarp (BitNet Simülasyonu)
        # EN: V24.0: Multiply with quantized weights (BitNet Simulation)
        val_in = torch.matmul(x_t, jit_quant(input_w_weight).t())
        val_rec = torch.matmul(h, jit_quant(hidden_w_weight).t())
        A = torch.tanh(val_in + val_rec)
        
        tau_in = torch.matmul(x_t, jit_quant(tau_input_w_weight).t())
        tau_rec = torch.matmul(h, jit_quant(tau_hidden_w_weight).t())
        
        # TR: V25.1 GÜVENLİK: Tau Sınırı (Patlama/Kaybolmayı Önle)
        # EN: V25.1 SAFEGUARD: Tau Cap (Prevent Exploding/Vanishing)
        # TR: softplus tek başına büyüyebilir; clamp 'her şeyi unutmayı' engeller.
        # EN: softplus alone can grow large; clamping ensures we don't 'forget everything'.
        raw_tau = torch.nn.functional.softplus(tau_in + tau_rec + tau_bias)
        time_decay = torch.clamp(raw_tau, min=1e-4, max=5.0)
        
        decay = torch.exp(-time_decay * dt)
        h = A + (h - A) * decay
        
        out_seq[:, t, :] = h
    
    return out_seq


class LiquidMixer(nn.Module):
    """
    TR: Liquid Mixer V25.0 - JIT Accelerated.
    EN: Liquid Mixer V25.0 - JIT Accelerated.
    """

    def __init__(self, h: int) -> None:
        super().__init__()
        self.cell = LiquidCell(h)
        self.norm = nn.LayerNorm(h)

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        B, T, H = x.shape
        h_init = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        
        # TR: [V26.0 FIX] Eğitim vs Çıkarım Yolu / EN: [V26.0 FIX] Training vs Inference Path
        # TR: Eğitim: Python döngüsü (LiquidCell) kullan -> STE gradyanları çalışır.
        # EN: Training: Python loop (LiquidCell) use -> STE gradients work.
        # TR: Çıkarım: JIT döngüsü kullan -> NPU optimizasyonu.
        # EN: Inference: JIT loop use -> NPU optimization.
        if self.training:
            h = h_init
            outs = []
            for t in range(T):
                h = self.cell(x[:, t, :], h, dt)
                outs.append(h)
            out_seq = torch.stack(outs, dim=1)
        else:
            # TR: V25.0 NPU: JIT derlenmiş döngü / EN: V25.0 NPU: JIT compiled loop
            out_seq = jit_liquid_loop(
                x, 
                h_init, 
                dt,
                self.cell.input_w.weight,
                self.cell.hidden_w.weight,
                self.cell.tau_input_w.weight,
                self.cell.tau_hidden_w.weight,
                self.cell.tau_bias
            )
        
        # TR: [V26.4 FIX] Residual'ı Geri Yükle (Block Liquid için residual eklemez)
        # EN: [V26.4 FIX] Restore Residual (Block does NOT add residual for Liquid)
        return self.norm(out_seq + x)
