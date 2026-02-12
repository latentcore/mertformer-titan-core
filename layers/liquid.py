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
import sys
import warnings
from typing import Optional, Tuple

from layers.bitlinear import BitLinear


def _jit_script_if_supported(fn):
    """Use TorchScript where supported; fall back cleanly on Python 3.14+."""
    if sys.version_info >= (3, 14):
        return fn
    try:
        return torch.jit.script(fn)
    except Exception as exc:
        warnings.warn(
            f"TorchScript disabled for liquid kernel due to: {exc}",
            RuntimeWarning,
        )
        return fn


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
@_jit_script_if_supported
def jit_quant(w: torch.Tensor) -> torch.Tensor:
    """JIT-compatible weight quantization (1.58-bit BitNet)"""
    # TR: Quant hesaplarını fp32'de yap, sonra orijinal dtype'a dön.
    # EN: Do quant math in fp32, then cast back to original dtype.
    w_f = w.float()
    scale = torch.sqrt((w_f * w_f).mean(dim=1, keepdim=True)).clamp(min=1e-5)
    w_q = torch.round(w_f / scale).clamp(-1.0, 1.0)
    return (w_q * scale).to(dtype=w.dtype)


@_jit_script_if_supported
def jit_liquid_loop_cached(
    input_seq: torch.Tensor,
    h_init: torch.Tensor,
    dt: float,
    input_w_q_t: torch.Tensor,
    hidden_w_q_t: torch.Tensor,
    tau_input_w_q_t: torch.Tensor,
    tau_hidden_w_q_t: torch.Tensor,
    tau_bias: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TR: JIT-Compiled Recurrent Loop for NPU.
    EN: JIT-Compiled Recurrent Loop for NPU.
    
    This function is compiled to a static graph node, removing Python control flow overhead.
    V24.0: BitNet quantization integrated for consistent 1.58-bit inference.
    """
    B, T, H = input_seq.shape
    h = h_init
    # TR: Çıktı için ön-tahsisat (Pre-allocation)
    # EN: Output pre-allocation
    out_seq = torch.zeros(B, T, H, device=input_seq.device, dtype=input_seq.dtype)
    
    # TR: Derleyici için açılabilir döngü / EN: Unrollable loop for compiler
    for t in range(T):
        x_t = input_seq[:, t, :]
        
        # TR: --- JIT için manuel olarak açılmış LiquidCell.forward ---
        # EN: --- Manually unrolled LiquidCell.forward for JIT ---
        # TR: V24.0: Ağırlıkları quantize ederek çarp (BitNet Simülasyonu)
        # EN: V24.0: Multiply with quantized weights (BitNet Simulation)
        val_in = torch.matmul(x_t, input_w_q_t)
        val_rec = torch.matmul(h, hidden_w_q_t)
        A = torch.tanh(val_in + val_rec)
        
        tau_in = torch.matmul(x_t, tau_input_w_q_t)
        tau_rec = torch.matmul(h, tau_hidden_w_q_t)
        
        # TR: V25.1 GÜVENLİK: Tau Sınırı (Patlama/Kaybolmayı Önle)
        # EN: V25.1 SAFEGUARD: Tau Cap (Prevent Exploding/Vanishing)
        # TR: softplus tek başına büyüyebilir; clamp 'her şeyi unutmayı' engeller.
        # EN: softplus alone can grow large; clamping ensures we don't 'forget everything'.
        raw_tau = torch.nn.functional.softplus(tau_in + tau_rec + tau_bias)
        time_decay = torch.clamp(raw_tau, min=1e-4, max=5.0)
        
        decay = torch.exp(-time_decay * dt)
        h = A + (h - A) * decay
        
        out_seq[:, t, :] = h
    
    return out_seq, h


@_jit_script_if_supported
def jit_liquid_loop(
    input_seq: torch.Tensor,
    h_init: torch.Tensor,
    dt: float,
    input_w_weight: torch.Tensor,
    hidden_w_weight: torch.Tensor,
    tau_input_w_weight: torch.Tensor,
    tau_hidden_w_weight: torch.Tensor,
    tau_bias: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward-compatible wrapper:
    quantizes once per call, then runs cached loop kernel.
    """
    input_w_q_t = jit_quant(input_w_weight).t().contiguous()
    hidden_w_q_t = jit_quant(hidden_w_weight).t().contiguous()
    tau_input_w_q_t = jit_quant(tau_input_w_weight).t().contiguous()
    tau_hidden_w_q_t = jit_quant(tau_hidden_w_weight).t().contiguous()
    return jit_liquid_loop_cached(
        input_seq,
        h_init,
        dt,
        input_w_q_t,
        hidden_w_q_t,
        tau_input_w_q_t,
        tau_hidden_w_q_t,
        tau_bias.to(dtype=input_seq.dtype),
    )


class LiquidMixer(nn.Module):
    """
    TR: Liquid Mixer V25.0 - JIT Accelerated.
    EN: Liquid Mixer V25.0 - JIT Accelerated.
    """

    def __init__(self, h: int) -> None:
        super().__init__()
        self.cell = LiquidCell(h)
        self.norm = nn.LayerNorm(h)

        # TR: Eval/inference quant cache (checkpoint'e yazılmaz)
        # EN: Eval/inference quant cache (excluded from checkpoints)
        self.register_buffer("_q_input_w_t", torch.empty(0), persistent=False)
        self.register_buffer("_q_hidden_w_t", torch.empty(0), persistent=False)
        self.register_buffer("_q_tau_input_w_t", torch.empty(0), persistent=False)
        self.register_buffer("_q_tau_hidden_w_t", torch.empty(0), persistent=False)
        self.register_buffer("_q_tau_bias", torch.empty(0), persistent=False)
        self._cache_ready = False

    def _set_cache(self, name: str, value: torch.Tensor) -> None:
        buf = getattr(self, name)
        if buf.device != value.device or buf.dtype != value.dtype:
            setattr(self, name, value.detach().clone())
            return
        buf.resize_(value.shape[0], value.shape[1] if value.dim() > 1 else 1)
        if value.dim() == 1:
            buf.copy_(value.unsqueeze(0)).resize_(value.shape[0])
        else:
            buf.copy_(value)

    def reset_stream_state(self) -> None:
        self._q_input_w_t.resize_(0)
        self._q_hidden_w_t.resize_(0)
        self._q_tau_input_w_t.resize_(0)
        self._q_tau_hidden_w_t.resize_(0)
        self._q_tau_bias.resize_(0)
        self._cache_ready = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.reset_stream_state()
        return self

    def _ensure_qcache(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.training:
            return
        if (
            self._cache_ready
            and self._q_input_w_t.numel() > 0
            and self._q_input_w_t.device == device
            and self._q_input_w_t.dtype == dtype
        ):
            return

        with torch.no_grad():
            iw = jit_quant(self.cell.input_w.weight).to(device=device, dtype=dtype).t().contiguous()
            hw = jit_quant(self.cell.hidden_w.weight).to(device=device, dtype=dtype).t().contiguous()
            tiw = jit_quant(self.cell.tau_input_w.weight).to(device=device, dtype=dtype).t().contiguous()
            thw = jit_quant(self.cell.tau_hidden_w.weight).to(device=device, dtype=dtype).t().contiguous()
            tb = self.cell.tau_bias.to(device=device, dtype=dtype).contiguous()

            self._set_cache("_q_input_w_t", iw)
            self._set_cache("_q_hidden_w_t", hw)
            self._set_cache("_q_tau_input_w_t", tiw)
            self._set_cache("_q_tau_hidden_w_t", thw)
            # tau_bias is [1, H]
            if self._q_tau_bias.device != tb.device or self._q_tau_bias.dtype != tb.dtype:
                self._q_tau_bias = tb.detach().clone()
            else:
                self._q_tau_bias.resize_(tb.shape[0], tb.shape[1]).copy_(tb)

            self._cache_ready = True

    def forward(
        self,
        x: torch.Tensor,
        dt: float = 1.0,
        h_init: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        B, T, H = x.shape
        if h_init is None:
            h = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        else:
            if h_init.dim() != 2 or h_init.shape != (B, H):
                raise ValueError(f"h_init must be [B,H] = [{B},{H}], got {tuple(h_init.shape)}")
            if h_init.device != x.device or h_init.dtype != x.dtype:
                raise RuntimeError(
                    "h_init device/dtype mismatch with x. "
                    f"h_init={h_init.device}/{h_init.dtype}, x={x.device}/{x.dtype}."
                )
            h = h_init
        
        # TR: [V26.0 FIX] Eğitim vs Çıkarım Yolu / EN: [V26.0 FIX] Training vs Inference Path
        # TR: Eğitim: Python döngüsü (LiquidCell) kullan -> STE gradyanları çalışır.
        # EN: Training: Python loop (LiquidCell) use -> STE gradients work.
        # TR: Çıkarım: JIT döngüsü kullan -> NPU optimizasyonu.
        # EN: Inference: JIT loop use -> NPU optimization.
        if self.training:
            out_seq = torch.empty(B, T, H, device=x.device, dtype=x.dtype)
            for t in range(T):
                h = self.cell(x[:, t, :], h, dt)
                out_seq[:, t, :] = h
        else:
            # TR: Eval'de quant cache ile JIT döngüsü (forward başına tekrar quant yok)
            # EN: Eval uses quant cache + JIT loop (no repeated quant per forward)
            self._ensure_qcache(device=x.device, dtype=x.dtype)
            out_seq, h = jit_liquid_loop_cached(
                x,
                h,
                dt,
                self._q_input_w_t,
                self._q_hidden_w_t,
                self._q_tau_input_w_t,
                self._q_tau_hidden_w_t,
                self._q_tau_bias,
            )
        
        # TR: [V26.4 FIX] Residual'ı Geri Yükle (Block Liquid için residual eklemez)
        # EN: [V26.4 FIX] Restore Residual (Block does NOT add residual for Liquid)
        y = self.norm(out_seq + x)
        if return_state:
            return y, h
        return y
