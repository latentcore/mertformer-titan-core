"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert Yünlü"

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import warnings
from typing import Optional, Tuple

from layers.bitlinear import BitLinear, activation_quant, weight_quant

DEFAULT_TORCHSCRIPT_COMPAT_ENV = "MERTFORMER_ENABLE_TORCHSCRIPT_COMPAT"
DEFAULT_LIQUID_TRAIN_IMPL_ENV = "TITAN_LIQUID_TRAIN_IMPL"
LIQUID_TRAIN_IMPLS = {"baseline", "precompute_input", "packed_pair", "packed_pair_compile"}


def _jit_script_if_supported(fn):
    """Use TorchScript only when explicitly requested; default verify path stays warning-free."""
    if sys.version_info >= (3, 14):
        return fn
    if os.environ.get(DEFAULT_TORCHSCRIPT_COMPAT_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}:
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

    Notes:
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
        # TR: Zaman-sabiti için ayrı ağırlıklar / EN: Separate weights for time-constant
        self.tau_input_w = BitLinear(h, h, bias=False)
        self.tau_hidden_w = BitLinear(h, h, bias=False)
        # TR: Daha yavaş bozulma = daha uzun zamansal hafıza için 0.5 ile başlat
        # EN: Initialize with 0.5 for slower decay = longer temporal memory
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
        # TR: Tamamen girdi-bağımlı dinamik tau / EN: Fully input-dependent dynamic tau
        tau_in = self.tau_input_w(x)
        tau_rec = self.tau_hidden_w(h_prev)
        # TR: softplus tau > 0 olmasını sağlar.
        # EN: softplus ensures tau > 0.
        # TR: (input + hidden + bias) "bağlam-farkında" zaman algısı verir.
        # EN: (input + hidden + bias) gives "context-aware" time perception.
        time_decay = F.softplus(tau_in + tau_rec + self.tau_bias)
        time_decay = torch.clamp(time_decay, min=1e-4, max=5.0)
        
        # TR: --- CfC Güncellemesi --- / EN: --- CfC Update ---
        # TR: h(t) = A + (h_prev - A) * exp(-time_decay * dt) / EN: h(t) = A + (h_prev - A) * exp(-time_decay * dt)
        decay = torch.exp(torch.clamp(-time_decay * dt, min=-20.0, max=20.0))
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
    BitNet quantization integrated for consistent 1.58-bit inference.
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
        # TR: Ağırlıkları quantize ederek çarp (BitNet Simülasyonu)
        # EN: Multiply with quantized weights (BitNet Simulation)
        val_in = torch.matmul(x_t, input_w_q_t)
        val_rec = torch.matmul(h, hidden_w_q_t)
        A = torch.tanh(val_in + val_rec)
        
        tau_in = torch.matmul(x_t, tau_input_w_q_t)
        tau_rec = torch.matmul(h, tau_hidden_w_q_t)
        
        # TR: GÜVENLİK: Tau Sınırı (Patlama/Kaybolmayı Önle)
        # EN: SAFEGUARD: Tau Cap (Prevent Exploding/Vanishing)
        # TR: softplus tek başına büyüyebilir; clamp 'her şeyi unutmayı' engeller.
        # EN: softplus alone can grow large; clamping ensures we don't 'forget everything'.
        raw_tau = torch.nn.functional.softplus(tau_in + tau_rec + tau_bias)
        time_decay = torch.clamp(raw_tau, min=1e-4, max=5.0)
        
        decay = torch.exp(torch.clamp(-time_decay * dt, min=-20.0, max=20.0))
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
        tau_bias.to(device=input_seq.device, dtype=input_seq.dtype),
    )


class LiquidMixer(nn.Module):
    """
    TR: Liquid Mixer - fast_path (torch.compile guarded).
    EN: Liquid Mixer - fast_path (torch.compile guarded).
    """

    def __init__(
        self,
        h: int,
        fast_path: Optional[bool] = None,
        train_impl: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cell = LiquidCell(h)
        self.norm = nn.LayerNorm(h)
        if fast_path is None:
            fast_path = os.environ.get("TITAN_LIQUID_FAST_PATH", "1") == "1"
        self.fast_path = bool(fast_path)
        if train_impl is None:
            train_impl = os.environ.get(DEFAULT_LIQUID_TRAIN_IMPL_ENV, "baseline")
        train_impl = str(train_impl).strip().lower()
        if train_impl not in LIQUID_TRAIN_IMPLS:
            raise ValueError(
                f"{DEFAULT_LIQUID_TRAIN_IMPL_ENV} must be one of "
                f"{sorted(LIQUID_TRAIN_IMPLS)}, got {train_impl!r}"
            )
        self.train_impl = train_impl
        self._compiled_train_loop = None
        self._compiled_packed_train_loop = None

        # TR: Eval/inference quant cache (checkpoint'e yazılmaz)
        # EN: Eval/inference quant cache (excluded from checkpoints)
        self.register_buffer("_q_input_w_t", torch.empty(0), persistent=False)
        self.register_buffer("_q_hidden_w_t", torch.empty(0), persistent=False)
        self.register_buffer("_q_tau_input_w_t", torch.empty(0), persistent=False)
        self.register_buffer("_q_tau_hidden_w_t", torch.empty(0), persistent=False)
        self.register_buffer("_q_tau_bias", torch.empty(0), persistent=False)
        self.register_buffer("_weight_version", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("_cached_weight_version", torch.full((), -1, dtype=torch.int64), persistent=False)
        self._cache_ready = False

    def _set_cache(self, name: str, value: torch.Tensor) -> None:
        """
        BUFFER-SAFE CACHE WRITE
        - preserves buffer tracking/state_dict semantics
        - preserves tensor identity for existing buffers
        """
        value = value.detach().contiguous()
        buf = self._buffers.get(name, None)

        if buf is None:
            self.register_buffer(name, value.clone(), persistent=False)
            return

        if buf.device != value.device or buf.dtype != value.dtype:
            # Keep the registered buffer object and only swap backing storage.
            # This avoids plain attribute/buffer rebind patterns.
            buf.data = buf.data.to(device=value.device, dtype=value.dtype)  # nosec B614

        buf.resize_(value.shape)
        buf.copy_(value)

    def _compute_weight_version(self) -> int:
        # TR: Parametre in-place güncellemelerini izlemek için sürüm imzası.
        # EN: Version signature to detect in-place parameter updates.
        return int(
            self.cell.input_w.weight._version
            + self.cell.hidden_w.weight._version
            + self.cell.tau_input_w.weight._version
            + self.cell.tau_hidden_w.weight._version
            + self.cell.tau_bias._version
        )

    def reset_stream_state(self) -> None:
        self._q_input_w_t.resize_(0)
        self._q_hidden_w_t.resize_(0)
        self._q_tau_input_w_t.resize_(0)
        self._q_tau_hidden_w_t.resize_(0)
        self._q_tau_bias.resize_(0)
        self._cached_weight_version.fill_(-1)
        self._cache_ready = False

    def mark_weights_updated(self):
        """
        TR: Ağırlık güncellemesi sonrası cache invalidation işareti.
        EN: Marks cache invalidation after weight updates.
        """
        self._weight_version += 1
        self._cache_ready = False

    def train(self, mode: bool = True):
        super().train(mode)
        # Always invalidate runtime cache on mode transitions.
        self.reset_stream_state()
        if not mode:
            self._weight_version.fill_(self._compute_weight_version())
        return self

    def _ensure_qcache(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.training:
            return
        current_weight_version = self._compute_weight_version()
        if int(self._weight_version.item()) != current_weight_version:
            self._weight_version.fill_(current_weight_version)
            self._cache_ready = False

        if (
            self._cache_ready
            and self._q_input_w_t.numel() > 0
            and self._q_input_w_t.device == device
            and self._q_input_w_t.dtype == dtype
            and int(self._cached_weight_version.item()) == int(self._weight_version.item())
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
            self._set_cache("_q_tau_bias", tb)

            self._cached_weight_version.copy_(self._weight_version)
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
        
        # TR: Eğitim vs Çıkarım Yolu / EN: Training vs Inference Path
        # TR: Eğitim: Python döngüsü (LiquidCell) kullan -> STE gradyanları çalışır.
        # EN: Training: Python loop (LiquidCell) use -> STE gradients work.
        # TR: Çıkarım: JIT döngüsü kullan -> NPU optimizasyonu.
        # EN: Inference: JIT loop use -> NPU optimization.
        if self.training:
            if self.train_impl == "precompute_input":
                out_seq, h = self._train_loop_precompute_input(x, h, dt)
            elif self.train_impl == "packed_pair":
                out_seq, h = self._train_loop_packed_pair(x, h, dt)
            elif self.train_impl == "packed_pair_compile":
                if self.fast_path and x.device.type != "mps":
                    try:
                        out_seq, h = self._train_loop_packed_pair_compiled(x, h, dt)
                    except Exception as exc:
                        warnings.warn(
                            f"Liquid packed-pair compile failed; falling back to eager: {exc}",
                            RuntimeWarning,
                        )
                        out_seq, h = self._train_loop_packed_pair(x, h, dt)
                else:
                    out_seq, h = self._train_loop_packed_pair(x, h, dt)
            elif self.fast_path and x.device.type != "mps":
                try:
                    out_seq, h = self._train_loop_compiled(x, h, dt)
                except Exception as exc:
                    warnings.warn(
                        f"Liquid fast path failed; falling back to eager: {exc}",
                        RuntimeWarning,
                    )
                    out_seq, h = self._train_loop(x, h, dt)
            else:
                out_seq, h = self._train_loop(x, h, dt)
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
        
        # TR: Residual'ı Geri Yükle (Block Liquid için residual eklemez)
        # EN: Restore Residual (Block does NOT add residual for Liquid)
        y = self.norm(out_seq + x)
        if return_state:
            return y, h
        return y

    def _train_loop(self, x: torch.Tensor, h: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H = x.shape
        out_seq = torch.empty(B, T, H, device=x.device, dtype=x.dtype).contiguous()
        for t in range(T):
            h = self.cell(x[:, t, :], h, dt)
            out_seq[:, t, :] = h
        return out_seq, h

    def _train_loop_precompute_input(
        self, x: torch.Tensor, h: torch.Tensor, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H = x.shape
        out_seq = torch.empty(B, T, H, device=x.device, dtype=x.dtype).contiguous()
        val_in_seq = self.cell.input_w(x)
        tau_in_seq = self.cell.tau_input_w(x)

        for t in range(T):
            val_rec = self.cell.hidden_w(h)
            A = torch.tanh(val_in_seq[:, t, :] + val_rec)

            tau_rec = self.cell.tau_hidden_w(h)
            time_decay = F.softplus(tau_in_seq[:, t, :] + tau_rec + self.cell.tau_bias)
            time_decay = torch.clamp(time_decay, min=1e-4, max=5.0)

            decay = torch.exp(torch.clamp(-time_decay * dt, min=-20.0, max=20.0))
            h = A + (h - A) * decay
            out_seq[:, t, :] = h

        return out_seq, h

    @staticmethod
    def _packed_bitlinear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(activation_quant(x), weight_quant(weight), None)

    def _train_loop_packed_pair(
        self, x: torch.Tensor, h: torch.Tensor, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H = x.shape
        out_seq = torch.empty(B, T, H, device=x.device, dtype=x.dtype).contiguous()

        input_pair_w = torch.cat(
            [self.cell.input_w.weight, self.cell.tau_input_w.weight],
            dim=0,
        )
        input_pair = self._packed_bitlinear(x, input_pair_w)
        val_in_seq, tau_in_seq = input_pair.chunk(2, dim=-1)

        hidden_pair_w = torch.cat(
            [self.cell.hidden_w.weight, self.cell.tau_hidden_w.weight],
            dim=0,
        )

        for t in range(T):
            hidden_pair = self._packed_bitlinear(h, hidden_pair_w)
            val_rec, tau_rec = hidden_pair.chunk(2, dim=-1)

            A = torch.tanh(val_in_seq[:, t, :] + val_rec)
            time_decay = F.softplus(tau_in_seq[:, t, :] + tau_rec + self.cell.tau_bias)
            time_decay = torch.clamp(time_decay, min=1e-4, max=5.0)

            decay = torch.exp(torch.clamp(-time_decay * dt, min=-20.0, max=20.0))
            h = A + (h - A) * decay
            out_seq[:, t, :] = h

        return out_seq, h

    def _train_loop_compiled(self, x: torch.Tensor, h: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._compiled_train_loop is None:
            try:
                self._compiled_train_loop = torch.compile(self._train_loop, mode="reduce-overhead")
            except Exception as exc:
                warnings.warn(
                    f"Liquid fast path compile disabled: {exc}",
                    RuntimeWarning,
                )
                self._compiled_train_loop = self._train_loop
        return self._compiled_train_loop(x, h, dt)

    def _train_loop_packed_pair_compiled(
        self, x: torch.Tensor, h: torch.Tensor, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._compiled_packed_train_loop is None:
            try:
                self._compiled_packed_train_loop = torch.compile(
                    self._train_loop_packed_pair,
                    mode="reduce-overhead",
                )
            except Exception as exc:
                warnings.warn(
                    f"Liquid packed-pair compile disabled: {exc}",
                    RuntimeWarning,
                )
                self._compiled_packed_train_loop = self._train_loop_packed_pair
        return self._compiled_packed_train_loop(x, h, dt)

    def load_state_dict(self, *args, **kwargs):
        out = super().load_state_dict(*args, **kwargs)
        self.reset_stream_state()
        self._weight_version.fill_(self._compute_weight_version())
        return out
