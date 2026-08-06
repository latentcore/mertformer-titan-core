"""Closed-form continuous-time (CfC) "Liquid" cell and mixer.

Mirrors ``vendor/upstream/layers/liquid.py``.

DRIFT FIX #1 -- tau clamp
-------------------------
Canonical ``liquid.py:93-94``::

    time_decay = F.softplus(tau_in + tau_rec + self.tau_bias)
    time_decay = torch.clamp(time_decay, min=1e-4, max=5.0)

``scripts/chess_5080_onefile.py:2597-2598`` dropped the clamp from the eager
cell -- but kept it in the TorchScript eval kernel
(``jit_liquid_loop_cached``, which clamps to ``[1e-4, 5.0]``). So the onefile
trained under one recurrence and evaluated under a different one. Both paths
here clamp, matching the canonical layer, and
``tests/test_arch_parity.py::test_liquid_train_eval_paths_agree`` pins them
together.

DRIFT FIX #3 -- hidden-state threading
--------------------------------------
Canonical ``mertformer_block.py:230`` threads the recurrent state through the
block (``h_init=liquid_state, return_state=True``); the onefile called
``self.liquid(x)`` and discarded the final state. ``forward`` here accepts
``h_init`` and can return the final state, so ``block.py`` can carry it.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import activation_quant, make_linear

TAU_MIN = 1e-4
TAU_MAX = 5.0
DECAY_CLAMP = 20.0


def quant_for_liquid(w: torch.Tensor) -> torch.Tensor:
    """Mirror of ``liquid.jit_quant``: per-row RMS ternary, fp32 math, cast back.

    Locked to ``bitlinear.weight_quant`` (both per-row RMS) -- the canonical
    file carries an explicit parity note about this pairing.
    """
    w_f = w.float()
    scale = torch.sqrt((w_f * w_f).mean(dim=1, keepdim=True)).clamp(min=1e-5)
    w_q = torch.round(w_f / scale).clamp(-1.0, 1.0)
    return (w_q * scale).to(dtype=w.dtype)


class LiquidCell(nn.Module):
    """One CfC step with fully input-dependent time constants."""

    def __init__(self, h: int, use_bitnet: bool = False) -> None:
        super().__init__()
        self.input_w = make_linear(use_bitnet, h, h)
        self.hidden_w = make_linear(use_bitnet, h, h)
        self.tau_input_w = make_linear(use_bitnet, h, h)
        self.tau_hidden_w = make_linear(use_bitnet, h, h)
        # 0.5 init => slower decay => longer temporal memory (canonical choice).
        self.tau_bias = nn.Parameter(torch.ones(1, h) * 0.5)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        val_in = self.input_w(x)
        val_rec = self.hidden_w(h_prev)
        A = torch.tanh(val_in + val_rec)

        tau_in = self.tau_input_w(x)
        tau_rec = self.tau_hidden_w(h_prev)
        time_decay = F.softplus(tau_in + tau_rec + self.tau_bias)
        # DRIFT FIX #1: the onefile omitted this clamp in the eager path.
        time_decay = torch.clamp(time_decay, min=TAU_MIN, max=TAU_MAX)

        decay = torch.exp(torch.clamp(-time_decay * dt, min=-DECAY_CLAMP, max=DECAY_CLAMP))
        return A + (h_prev - A) * decay


def liquid_loop_quantized(
    input_seq: torch.Tensor,
    h_init: torch.Tensor,
    dt: float,
    input_w_q_t: torch.Tensor,
    hidden_w_q_t: torch.Tensor,
    tau_input_w_q_t: torch.Tensor,
    tau_hidden_w_q_t: torch.Tensor,
    tau_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inference kernel over pre-quantized, pre-transposed weights.

    Mirrors ``liquid.jit_liquid_loop_cached`` -- hoisting weight quantization
    out of the per-timestep loop -- with the same tau clamps as ``LiquidCell``.

    FINDING #4 (upstream inconsistency, distinct from the onefile drifts):
    the canonical ``jit_liquid_loop_cached`` multiplies the *raw* activation by
    the quantized weight, while ``LiquidCell`` routes through ``BitLinear``,
    which also applies ``activation_quant``. Under BitNet those two are
    different functions, so the canonical inference kernel evaluates a model
    that was never trained. We apply ``activation_quant`` here so this kernel is
    a pure speed optimization of the eager path rather than a second, different
    recurrence. ``tests/test_arch_parity.py::test_liquid_train_eval_paths_agree``
    is the gate.
    """
    B, T, H = input_seq.shape
    h = h_init
    outs = []
    for t in range(T):
        x_t = activation_quant(input_seq[:, t, :])
        h_q = activation_quant(h)
        val_in = torch.matmul(x_t, input_w_q_t)
        val_rec = torch.matmul(h_q, hidden_w_q_t)
        A = torch.tanh(val_in + val_rec)

        tau_in = torch.matmul(x_t, tau_input_w_q_t)
        tau_rec = torch.matmul(h_q, tau_hidden_w_q_t)
        raw_tau = F.softplus(tau_in + tau_rec + tau_bias)
        time_decay = torch.clamp(raw_tau, min=TAU_MIN, max=TAU_MAX)

        decay = torch.exp(torch.clamp(-time_decay * dt, min=-DECAY_CLAMP, max=DECAY_CLAMP))
        h = A + (h - A) * decay
        outs.append(h)
    return torch.stack(outs, dim=1), h


class LiquidMixer(nn.Module):
    """Residual + LayerNorm wrapper around the CfC recurrence.

    NOTE ON COST: this recurrence is inherently sequential -- ``seq_len`` small
    matmuls per layer per step (76 for a chess board). ``profile.py`` measures
    that cost explicitly; the flag defaults to off and the measured delta is
    reported rather than assumed.
    """

    def __init__(self, h: int, use_bitnet: bool = False, quantized_inference: bool = False) -> None:
        super().__init__()
        self.cell = LiquidCell(h, use_bitnet=use_bitnet)
        self.norm = nn.LayerNorm(h)
        self.hidden_size = int(h)
        # Only meaningful when the cell uses BitLinear; otherwise the eager and
        # quantized kernels are the same math and we always take the eager path.
        self.quantized_inference = bool(quantized_inference and use_bitnet)

    def _quantized_weights(self, device: torch.device, dtype: torch.dtype):
        with torch.no_grad():
            return (
                quant_for_liquid(self.cell.input_w.weight).to(device=device, dtype=dtype).t().contiguous(),
                quant_for_liquid(self.cell.hidden_w.weight).to(device=device, dtype=dtype).t().contiguous(),
                quant_for_liquid(self.cell.tau_input_w.weight).to(device=device, dtype=dtype).t().contiguous(),
                quant_for_liquid(self.cell.tau_hidden_w.weight).to(device=device, dtype=dtype).t().contiguous(),
                self.cell.tau_bias.to(device=device, dtype=dtype).contiguous(),
            )

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
            if h_init.shape != (B, H):
                raise ValueError(f"h_init must be [B,H] = [{B},{H}], got {tuple(h_init.shape)}")
            if h_init.device != x.device or h_init.dtype != x.dtype:
                raise RuntimeError(
                    "h_init device/dtype mismatch with x: "
                    f"h_init={h_init.device}/{h_init.dtype}, x={x.device}/{x.dtype}"
                )
            h = h_init

        if self.quantized_inference and not self.training:
            weights = self._quantized_weights(x.device, x.dtype)
            out_seq, h = liquid_loop_quantized(x, h, dt, *weights)
        else:
            outs = []
            for t in range(T):
                h = self.cell(x[:, t, :], h, dt)
                outs.append(h)
            out_seq = torch.stack(outs, dim=1)

        y = self.norm(out_seq + x)
        if return_state:
            return y, h
        return y
