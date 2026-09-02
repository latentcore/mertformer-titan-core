"""Regression tests for cfg.use_bitnet actually gating BitLinear vs nn.Linear.

[2026-09-02] Before this fix, layers/ffn.py, layers/mla.py, layers/moe.py and
layers/liquid.py called BitLinear(...) UNCONDITIONALLY at construction time --
cfg.use_bitnet (config/config.py, default True) was defined but never read by
any of the four canonical model files. Setting it to False had NO effect on the
model at all, which made the ablations/bitlinear_off ablation structurally
meaningless (confirmed empirically: both arms produced byte-identical loss
curves before this fix). These tests fail on the pre-fix code and pass on the
fixed code -- that is their entire point; they did not exist before because the
bug they catch was never suspected.

Same _cfg_patch pattern as tests/test_mla_regressions.py, for consistency.
"""
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from config.config import cfg  # noqa: E402
from layers.bitlinear import BitLinear, make_linear  # noqa: E402
from layers.ffn import MertFormerFFN  # noqa: E402
from layers.liquid import LiquidCell  # noqa: E402
from layers.mla import GQA  # noqa: E402
from layers.moe import MoE  # noqa: E402


@contextmanager
def _cfg_patch(**overrides):
    _missing = object()
    original = {}
    for key, value in overrides.items():
        original[key] = getattr(cfg, key, _missing)
        setattr(cfg, key, value)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is _missing:
                delattr(cfg, key)
            else:
                setattr(cfg, key, value)


# ---------------------------------------------------------------------------
# Direct unit tests of the helper itself (layers/bitlinear.py::make_linear).
# ---------------------------------------------------------------------------

def test_make_linear_returns_bitlinear_when_true():
    layer = make_linear(True, 8, 16)
    assert isinstance(layer, BitLinear)
    assert layer.in_features == 8 and layer.out_features == 16


def test_make_linear_returns_plain_linear_when_false():
    layer = make_linear(False, 8, 16)
    assert isinstance(layer, nn.Linear)
    assert not isinstance(layer, BitLinear)
    assert layer.in_features == 8 and layer.out_features == 16


# ---------------------------------------------------------------------------
# Per-module regression tests: each of the four call sites the bug lived in.
# ---------------------------------------------------------------------------

_FFN_TINY = dict(hidden_size=32, intermediate_size=64, ffn_dropout=0.0)


def test_ffn_respects_use_bitnet():
    with _cfg_patch(use_bitnet=True, **_FFN_TINY):
        ffn_on = MertFormerFFN()
    assert isinstance(ffn_on.gate_proj, BitLinear)
    assert isinstance(ffn_on.up_proj, BitLinear)
    assert isinstance(ffn_on.down_proj, BitLinear)

    with _cfg_patch(use_bitnet=False, **_FFN_TINY):
        ffn_off = MertFormerFFN()
    assert type(ffn_off.gate_proj) is nn.Linear
    assert type(ffn_off.up_proj) is nn.Linear
    assert type(ffn_off.down_proj) is nn.Linear


_MLA_TINY = dict(
    hidden_size=32, num_heads=4, head_dim=8, num_kv_heads=2,
    max_seq_len=32, rope_dim=8, attention_dropout=0.0,
)


def test_mla_projections_respect_use_bitnet():
    with _cfg_patch(use_bitnet=True, **_MLA_TINY):
        attn_on = GQA()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert isinstance(getattr(attn_on, name), BitLinear), name

    with _cfg_patch(use_bitnet=False, **_MLA_TINY):
        attn_off = GQA()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert type(getattr(attn_off, name)) is nn.Linear, name


_MOE_TINY = dict(hidden_size=32, num_experts=2, num_experts_per_tok=1, moe_intermediate=64)


def test_moe_bitswiglu_and_router_respect_use_bitnet():
    with _cfg_patch(use_bitnet=True, **_MOE_TINY):
        moe_on = MoE()
    assert isinstance(moe_on.experts[0].gate_proj, BitLinear)
    assert isinstance(moe_on.shared_expert.gate_proj, BitLinear)
    assert isinstance(moe_on.router.main_proj, BitLinear)
    assert isinstance(moe_on.router.fluid_gate, BitLinear)

    with _cfg_patch(use_bitnet=False, **_MOE_TINY):
        moe_off = MoE()
    assert type(moe_off.experts[0].gate_proj) is nn.Linear
    assert type(moe_off.shared_expert.gate_proj) is nn.Linear
    assert type(moe_off.router.main_proj) is nn.Linear
    assert type(moe_off.router.fluid_gate) is nn.Linear


def test_liquidcell_projections_respect_use_bitnet():
    with _cfg_patch(use_bitnet=True):
        cell_on = LiquidCell(16)
    for name in ("input_w", "hidden_w", "tau_input_w", "tau_hidden_w"):
        assert isinstance(getattr(cell_on, name), BitLinear), name

    with _cfg_patch(use_bitnet=False):
        cell_off = LiquidCell(16)
    for name in ("input_w", "hidden_w", "tau_input_w", "tau_hidden_w"):
        assert type(getattr(cell_off, name)) is nn.Linear, name


# ---------------------------------------------------------------------------
# Forward-pass smoke: both configs must actually run end-to-end, not just
# construct with the right module types.
# ---------------------------------------------------------------------------

def test_ffn_forward_runs_and_is_finite_in_both_configs():
    x = torch.randn(2, 3, 32)
    with _cfg_patch(use_bitnet=True, **_FFN_TINY):
        out_on = MertFormerFFN()(x)
    with _cfg_patch(use_bitnet=False, **_FFN_TINY):
        out_off = MertFormerFFN()(x)
    assert out_on.shape == (2, 3, 32)
    assert out_off.shape == (2, 3, 32)
    assert torch.isfinite(out_on).all()
    assert torch.isfinite(out_off).all()
