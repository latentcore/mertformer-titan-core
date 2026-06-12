"""Coverage for the kd_loss_safe(mask=...) dense path (train/trainer_eval.py).

The mask ties KD supervision to the CE -100 (pad==eos) mask. Until now only an AST test
checked that kd_loss_safe is *called* with a mask; this exercises the runtime behavior:
masked-mean, all-False -> zero scalar, wrong-shape -> ValueError. Pure CPU, deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.trainer_eval import kd_loss_safe  # noqa: E402


def _logits(seed: int = 0, b: int = 1, t: int = 4, v: int = 8) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(b, t, v)


def test_kd_mask_all_false_returns_zero_scalar():
    out = kd_loss_safe(_logits(), _logits(seed=1), temp=2.0, mask=torch.zeros(1, 4, dtype=torch.bool))
    assert out.dim() == 0
    assert float(out) == 0.0


def test_kd_mask_wrong_shape_raises():
    with pytest.raises(ValueError):
        kd_loss_safe(_logits(), _logits(seed=1), temp=2.0, mask=torch.ones(1, 3, dtype=torch.bool))


def test_kd_mask_selects_only_unmasked_tokens():
    s, te = _logits(), _logits(seed=1)
    half = torch.tensor([[True, False, True, False]])
    out = kd_loss_safe(s, te, temp=2.0, mask=half)
    assert out.dim() == 0
    assert torch.isfinite(out)
    assert float(out) > 0.0  # the two kept tokens carry real KD signal
