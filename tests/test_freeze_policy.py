"""Coverage for apply_freeze_policy (train/trainer_core.py).

This is the freeze half of the DDP find_unused_parameters story (train/train.py sets
find_unused_parameters=True precisely because this flips requires_grad). The name-matching
contract (core frozen; router/shared_expert/liquid/tau/lm_head/tok_embeddings/norm trainable)
is load-bearing and would silently break on a rename, so pin it. Pure CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.trainer_core import apply_freeze_policy  # noqa: E402


class _Stub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core_block = nn.Linear(4, 4)
        self.moe_router = nn.Linear(4, 4)
        self.shared_expert = nn.Linear(4, 4)
        self.liquid_cell = nn.Linear(4, 4)
        self.tau_w = nn.Parameter(torch.zeros(4))
        self.lm_head = nn.Linear(4, 4)
        self.tok_embeddings = nn.Embedding(4, 4)
        self.input_norm = nn.LayerNorm(4)


def test_freeze_policy_freezes_core_keeps_special_trainable():
    m = _Stub()
    apply_freeze_policy(m, freeze_core_layers=True)
    assert all(not p.requires_grad for n, p in m.named_parameters() if "core" in n.lower())
    for sub in ("router", "shared_expert", "liquid", "tau", "lm_head", "tok_embeddings", "norm"):
        params = [p for n, p in m.named_parameters() if sub in n.lower()]
        assert params, f"stub is missing a '{sub}' param"
        assert all(p.requires_grad for p in params), f"'{sub}' params must stay trainable"


def test_freeze_policy_noop_when_disabled():
    m = _Stub()
    before = {n: p.requires_grad for n, p in m.named_parameters()}
    apply_freeze_policy(m, freeze_core_layers=False)
    after = {n: p.requires_grad for n, p in m.named_parameters()}
    assert before == after
