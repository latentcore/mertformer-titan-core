"""Regression test for B1: gradient checkpointing + MoE must survive backward.

This is the exact path tests/test_comprehensive.py disabled (use_gradient_checkpointing
=False). With GC=True AND MoE=True, the pre-fix code raised CheckpointError on the
first backward because MoE.forward mutates buffers under non-reentrant recompute.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from config.config import cfg  # noqa: E402
from model.transformers import MertFormer  # noqa: E402

_KEYS = [
    "hidden_size", "intermediate_size", "num_experts", "num_experts_per_tok",
    "active_experts", "num_heads", "num_kv_heads", "head_dim", "num_layers",
    "num_hidden_layers", "vocab_size", "max_seq_len", "use_moe", "use_liquid",
    "use_qinn", "liquid_layers_idx", "moe_every_n_layers", "use_gradient_checkpointing",
    "attention_dropout", "dropout",
]


@pytest.fixture()
def gc_moe_cfg():
    orig = {k: getattr(cfg, k, None) for k in _KEYS}
    cfg.hidden_size = 128
    cfg.intermediate_size = 256
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
    cfg.active_experts = 2
    cfg.num_heads = 4
    cfg.num_kv_heads = 4
    cfg.head_dim = 32
    cfg.num_layers = 4
    cfg.num_hidden_layers = 4
    cfg.vocab_size = 1000
    cfg.max_seq_len = 64
    cfg.use_moe = True
    cfg.use_liquid = True
    cfg.use_qinn = False
    cfg.liquid_layers_idx = [1]
    cfg.moe_every_n_layers = 2  # multiple MoE layers stacked
    cfg.use_gradient_checkpointing = True  # the path that used to crash
    if hasattr(cfg, "attention_dropout"):
        cfg.attention_dropout = 0.1
    if hasattr(cfg, "dropout"):
        cfg.dropout = 0.1
    yield cfg
    for k, v in orig.items():
        setattr(cfg, k, v)


def test_gc_true_moe_true_real_backward(gc_moe_cfg):
    model = MertFormer().to("cpu")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.zero_grad()

    input_ids = torch.randint(0, gc_moe_cfg.vocab_size, (2, 16))
    logits, aux_loss, _ = model(input_ids)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, gc_moe_cfg.vocab_size), shift_labels.view(-1)
    ) + 0.01 * aux_loss

    # The assertion: this must NOT raise CheckpointError.
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradients"
