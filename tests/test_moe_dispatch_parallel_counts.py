"""Regression test for the MoE._dispatch_parallel per-expert count computation
(layers/moe.py) — MPS-portability fix.

``_dispatch_parallel`` used ``torch.bincount`` to count how many tokens sorted
into each expert. The telemetry path a few lines above it in the same file was
already switched to ``scatter_add_`` with an explicit "MPS Safe Bincount"
comment, but the dispatch path was missed, leaving the file internally
inconsistent. ``requirements.txt`` pins ``torch>=2.0`` with no upper bound, and
``torch.bincount`` has historically lacked MPS coverage on older torch
releases (it happens to work on the torch installed in this environment, so
this is a portability/consistency fix, not a reproduction of an active local
crash). This test locks the counting math itself and confirms the real
dispatch path still runs and produces correct per-expert routing after the
scatter_add_ rewrite.
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
    "attention_dropout", "dropout", "moe_dispatch_mode", "moe_capacity_enforce",
]


@pytest.fixture()
def parallel_moe_cfg():
    orig = {k: getattr(cfg, k, None) for k in _KEYS}
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_experts = 6
    cfg.num_experts_per_tok = 2
    cfg.active_experts = 2
    cfg.num_heads = 4
    cfg.num_kv_heads = 4
    cfg.head_dim = 16
    cfg.num_layers = 2
    cfg.num_hidden_layers = 2
    cfg.vocab_size = 200
    cfg.max_seq_len = 32
    cfg.use_moe = True
    cfg.use_liquid = False
    cfg.use_qinn = False
    cfg.liquid_layers_idx = []
    cfg.moe_every_n_layers = 1
    cfg.use_gradient_checkpointing = False
    cfg.attention_dropout = 0.0
    cfg.dropout = 0.0
    cfg.moe_dispatch_mode = "parallel"
    cfg.moe_capacity_enforce = True
    yield cfg
    for k, v in orig.items():
        setattr(cfg, k, v)


def test_scatter_add_counts_match_bincount_reference():
    """The replacement counting logic must be numerically identical to bincount."""
    torch.manual_seed(0)
    num_experts = 6
    expert_sorted = torch.randint(0, num_experts, (500,))

    reference = torch.bincount(expert_sorted, minlength=num_experts)

    replacement = torch.zeros(num_experts, dtype=torch.long)
    replacement.scatter_add_(0, expert_sorted, torch.ones_like(expert_sorted))

    assert torch.equal(reference, replacement)


def test_scatter_add_counts_handles_missing_experts():
    """An expert with zero routed tokens must still read back as count 0, not omitted."""
    num_experts = 5
    expert_sorted = torch.tensor([0, 0, 2, 2, 2])  # experts 1, 3, 4 get nothing

    replacement = torch.zeros(num_experts, dtype=torch.long)
    replacement.scatter_add_(0, expert_sorted, torch.ones_like(expert_sorted))

    assert replacement.tolist() == [2, 0, 3, 0, 0]


def test_dispatch_parallel_real_forward_backward(parallel_moe_cfg):
    """The actual MoE forward/backward through _dispatch_parallel must run clean
    post-fix, on whichever device is available (MPS on this machine)."""
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = MertFormer().to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.zero_grad()

    input_ids = torch.randint(0, parallel_moe_cfg.vocab_size, (3, 20), device=device)
    logits, aux_loss, _ = model(input_ids)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, parallel_moe_cfg.vocab_size), shift_labels.view(-1)
    ) + 0.01 * aux_loss
    assert torch.isfinite(loss)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradients"


def test_dispatch_parallel_and_sequential_route_the_same_tokens(parallel_moe_cfg):
    """Parallel and sequential dispatch must select the same expert for each token
    (the counting rewrite must not silently change routing assignment)."""
    parallel_moe_cfg.moe_dispatch_mode = "parallel"
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(7)
    model_parallel = MertFormer().to(device)
    model_parallel.eval()

    parallel_moe_cfg.moe_dispatch_mode = "sequential"
    torch.manual_seed(7)
    model_sequential = MertFormer().to(device)
    model_sequential.eval()

    input_ids = torch.randint(0, parallel_moe_cfg.vocab_size, (2, 12), device=device)
    with torch.no_grad():
        logits_p, _, _ = model_parallel(input_ids)
        logits_s, _, _ = model_sequential(input_ids)

    assert torch.allclose(logits_p, logits_s, atol=1e-4), (
        "parallel vs sequential dispatch diverged beyond floating-point tolerance "
        f"(max diff={(logits_p - logits_s).abs().max().item():.3e})"
    )
