"""Pins the DROP-ORDER semantics of the vectorized MoE capacity mask.

[2026-07-29] ``MoE.forward``'s capacity block was rewritten from an O(E) loop of
``(topk_idx == e).nonzero()`` calls into a vectorized ``argsort``/``cumsum`` form, to
remove ~50+ device->host synchronisations per micro-batch (``torch.nonzero`` must sync to
size its output; so must boolean-mask indexing and every ``.item()``).

The formula (how many assignments an expert may keep) is covered by
tests/test_moe_capacity.py. What THIS file pins is the part that is easy to get subtly
wrong and impossible to notice from a loss curve: **which** assignments survive when an
expert is over capacity.

``torch.nonzero`` returns hits in ROW-MAJOR order, so the old code kept each expert's
first ``capacity`` assignments ordered by (row, then column). The replacement relies on
two facts: ``topk_idx.reshape(-1)`` yields flat index ``row * k + col`` (that same
row-major order), and a STABLE ``argsort`` groups by expert while preserving input order
inside each group. These tests assert the vectorized mask is bit-identical to a direct
``nonzero``-based reference on adversarial routing patterns -- including a single hot
expert, ties, and every-token-dropped rows.

End-to-end numeric equivalence against a real MoE module is separately gated by
scripts/cfc_moe_tolerance_check.py (max_diff must stay 0.0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")

from layers.moe import moe_capacity  # noqa: E402


def _reference_mask(topk_idx: "torch.Tensor", num_experts: int, capacity: int) -> "torch.Tensor":
    """The ORIGINAL nonzero()-loop semantics, kept as an independent reference."""
    mask = torch.ones_like(topk_idx, dtype=torch.bool)
    for expert in range(num_experts):
        hits = (topk_idx == expert).nonzero(as_tuple=False)
        if hits.size(0) > capacity:
            overflow = hits[capacity:]
            mask[overflow[:, 0], overflow[:, 1]] = False
    return mask


def _vectorized_mask(topk_idx: "torch.Tensor", num_experts: int, capacity: int) -> "torch.Tensor":
    """The shipped vectorized form, mirrored here to isolate it from module state."""
    flat_e = topk_idx.reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    slot_counts = torch.zeros(num_experts, device=flat_e.device, dtype=torch.long)
    slot_counts.scatter_add_(0, flat_e, torch.ones_like(flat_e))
    group_starts = torch.cumsum(slot_counts, 0) - slot_counts
    rank_in_sorted = torch.empty_like(order)
    rank_in_sorted[order] = torch.arange(order.numel(), device=order.device)
    within_group = rank_in_sorted - group_starts[flat_e]
    return (within_group < capacity).reshape(topk_idx.shape)


def _assert_equivalent(topk_idx, num_experts, capacity):
    ref = _reference_mask(topk_idx, num_experts, capacity)
    vec = _vectorized_mask(topk_idx, num_experts, capacity)
    assert torch.equal(ref, vec), (
        f"drop order diverged\nidx=\n{topk_idx}\ncapacity={capacity}\n"
        f"reference=\n{ref}\nvectorized=\n{vec}"
    )


def test_single_hot_expert_keeps_the_first_capacity_in_row_major_order():
    """Every token picks expert 0; only the first `capacity` (row-major) survive."""
    topk_idx = torch.zeros(8, 2, dtype=torch.long)
    _assert_equivalent(topk_idx, num_experts=4, capacity=5)
    vec = _vectorized_mask(topk_idx, 4, 5)
    # Flat row-major positions 0..4 kept, 5..15 dropped.
    assert vec.reshape(-1).tolist() == [True] * 5 + [False] * 11


def test_no_drops_when_under_capacity():
    topk_idx = torch.tensor([[0, 1], [2, 3], [0, 1]], dtype=torch.long)
    _assert_equivalent(topk_idx, num_experts=4, capacity=100)
    assert _vectorized_mask(topk_idx, 4, 100).all()


def test_capacity_one_keeps_exactly_the_first_hit_per_expert():
    topk_idx = torch.tensor([[0, 1], [0, 1], [2, 2]], dtype=torch.long)
    _assert_equivalent(topk_idx, num_experts=3, capacity=1)
    vec = _vectorized_mask(topk_idx, 3, 1)
    # expert 0 first at (0,0); expert 1 first at (0,1); expert 2 first at (2,0).
    assert vec.tolist() == [[True, True], [False, False], [True, False]]


def test_second_column_hit_can_outrank_a_later_rows_first_column():
    """Row-major means (0,1) precedes (1,0) -- a column-major reading would flip this."""
    topk_idx = torch.tensor([[5, 7], [7, 5]], dtype=torch.long)
    _assert_equivalent(topk_idx, num_experts=8, capacity=1)
    vec = _vectorized_mask(topk_idx, 8, 1)
    # expert 7's first hit is (0,1), not (1,0).
    assert vec.tolist() == [[True, True], [False, False]]


def test_unrouted_experts_do_not_shift_ranks():
    """Experts with zero hits must not consume group offsets."""
    topk_idx = torch.tensor([[9, 9], [9, 9]], dtype=torch.long)
    _assert_equivalent(topk_idx, num_experts=16, capacity=3)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_randomized_routing_matches_reference(seed):
    """Randomised sweep over shapes, expert counts and capacities."""
    generator = torch.Generator().manual_seed(seed)
    num_experts = int(torch.randint(1, 9, (1,), generator=generator).item())
    n_tokens = int(torch.randint(1, 40, (1,), generator=generator).item())
    top_k = int(torch.randint(1, min(4, num_experts) + 1, (1,), generator=generator).item())
    topk_idx = torch.randint(0, num_experts, (n_tokens, top_k), generator=generator)
    for factor in (0.25, 1.0, 1.25, 4.0):
        capacity = moe_capacity(n_tokens, top_k, num_experts, factor)
        _assert_equivalent(topk_idx, num_experts, capacity)


def test_every_slot_dropped_rows_are_detectable():
    """A row can lose all its choices -- forward() then forces its top-1 back on.

    This pins the precondition for that fallback (such rows exist and are detected),
    which forward() handles with a torch.where instead of the old `.any()` sync.
    """
    topk_idx = torch.zeros(4, 2, dtype=torch.long)
    vec = _vectorized_mask(topk_idx, num_experts=2, capacity=2)
    fully_dropped = ~vec.any(dim=-1)
    assert fully_dropped.tolist() == [False, True, True, True]
