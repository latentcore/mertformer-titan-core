"""Characterization test for the MoE Switch-style capacity math (layers/moe.py:704-710).

The capacity path caps per-expert token assignments at
    capacity = max(1, ceil(capacity_factor * (N * k) / E))
and drops the overflow (assignments beyond capacity for a hot expert), tracking the
dropped fraction in ``last_capacity_overflow_ratio``. Constructing the real MoE requires
the full config singleton (hardware-scale) and mutating that singleton makes the test
order-dependent, so this locks the enforcement math on controlled counts instead. It
mirrors the inline logic; if that logic is extracted into a helper, import it here.
"""
import math

import torch


def _capacity(n_tokens: int, k: int, num_experts: int, factor: float) -> int:
    # Verbatim mirror of the layers/moe.py capacity formula.
    return max(1, int(math.ceil(factor * (n_tokens * k) / max(1, num_experts))))


def test_capacity_formula_known_values():
    assert _capacity(16, 2, 4, 1.25) == 10   # ceil(1.25 * 32 / 4) = 10
    assert _capacity(8, 1, 8, 1.0) == 1      # ceil(8 / 8) = 1
    assert _capacity(1, 1, 100, 1.25) == 1   # floored at >= 1


def test_capacity_scales_with_factor():
    assert _capacity(64, 2, 8, 2.0) > _capacity(64, 2, 8, 1.0)


def test_overflow_ratio_matches_dropped_fraction():
    n_tokens, k, num_experts, factor = 32, 2, 4, 1.0
    capacity = _capacity(n_tokens, k, num_experts, factor)  # ceil(64/4) = 16
    assert capacity == 16
    # Force a hot expert: every assignment routes to expert 0.
    assignments = torch.zeros(n_tokens * k, dtype=torch.long)
    total = assignments.numel()
    dropped = 0
    for e in range(num_experts):
        hits = (assignments == e).nonzero(as_tuple=True)[0]
        if hits.numel() > capacity:
            dropped += hits.numel() - capacity
    overflow_ratio = dropped / total
    assert abs(overflow_ratio - 0.75) < 1e-9   # 64 hits, cap 16 -> drop 48 -> 48/64


def test_no_overflow_when_balanced():
    n_tokens, k, num_experts, factor = 32, 2, 4, 1.25
    capacity = _capacity(n_tokens, k, num_experts, factor)  # ceil(1.25*64/4) = 20
    per_expert = (n_tokens * k) // num_experts               # 16 each, balanced
    assert per_expert <= capacity
