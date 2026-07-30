"""Tests for the MoE Switch-style capacity math in layers/moe.py.

The capacity path caps per-expert token assignments at
    capacity = max(1, ceil(capacity_factor * (N * k) / E))
and drops the overflow (assignments beyond capacity for a hot expert), tracking the
dropped fraction in ``last_capacity_overflow_ratio``.

[2026-07-29] This file used to re-implement that formula locally, with a note saying
"if that logic is extracted into a helper, import it here instead of mirroring". It has
been extracted (``layers.moe.moe_capacity``), so these tests now import the SHIPPED
function. Mirroring meant a change to the real capacity path could break it while these
tests stayed green -- which is exactly what happened when the whole capacity block was
rewritten from an O(E) `nonzero()` loop to a vectorized `argsort`/`cumsum` form.

The drop-ORDER semantics (which assignments survive when an expert is over capacity) are
covered by tests/test_property_moe_capacity.py and, end to end against a real MoE
module, by scripts/cfc_moe_tolerance_check.py.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from layers.moe import moe_capacity


def test_capacity_formula_known_values():
    assert moe_capacity(16, 2, 4, 1.25) == 10   # ceil(1.25 * 32 / 4) = 10
    assert moe_capacity(8, 1, 8, 1.0) == 1      # ceil(8 / 8) = 1
    assert moe_capacity(1, 1, 100, 1.25) == 1   # floored at >= 1


def test_capacity_scales_with_factor():
    assert moe_capacity(64, 2, 8, 2.0) > moe_capacity(64, 2, 8, 1.0)


def test_capacity_is_never_below_one():
    # A tiny factor must still leave room for at least one assignment.
    assert moe_capacity(1024, 2, 8, 1e-9) == 1


def test_capacity_covers_balanced_share_at_factor_one():
    n_tokens, k, num_experts = 32, 2, 4
    capacity = moe_capacity(n_tokens, k, num_experts, 1.0)
    assert capacity >= (n_tokens * k) // num_experts


def test_overflow_ratio_matches_dropped_fraction():
    n_tokens, k, num_experts, factor = 32, 2, 4, 1.0
    capacity = moe_capacity(n_tokens, k, num_experts, factor)  # ceil(64/4) = 16
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
    capacity = moe_capacity(n_tokens, k, num_experts, factor)  # ceil(1.25*64/4) = 20
    per_expert = (n_tokens * k) // num_experts                 # 16 each, balanced
    assert per_expert <= capacity


def test_helper_matches_the_formula_documented_in_forward():
    """Guard against the helper and the docstring formula drifting apart."""
    for n, k, e, f in [(16, 2, 4, 1.25), (1024, 2, 8, 1.25), (7, 3, 5, 0.9)]:
        assert moe_capacity(n, k, e, f) == max(1, int(math.ceil(f * (n * k) / max(1, e))))
