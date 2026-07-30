"""Property-based tests for the MoE capacity formula (I.7.4 · hypothesis gate).

Complements the fixed-value cases in tests/test_moe_capacity.py with hypothesis-
generated inputs, pinning the invariants that must hold for ALL valid
``(tokens, k, experts, factor)``: capacity is always >= 1, monotonically
non-decreasing in the capacity factor, and covers the perfectly-balanced per-expert
share at factor 1.0. Pure arithmetic (no torch), so it is fast and deterministic
under hypothesis's own seeding.

[2026-07-29] These properties now hold over the SHIPPED ``layers.moe.moe_capacity``
helper, not a local copy of the formula. Keep it that way: while the formula was
mirrored here, the entire capacity block in ``MoE.forward`` could be rewritten and both
capacity test files would stay green regardless.
"""
import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# [2026-07-29] Now imports the SHIPPED helper instead of mirroring the formula locally,
# per this file's own standing note ("Mirrors the inline formula in layers/moe.py; if
# that logic is extracted into a helper, import it here instead of mirroring").
# layers.moe.moe_capacity is that extraction. While mirrored, these properties held
# over a private copy and could not detect a change to the real capacity path.
from layers.moe import moe_capacity as _capacity


@settings(max_examples=200)
@given(
    n=st.integers(min_value=0, max_value=8192),
    k=st.integers(min_value=1, max_value=8),
    e=st.integers(min_value=1, max_value=64),
    f=st.floats(min_value=0.1, max_value=4.0, allow_nan=False, allow_infinity=False),
)
def test_capacity_is_always_at_least_one(n, k, e, f):
    assert _capacity(n, k, e, f) >= 1


@settings(max_examples=200)
@given(
    n=st.integers(min_value=1, max_value=8192),
    k=st.integers(min_value=1, max_value=8),
    e=st.integers(min_value=1, max_value=64),
    f1=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    df=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_capacity_is_monotonic_in_factor(n, k, e, f1, df):
    assert _capacity(n, k, e, f1) <= _capacity(n, k, e, f1 + df)


@settings(max_examples=200)
@given(
    n=st.integers(min_value=1, max_value=8192),
    k=st.integers(min_value=1, max_value=8),
    e=st.integers(min_value=1, max_value=64),
)
def test_capacity_covers_balanced_share_at_factor_one(n, k, e):
    # At factor 1.0 the per-expert cap must cover the perfectly-balanced assignment.
    assert _capacity(n, k, e, 1.0) >= (n * k) // e
