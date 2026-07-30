"""Tests for train/packing.py — the shared teacher/student packer + identity (B2)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train import packing  # noqa: E402


class _StubTokenizer:
    """Deterministic char-based tokenizer (no HF, no network)."""
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=True, truncation=True, max_length=None):
        ids = [ord(c) % 50 + 3 for c in text]
        if max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def _rows(*texts):
    return [(i, t) for i, t in enumerate(texts)]


def test_packed_stream_is_deterministic():
    tok = _StubTokenizer()
    rows = ["alpha", "beta", "gamma delta"]
    a = list(packing.iter_packed_sequences(_rows(*rows), tok, 8, 2, 0))
    b = list(packing.iter_packed_sequences(_rows(*rows), tok, 8, 2, 0))
    assert [s["input_ids"] for s in a] == [s["input_ids"] for s in b]
    assert [s["identity"] for s in a] == [s["identity"] for s in b]


def test_eos_separates_rows_and_pads_last():
    tok = _StubTokenizer()
    seqs = list(packing.iter_packed_sequences(_rows("ab", "cd"), tok, 8, 2, 0))
    # Single packed sequence: ab<eos>cd<eos> then pad to 8.
    s = seqs[0]
    assert len(s["input_ids"]) == 8
    assert 2 in s["input_ids"]  # EOS separator present
    assert s["input_ids"][s["true_len"]:] == [0] * (8 - s["true_len"])  # trailing pad


def test_oversized_row_truncated_to_max_seq_len():
    tok = _StubTokenizer()
    seqs = list(packing.iter_packed_sequences(_rows("abcdefghijklmnop"), tok, 6, 2, 0))
    assert all(len(s["input_ids"]) == 6 for s in seqs)
    assert seqs[0]["true_len"] <= 6


def test_sequence_identity_ignores_trailing_pad():
    a = packing.sequence_identity([5, 6, 7, 0, 0, 0], true_len=3)
    b = packing.sequence_identity([5, 6, 7, 9, 9, 9], true_len=3)
    assert a == b  # only first true_len tokens are hashed


def test_assert_sequence_identity_hardfails_on_mismatch():
    ident = packing.sequence_identity([5, 6, 7, 0, 0], true_len=3)
    packing.assert_sequence_identity([5, 6, 7, 0, 0], 3, ident)  # ok
    with pytest.raises(packing.LogitAlignmentError):
        packing.assert_sequence_identity([5, 6, 8, 0, 0], 3, ident)  # one token flipped
    with pytest.raises(packing.LogitAlignmentError):
        packing.assert_sequence_identity([5, 6, 7, 0, 0], 3, None)  # missing identity


def test_extract_row_text_unifies_to_text_only():
    assert packing.extract_row_text({"text": " hi "}) == "hi"
    # content/instruction-only rows collapse to empty (precompute == train skip).
    assert packing.extract_row_text({"content": "x"}) == ""
    assert packing.extract_row_text({"instruction": "y"}) == ""


def test_consumed_through_advances_for_resume():
    tok = _StubTokenizer()
    seqs = list(packing.iter_packed_sequences(_rows("aa", "bb", "cc", "dd"), tok, 6, 2, 0))
    # consumed_through is monotonically non-decreasing and bounded by row indices.
    cons = [s["consumed_through"] for s in seqs]
    assert cons == sorted(cons)
    assert cons[-1] <= 3


# --- Y-5 [2026-07-29]: oversized rows must not desynchronise the resume counter -------
#
# The `pending` window retires a row when `end_cum <= emitted_tokens`, which is only a
# valid test while `len(buf) == cum - emitted_tokens` holds. The oversized branch used to
# do `cum += len(piece)` for a row it emitted TRUNCATED (and whose tail it discarded),
# leaking exactly 1 token -- the EOS -- per oversized row into a permanent offset. The
# tests below pin the observable consequences, all of which were wrong before the fix.

def _mixed_rows(n_oversized: int, n_normal: int, wide: str = "x" * 99):
    """`n_oversized` over-length rows first, then `n_normal` short rows."""
    rows = [(i, wide) for i in range(n_oversized)]
    rows += [(n_oversized + i, "ab") for i in range(n_normal)]
    return rows


def _worst_resume_lag(rows, max_seq_len=8):
    """Largest gap between a sequence's last contributing row and its resume point.

    NOTE for future maintainers: do NOT probe this with ``seqs[-1]``. The trailing
    ``if buf:`` flush sets consumed_through from ``pending[-1][0]`` unconditionally, so
    the final sequence reports a perfect resume point even when every mid-stream one is
    stale. Measured with the pre-fix code, the final-sequence lag was 0 across
    1..400 oversized rows while the mid-stream lag ran to 134 rows.
    """
    tok = _StubTokenizer()
    seqs = list(packing.iter_packed_sequences(iter(rows), tok, max_seq_len, 2, 0))
    assert seqs, "packer produced nothing"
    return max(s["row_span"][1] - s["consumed_through"] for s in seqs)


def test_resume_point_never_trails_its_own_row_span():
    """A sequence's resume point must stay adjacent to the rows it just consumed.

    One row of lag is structural: the row straddling the sequence boundary is genuinely
    not finished yet. Anything beyond that is leaked `cum`. Pre-fix this test failed from
    5 oversized rows upward (lag 3), reaching 134 at 400.
    """
    for n_oversized in (0, 1, 5, 30, 100, 400):
        lag = _worst_resume_lag(_mixed_rows(n_oversized, 300))
        assert lag <= 1, (
            f"{n_oversized} oversized rows left the resume point {lag} rows behind "
            f"the sequence's own row_span"
        )


def test_resume_lag_does_not_grow_with_oversized_row_count():
    """The lag must be INDEPENDENT of how many oversized rows preceded it.

    Pre-fix the same sweep produced [1, 1, 3, 11, 34, 134] -- linear in the oversized
    count, hence unbounded over a real multi-billion-token corpus. Post-fix it is flat.
    """
    lags = [_worst_resume_lag(_mixed_rows(n, 300))
            for n in (0, 1, 5, 30, 100, 400)]
    assert len(set(lags)) == 1, f"resume lag scales with oversized rows: {lags}"


def test_pending_window_stays_bounded_across_oversized_rows():
    """`pending` is internal, so probe it through its observable effects.

    An unretired `pending` entry is exactly what holds consumed_through back, so a
    bounded lag over a long mixed stream proves the window is being drained. The old
    code let `pending` grow linearly with the oversized count, which also made
    ``_consumed_through()`` -- an O(len(pending)) scan run once per emitted sequence --
    quadratic overall (0.6 ms -> 32 ms over a 0..6400 oversized sweep).
    """
    tok = _StubTokenizer()
    rows = _mixed_rows(200, 600)
    seqs = list(packing.iter_packed_sequences(iter(rows), tok, 8, 2, 0))

    cons = [s["consumed_through"] for s in seqs]
    assert cons == sorted(cons), "consumed_through must stay monotone"
    assert _worst_resume_lag(rows) <= 1


def test_oversized_row_emits_exactly_max_seq_len_and_discards_its_tail():
    """Pins WHY cum must not advance by len(piece): the tail is never emitted."""
    tok = _StubTokenizer()
    max_seq_len = 8
    seqs = list(packing.iter_packed_sequences(_rows("x" * 99), tok, max_seq_len, 2, 0))
    assert len(seqs) == 1
    assert seqs[0]["true_len"] == max_seq_len
    assert len(seqs[0]["input_ids"]) == max_seq_len
    assert seqs[0]["consumed_through"] == 0


def test_oversized_fix_preserves_the_token_stream():
    """The fix touches resume bookkeeping only -- packed tokens must be untouched.

    The packer is contractually a pure function of (rows, max_seq_len, eos, pad,
    tokenizer) because the teacher and student streams must stay byte-identical; a
    change to the emitted tokens would invalidate every precomputed teacher logit.

    The constants below were captured by RUNNING THE PRE-FIX implementation on this
    exact input, so this test fails if the resume-counter fix ever perturbs the tokens.
    """
    tok = _StubTokenizer()
    rows = [(i, "x" * 99 if i % 7 == 0 else "ab") for i in range(200)]
    seqs = list(packing.iter_packed_sequences(iter(rows), tok, 8, 2, 0))

    assert len(seqs) == 115
    assert [s["true_len"] for s in seqs] == [8, 8, 8, 2] * 28 + [8, 8, 1]
    # Byte-level anchor: the packed-token hash of the first sequence.
    assert seqs[0]["identity"] == {"len": 8, "hash": "9578f9bc5668648abb3560c7e539d0a4"}

    # Determinism, not merely stability: a second pass must reproduce it exactly.
    again = list(packing.iter_packed_sequences(iter(rows), tok, 8, 2, 0))
    assert [s["identity"] for s in again] == [s["identity"] for s in seqs]
    assert [s["input_ids"] for s in again] == [s["input_ids"] for s in seqs]
