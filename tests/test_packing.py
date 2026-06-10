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
