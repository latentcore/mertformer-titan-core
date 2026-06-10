"""H2: when pad_id == eos_id, the loss mask must come from true_len (-100), so the
EOS separator is supervised while only trailing pad is ignored."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

from train import train as T  # noqa: E402


class _Tok:
    def __init__(self, pad_id, eos_id):
        self.pad_token_id = pad_id
        self.eos_token_id = eos_id

    def __call__(self, text, add_special_tokens=True, truncation=True, max_length=None):
        ids = [ord(c) % 40 + 5 for c in text]
        if max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_eos_is_supervised_when_pad_equals_eos():
    # pad_id == eos_id == 2  (the Llama/online-lane trap)
    tok = _Tok(pad_id=2, eos_id=2)
    input_ids, labels = T._encode_with_eos_labels(tok, "abcd", max_len=12, pad_id=2, eos_id=2)
    # Exactly one EOS appended at the end of the real tokens, and it is a LABEL
    # (not -100), i.e. supervised.
    eos_positions = (input_ids == 2).nonzero(as_tuple=True)[0].tolist()
    assert eos_positions, "no EOS token present"
    first_eos = eos_positions[0]
    assert labels[first_eos].item() == 2, "EOS label was masked (the H2 bug)"


def test_trailing_pad_is_ignored():
    tok = _Tok(pad_id=2, eos_id=2)
    input_ids, labels = T._encode_with_eos_labels(tok, "ab", max_len=12, pad_id=2, eos_id=2)
    true_len = int((labels != -100).sum().item())
    # Everything after true_len is -100 (ignored).
    assert (labels[true_len:] == -100).all()
    # Real-token count counts the supervised tokens only (incl. the EOS).
    assert true_len >= 1


def test_distinct_pad_unchanged():
    tok = _Tok(pad_id=0, eos_id=2)
    input_ids, labels = T._encode_with_eos_labels(tok, "abc", max_len=12, pad_id=0, eos_id=2)
    # Input pads with pad_id=0; labels pad with -100.
    assert input_ids[-1].item() == 0
    assert labels[-1].item() == -100
