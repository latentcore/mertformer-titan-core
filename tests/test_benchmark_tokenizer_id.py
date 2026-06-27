"""Checkpoint-bound benchmark/eval steps must take their tokenizer from the
checkpoint identity (no silent teacher fallback).

Covers the post_train_autorun checkpoint_required decode paths:
  - scripts/benchmarks_internal.py (HumanEval/MBPP)
  - scripts/golden_score.py        (autorun 'golden_eval' step)
  - scripts/golden_eval.py         (golden sample decode)

Each test stops at the tokenizer-resolution boundary so no full model is built.
"""

from __future__ import annotations

import importlib
import json
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from utils import tokenizer_resolver

benchmarks_internal = importlib.import_module("scripts.benchmarks_internal")
golden_eval = importlib.import_module("scripts.golden_eval")
golden_score = importlib.import_module("scripts.golden_score")

IDENTITY = {
    "name_or_path": "data/tokenizer/tr",
    "vocab_size": 128000,
    "tokenizer_class": "BertTokenizer",
    "use_tr_tokenizer": True,
}


class _Stop(Exception):
    """Sentinel: raised by the spy to halt right after the identity is read."""


def _write_ckpt(path, with_id: bool) -> None:
    state = {"model": {}}
    if with_id:
        state["tokenizer_id"] = IDENTITY
    torch.save(state, path)


def _spy(monkeypatch):
    captured: dict = {}

    def spy(identity):
        captured["id"] = identity
        raise _Stop()

    monkeypatch.setattr(tokenizer_resolver, "load_tokenizer_from_identity", spy)
    return captured


# ---------------------------------------------------------------------------
# Positive: the decode path passes checkpoint["tokenizer_id"] to the resolver.
# ---------------------------------------------------------------------------
def test_golden_eval_uses_checkpoint_tokenizer_id(monkeypatch, tmp_path) -> None:
    ckpt = tmp_path / "c.pt"
    _write_ckpt(ckpt, with_id=True)
    captured = _spy(monkeypatch)

    with pytest.raises(_Stop):
        golden_eval.run_model([{"prompt": "merhaba"}], str(ckpt))

    assert captured["id"] == IDENTITY


def test_golden_score_uses_checkpoint_tokenizer_id(monkeypatch, tmp_path) -> None:
    assertions = tmp_path / "a.jsonl"
    assertions.write_text(
        json.dumps({"id": 1, "prompt": "merhaba", "assertions": []}) + "\n",
        encoding="utf-8",
    )
    ckpt = tmp_path / "c.pt"
    _write_ckpt(ckpt, with_id=True)
    captured = _spy(monkeypatch)

    with pytest.raises(_Stop):
        golden_score.run_model(assertions, ckpt, tmp_path / "out.jsonl", 8, 1)

    assert captured["id"] == IDENTITY


def test_benchmarks_internal_uses_checkpoint_tokenizer_id(monkeypatch, tmp_path) -> None:
    ckpt = tmp_path / "c.pt"
    _write_ckpt(ckpt, with_id=True)
    captured = _spy(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["prog", "--run", "--ckpt", str(ckpt)])

    with pytest.raises(_Stop):
        benchmarks_internal.main()

    assert captured["id"] == IDENTITY


# ---------------------------------------------------------------------------
# Negative: a checkpoint with no tokenizer_id is a hard error (no fallback).
# Uses the real resolver, which raises before any model build / network call.
# ---------------------------------------------------------------------------
def test_golden_eval_missing_tokenizer_id_errors(tmp_path) -> None:
    ckpt = tmp_path / "c.pt"
    _write_ckpt(ckpt, with_id=False)
    with pytest.raises(ValueError, match="tokenizer_id"):
        golden_eval.run_model([{"prompt": "merhaba"}], str(ckpt))


def test_golden_score_missing_tokenizer_id_errors(tmp_path) -> None:
    assertions = tmp_path / "a.jsonl"
    assertions.write_text(
        json.dumps({"id": 1, "prompt": "merhaba", "assertions": []}) + "\n",
        encoding="utf-8",
    )
    ckpt = tmp_path / "c.pt"
    _write_ckpt(ckpt, with_id=False)
    with pytest.raises(ValueError, match="tokenizer_id"):
        golden_score.run_model(assertions, ckpt, tmp_path / "out.jsonl", 8, 1)


def test_benchmarks_internal_missing_tokenizer_id_errors(monkeypatch, tmp_path) -> None:
    ckpt = tmp_path / "c.pt"
    _write_ckpt(ckpt, with_id=False)
    monkeypatch.setattr(sys, "argv", ["prog", "--run", "--ckpt", str(ckpt)])
    with pytest.raises(ValueError, match="tokenizer_id"):
        benchmarks_internal.main()
