"""save_checkpoint_smart must stamp the runtime tokenizer identity.

This is the write side of the train/eval tokenizer-parity contract: every
checkpoint records which tokenizer produced it so eval/demo can reload exactly
that one (see tests/test_tokenizer_parity.py for the read side).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train.train")


class _Stateless:
    def state_dict(self):
        return {}


def _make_cfg() -> SimpleNamespace:
    return SimpleNamespace(save_dir="ckpts", model_name="unit")


def test_checkpoint_records_tokenizer_id(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(train, "project_root", tmp_path)
    identity = {
        "name_or_path": "data/tokenizer/tr",
        "vocab_size": 128000,
        "tokenizer_class": "BertTokenizer",
        "use_tr_tokenizer": True,
    }
    monkeypatch.setattr(train, "RUNTIME_TOKENIZER_ID", identity)

    train.save_checkpoint_smart(
        _Stateless(), _Stateless(), _Stateless(), step=1, cfg=_make_cfg()
    )

    latest = tmp_path / "ckpts" / "unit_latest.pt"
    assert latest.exists()
    state = torch.load(latest, weights_only=False)
    assert state["tokenizer_id"] == identity


def test_end_to_end_train_write_eval_read_same_tokenizer(monkeypatch, tmp_path) -> None:
    """PROOF (task item c): the tokenizer eval reloads from a checkpoint written
    by the real train path is byte-for-byte the tokenizer training used."""
    from utils import tokenizer_resolver as tr

    tr_dir = tr.PROJECT_ROOT / "data" / "tokenizer" / "tr"
    if not (tr_dir / "tokenizer.json").exists():
        pytest.skip("local TR tokenizer artifact missing")

    cfg = SimpleNamespace(
        save_dir="ckpts",
        model_name="unit",
        use_tr_tokenizer=True,
        tr_tokenizer_id="data/tokenizer/tr",
    )

    # --- train side: resolve, lock identity, write a real checkpoint ---
    train_tok = tr.resolve_tokenizer(cfg)
    monkeypatch.setattr(train, "project_root", tmp_path)
    monkeypatch.setattr(train, "RUNTIME_TOKENIZER_ID", tr.tokenizer_identity(train_tok, cfg))
    train.save_checkpoint_smart(
        _Stateless(), _Stateless(), _Stateless(), step=7, cfg=cfg
    )

    # --- eval side: reload strictly from the checkpoint's recorded identity ---
    state = torch.load(tmp_path / "ckpts" / "unit_latest.pt", weights_only=False)
    eval_tok = tr.load_tokenizer_from_identity(state["tokenizer_id"])

    text = "iki kere iki dort eder"
    assert train_tok(text).input_ids == eval_tok(text).input_ids
    assert len(train_tok) == len(eval_tok) == state["tokenizer_id"]["vocab_size"]


def test_checkpoint_tokenizer_id_none_when_unset(monkeypatch, tmp_path) -> None:
    # Defensive: when training never locked a tokenizer, the field is present and
    # explicitly None so eval's loader raises rather than silently guessing.
    monkeypatch.setattr(train, "project_root", tmp_path)
    monkeypatch.setattr(train, "RUNTIME_TOKENIZER_ID", None)

    train.save_checkpoint_smart(
        _Stateless(), _Stateless(), _Stateless(), step=2, cfg=_make_cfg()
    )

    state = torch.load(tmp_path / "ckpts" / "unit_latest.pt", weights_only=False)
    assert "tokenizer_id" in state
    assert state["tokenizer_id"] is None
