"""Train/eval tokenizer parity guards.

These tests pin the fix for the train/eval tokenizer-family mismatch: a single
resolver, an explicit tokenizer identity stamped into the checkpoint, and a
hard error (never a silent teacher fallback) when that identity is absent or
inconsistent.

The local Turkish tokenizer (``data/tokenizer/tr``) is used as the offline
fixture because it ships in-repo and loads without network/gated access.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from utils import tokenizer_resolver as tr

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

# The TR tokenizer fixture is only present when its artifact exists.
_TR_TOKENIZER_DIR = tr.PROJECT_ROOT / "data" / "tokenizer" / "tr"
_HAS_TR = (_TR_TOKENIZER_DIR / "tokenizer.json").exists()
_requires_tr = pytest.mark.skipif(not _HAS_TR, reason="local TR tokenizer artifact missing")


def _tr_cfg() -> SimpleNamespace:
    return SimpleNamespace(use_tr_tokenizer=True, tr_tokenizer_id="data/tokenizer/tr")


# ---------------------------------------------------------------------------
# No silent fallback: a checkpoint without a tokenizer identity is fatal.
# ---------------------------------------------------------------------------
def test_load_from_identity_requires_tokenizer_id() -> None:
    with pytest.raises(ValueError, match="tokenizer_id"):
        tr.load_tokenizer_from_identity(None)
    with pytest.raises(ValueError, match="tokenizer_id"):
        tr.load_tokenizer_from_identity({})
    with pytest.raises(ValueError, match="tokenizer_id"):
        tr.load_tokenizer_from_identity({"vocab_size": 128000})


@_requires_tr
def test_load_from_identity_rejects_vocab_mismatch() -> None:
    identity = {
        "name_or_path": str(_TR_TOKENIZER_DIR),
        "vocab_size": 999999,  # deliberately wrong
        "tokenizer_class": "BertTokenizer",
    }
    with pytest.raises(ValueError, match="vocab mismatch"):
        tr.load_tokenizer_from_identity(identity)


# ---------------------------------------------------------------------------
# Single source of truth + identity round-trip.
# ---------------------------------------------------------------------------
@_requires_tr
def test_resolve_tokenizer_tr_optin_is_single_source() -> None:
    cfg = _tr_cfg()
    tok = tr.resolve_tokenizer(cfg)
    identity = tr.tokenizer_identity(tok, cfg)

    assert identity["use_tr_tokenizer"] is True
    assert identity["vocab_size"] == len(tok)
    assert identity["name_or_path"].endswith("data/tokenizer/tr")


# ---------------------------------------------------------------------------
# PROOF (task item c): the tokenizer eval reloads from the checkpoint is the
# IDENTICAL tokenizer training resolved and stamped.
# ---------------------------------------------------------------------------
@_requires_tr
def test_train_eval_tokenizer_parity(tmp_path) -> None:
    cfg = _tr_cfg()

    # --- "training" side: resolve + stamp identity into a checkpoint ---
    train_tok = tr.resolve_tokenizer(cfg)
    train_identity = tr.tokenizer_identity(train_tok, cfg)

    ckpt_path = tmp_path / "unit_latest.pt"
    torch.save({"model": {}, "tokenizer_id": train_identity}, ckpt_path)

    # --- "eval" side: reload strictly from the recorded identity ---
    checkpoint = torch.load(ckpt_path, weights_only=False)
    eval_tok = tr.load_tokenizer_from_identity(checkpoint["tokenizer_id"])
    eval_identity = tr.tokenizer_identity(eval_tok, cfg)

    # Same family, same path, same vocab => train and eval tokenizers match.
    assert eval_identity["name_or_path"] == train_identity["name_or_path"]
    assert eval_identity["vocab_size"] == train_identity["vocab_size"]
    assert eval_identity["tokenizer_class"] == train_identity["tokenizer_class"]
    assert type(eval_tok).__name__ == type(train_tok).__name__

    # And a concrete behavioural check: identical encoding of the same text.
    text = "merhaba dunya 123"
    assert train_tok(text).input_ids == eval_tok(text).input_ids


# ---------------------------------------------------------------------------
# vocab_size is derived from len(tokenizer) and the model aligns to it.
# ---------------------------------------------------------------------------
def test_resize_token_embeddings_aligns_vocab_and_keeps_tie() -> None:
    import torch.nn as nn

    from model.transformers import MertFormer

    fake = SimpleNamespace()
    fake.tok_embeddings = nn.Embedding(10, 4)
    fake.lm_head = nn.Linear(4, 10, bias=False)
    fake.lm_head.weight = fake.tok_embeddings.weight  # tied
    fake.cfg = SimpleNamespace(vocab_size=10)

    # Grow (mirrors 128000 -> 128256 alignment).
    MertFormer.resize_token_embeddings(fake, 16)
    assert fake.tok_embeddings.num_embeddings == 16
    assert fake.lm_head.out_features == 16
    assert fake.lm_head.weight is fake.tok_embeddings.weight  # tie preserved
    assert fake.cfg.vocab_size == 16

    # No-op when already aligned.
    same = fake.tok_embeddings
    MertFormer.resize_token_embeddings(fake, 16)
    assert fake.tok_embeddings is same


def test_resize_preserves_existing_rows() -> None:
    import torch.nn as nn

    from model.transformers import MertFormer

    fake = SimpleNamespace()
    emb = nn.Embedding(5, 3)
    with torch.no_grad():
        emb.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3))
    fake.tok_embeddings = emb
    fake.lm_head = nn.Linear(3, 5, bias=False)
    fake.lm_head.weight = fake.tok_embeddings.weight
    fake.cfg = SimpleNamespace(vocab_size=5)

    before = fake.tok_embeddings.weight[:5].clone()
    MertFormer.resize_token_embeddings(fake, 8)
    assert torch.equal(fake.tok_embeddings.weight[:5], before)
