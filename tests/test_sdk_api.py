from __future__ import annotations

import importlib
import pytest


def test_sdk_imports():
    api = importlib.import_module("mertformer_sdk.api")
    assert hasattr(api, "load_model")
    assert hasattr(api, "generate")
    assert hasattr(api, "benchmark")
    assert hasattr(api, "enable_lowbit_kernels")


def test_lowbit_toggle():
    from layers import bitlinear
    bitlinear.set_lowbit_kernel_enabled(False)
    bitlinear.set_lowbit_kernel_enabled(True)


def test_load_model_strict_missing_checkpoint(monkeypatch):
    api = importlib.import_module("mertformer_sdk.api")

    called = {"model_init": False}

    def _blocked_model_init():
        called["model_init"] = True
        raise AssertionError("MertFormer should not be initialized when strict checkpoint is missing")

    monkeypatch.setattr(api, "MertFormer", _blocked_model_init)

    with pytest.raises(FileNotFoundError):
        api.load_model(
            ckpt="checkpoints/does_not_exist.pt",
            device="cpu",
            strict_checkpoint=True,
        )

    assert called["model_init"] is False


def test_load_model_rejects_missing_tokenizer_identity(monkeypatch, tmp_path):
    """H4: a checkpoint with NO tokenizer_id must hard-fail (no silent teacher
    fallback) BEFORE the model is built."""
    import importlib
    import torch

    api = importlib.import_module("mertformer_sdk.api")

    def _blocked_model_init():
        raise AssertionError("MertFormer must not be built when tokenizer_id is missing")

    monkeypatch.setattr(api, "MertFormer", _blocked_model_init)

    ckpt = tmp_path / "no_id.pt"
    torch.save({"model": {}}, ckpt)  # no tokenizer_id

    with pytest.raises(ValueError, match="tokenizer_id"):
        api.load_model(ckpt=ckpt, device="cpu", strict_checkpoint=True)


def test_load_model_uses_checkpoint_tokenizer_identity(monkeypatch, tmp_path):
    """H4: load_model loads the tokenizer from checkpoint['tokenizer_id'] and resizes
    the model to that vocab (no current-cfg teacher substitution)."""
    import importlib
    import torch

    api = importlib.import_module("mertformer_sdk.api")
    import utils.tokenizer_resolver as tr

    class _StubTok:
        name_or_path = "stub-tok"

        def __len__(self):
            return 123

    monkeypatch.setattr(tr, "load_tokenizer_from_identity", lambda identity: _StubTok())

    captured = {}

    class _StubModel:
        def to(self, *_a, **_k):
            return self

        def resize_token_embeddings(self, n):
            captured["resized"] = n

        def load_state_dict(self, *_a, **_k):
            captured["loaded"] = True

        def eval(self):
            return self

    monkeypatch.setattr(api, "MertFormer", lambda: _StubModel())

    ckpt = tmp_path / "with_id.pt"
    torch.save({"model": {}, "tokenizer_id": {"name_or_path": "stub-tok", "vocab_size": 123}}, ckpt)

    model, tokenizer, device = api.load_model(ckpt=ckpt, device="cpu", strict_checkpoint=True)
    assert isinstance(tokenizer, _StubTok)
    assert captured.get("resized") == 123  # model resized to checkpoint tokenizer vocab
    assert captured.get("loaded") is True
