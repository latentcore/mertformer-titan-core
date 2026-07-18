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


def test_validate_generate_inputs_accepts_sane_call():
    api = importlib.import_module("mertformer_sdk.api")
    api._validate_generate_inputs("hello world", max_new_tokens=64, temperature=0.7, top_p=0.9)


def test_validate_generate_inputs_rejects_non_str_prompt():
    api = importlib.import_module("mertformer_sdk.api")
    with pytest.raises(api.InvalidGenerateInputError):
        api._validate_generate_inputs(123, max_new_tokens=64, temperature=0.7, top_p=0.9)


def test_validate_generate_inputs_rejects_empty_prompt():
    api = importlib.import_module("mertformer_sdk.api")
    with pytest.raises(api.InvalidGenerateInputError):
        api._validate_generate_inputs("   ", max_new_tokens=64, temperature=0.7, top_p=0.9)


def test_validate_generate_inputs_rejects_oversized_prompt():
    api = importlib.import_module("mertformer_sdk.api")
    with pytest.raises(api.InvalidGenerateInputError):
        api._validate_generate_inputs("a" * 100_001, max_new_tokens=64, temperature=0.7, top_p=0.9)


def test_validate_generate_inputs_rejects_control_characters():
    api = importlib.import_module("mertformer_sdk.api")
    with pytest.raises(api.InvalidGenerateInputError):
        api._validate_generate_inputs("hi\x00there", max_new_tokens=64, temperature=0.7, top_p=0.9)


def test_validate_generate_inputs_allows_newline_tab_cr():
    api = importlib.import_module("mertformer_sdk.api")
    api._validate_generate_inputs("line1\nline2\ttabbed\r\n", max_new_tokens=64, temperature=0.7, top_p=0.9)


@pytest.mark.parametrize("bad_max_new_tokens", [0, -1, 8193, 1.5, "64"])
def test_validate_generate_inputs_rejects_bad_max_new_tokens(bad_max_new_tokens):
    api = importlib.import_module("mertformer_sdk.api")
    with pytest.raises(api.InvalidGenerateInputError):
        api._validate_generate_inputs("hi", max_new_tokens=bad_max_new_tokens, temperature=0.7, top_p=0.9)


@pytest.mark.parametrize("bad_temperature", [0.0, -1.0, 5.1])
def test_validate_generate_inputs_rejects_bad_temperature(bad_temperature):
    api = importlib.import_module("mertformer_sdk.api")
    with pytest.raises(api.InvalidGenerateInputError):
        api._validate_generate_inputs("hi", max_new_tokens=64, temperature=bad_temperature, top_p=0.9)


@pytest.mark.parametrize("bad_top_p", [0.0, -0.1, 1.1])
def test_validate_generate_inputs_rejects_bad_top_p(bad_top_p):
    api = importlib.import_module("mertformer_sdk.api")
    with pytest.raises(api.InvalidGenerateInputError):
        api._validate_generate_inputs("hi", max_new_tokens=64, temperature=0.7, top_p=bad_top_p)
