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
