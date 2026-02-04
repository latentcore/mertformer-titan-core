from __future__ import annotations

import importlib


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
