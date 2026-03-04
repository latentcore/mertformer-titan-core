from __future__ import annotations

import os

import torch

from mertformer_sdk.kernels.dispatcher import select_backend


def test_dispatcher_forced_backend():
    x = torch.randn(2, 4)
    w = torch.randn(3, 4)
    os.environ["MERTFORMER_KERNEL_BACKEND"] = "vulkan_fallback"
    try:
        assert select_backend(x, w) == "vulkan_fallback"
    finally:
        os.environ.pop("MERTFORMER_KERNEL_BACKEND", None)


def test_dispatcher_cpu_prefers_cpp_when_enabled():
    x = torch.randn(2, 4)
    w = torch.randn(3, 4)
    os.environ["MERTFORMER_ENABLE_CPP_KERNEL"] = "1"
    os.environ.pop("MERTFORMER_KERNEL_BACKEND", None)
    assert select_backend(x, w) == "cpp_cpu"
