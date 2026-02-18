from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from mertformer_sdk.kernels.dispatcher import select_backend


def test_kernel_dispatcher_force_override():
    x = torch.randn(2, 4)
    w = torch.randn(8, 4)
    os.environ["MERTFORMER_KERNEL_BACKEND"] = "pytorch_fallback"
    try:
        assert select_backend(x, w) == "pytorch_fallback"
    finally:
        os.environ.pop("MERTFORMER_KERNEL_BACKEND", None)


def test_kernel_dispatcher_cpu_path():
    x = torch.randn(2, 4)
    w = torch.randn(8, 4)
    backend = select_backend(x, w)
    assert backend in {"cpp_cpu", "pytorch_fallback"}

