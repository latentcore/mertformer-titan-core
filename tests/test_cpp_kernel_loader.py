from __future__ import annotations

import sys
from pathlib import Path

import torch

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from mertformer_sdk.kernels.cpp.loader import bitnet_cpu_linear, load_cpp_kernel


def test_cpp_loader_fallback_path_works_without_build():
    ext = load_cpp_kernel(build=False)
    assert ext is None
    x = torch.randn(3, 5)
    w = torch.randn(7, 5)
    b = torch.randn(7)
    y = bitnet_cpu_linear(x, w, b)
    assert y.shape == (3, 7)
    assert torch.isfinite(y).all()

