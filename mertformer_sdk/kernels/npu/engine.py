from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def npu_linear(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """STUB / FALLBACK: NPU linear path.

    HONEST NOTE: There is currently NO real NPU code path here. This is a plain
    ``torch.nn.functional.linear`` (generic torch matmul) fallback, not a
    NPU/QNN/CoreML kernel. The body is identical to standard ``F.linear`` and
    runs on whatever device the input tensor lives on (CPU/CUDA/etc.).

    Note: the dispatcher may select this "npu_fallback" even for CUDA tensors,
    in which case it is just generic GPU matmul. Any "NPU-native" claim in docs
    should be read as not-yet-implemented. Behavior is intentionally preserved.
    """
    return F.linear(x, w, bias)
