from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def metal_linear(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Metal linear path -- STUB / PyTorch fallback only.

    NOTE (honest backend status): this is NOT a real Metal/MPS shader kernel.
    There is no custom Metal compute pipeline here; the body is a plain
    ``torch.nn.functional.linear`` passthrough used as a deterministic,
    verified fallback until real custom shaders are introduced. Despite the
    ``metal`` package/backend name, this routes to the generic PyTorch path
    (effectively ``pytorch_fallback``), not to GPU shader code.
    """
    return F.linear(x, w, bias)
