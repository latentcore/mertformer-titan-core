from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def metal_linear(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Fallback-safe Metal linear path.

    Uses deterministic PyTorch linear as verified fallback until custom shaders are introduced.
    """
    return F.linear(x, w, bias)
