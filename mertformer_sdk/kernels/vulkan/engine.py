from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def vulkan_linear(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Fallback-safe Vulkan linear path."""
    return F.linear(x, w, bias)
