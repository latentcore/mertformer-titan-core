from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def vulkan_linear(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """STUB/FALLBACK: Vulkan henüz uygulanmadı; gerçek Vulkan compute-shader yoktur.

    Bu fonksiyon hızlandırılmış bir Vulkan backend DEĞİLDİR; sadece saf PyTorch
    F.linear fallback'idir (metal_linear ile birebir aynı passthrough). Dispatcher
    bunu gerçek bir hızlandırılmış backend olarak saymamalıdır.
    """
    return F.linear(x, w, bias)
