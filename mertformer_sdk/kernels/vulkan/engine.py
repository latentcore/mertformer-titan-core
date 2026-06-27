from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def vulkan_linear(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """STUB/FALLBACK: Vulkan henuz uygulanmadi; gercek Vulkan compute-shader yoktur.

    Bu fonksiyon hizlandirilmis bir Vulkan backend DEGILDIR; sadece saf PyTorch
    F.linear fallback'idir (metal_linear ile birebir ayni passthrough). Dispatcher
    bunu gercek bir hizlandirilmis backend olarak saymamalidir.
    """
    return F.linear(x, w, bias)
