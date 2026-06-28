"""PyTorch fallback (gerçek Vulkan backend henüz yok; sadece F.linear re-export)."""

from .engine import vulkan_linear

__all__ = ["vulkan_linear"]
