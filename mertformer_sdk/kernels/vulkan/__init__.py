"""PyTorch fallback (gercek Vulkan backend henuz yok; sadece F.linear re-export)."""

from .engine import vulkan_linear

__all__ = ["vulkan_linear"]
