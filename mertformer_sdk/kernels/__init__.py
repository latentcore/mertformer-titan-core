"""Kernel availability helpers and backend interfaces."""
from __future__ import annotations

from .dispatcher import select_backend
from .onnx_custom_op import detect_onnx_custom_op_plugin, export_contract


def is_triton_available() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except Exception:
        return False


__all__ = [
    "is_triton_available",
    "select_backend",
    "detect_onnx_custom_op_plugin",
    "export_contract",
]
