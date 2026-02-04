"""Kernel availability helpers."""
from __future__ import annotations


def is_triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False
