"""
Safety utilities for numerical stability and kill-switch behavior.
"""
from __future__ import annotations

import math
import os
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _is_finite_number(x: Any) -> bool:
    # bool is a subclass of int in Python, so isinstance(True, (int, float))
    # is True. Handle it explicitly: a bool is always considered finite
    # (it carries no NaN/Inf risk). Behavior is unchanged (still returns True
    # for bools); this branch only makes the intent explicit.
    if isinstance(x, bool):
        return True
    if isinstance(x, (int, float)):
        return math.isfinite(x)
    return True


def is_finite(value: Any) -> bool:
    if _is_finite_number(value) is False:
        return False

    if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(value):
        try:
            return bool(torch.isfinite(value).all().item())
        except Exception:
            return False

    if isinstance(value, (list, tuple)):
        return all(is_finite(v) for v in value)

    return True


def kill_if_non_finite(value: Any, *, name: str = "value", action: str = "raise", exit_code: int = 3) -> None:
    """
    Enforce a kill switch when non-finite values are detected.

    action:
      - "raise": raise RuntimeError
      - "exit": os._exit(exit_code) for hard stop
      - "sys_exit": raise SystemExit(exit_code)
    """
    if is_finite(value):
        return

    msg = f"Non-finite detected in {name}."

    if action == "exit":
        os._exit(exit_code)
    if action == "sys_exit":
        raise SystemExit(exit_code)

    raise RuntimeError(msg)
