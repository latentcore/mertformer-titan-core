"""Coverage for utils.safety.kill_if_non_finite + is_finite (the exported numerical kill-switch).

Gives the public utility a concrete test reference and pins its three actions. Pure CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.safety import is_finite, kill_if_non_finite  # noqa: E402


def test_is_finite_scalars_and_tensors():
    assert is_finite(1.0) and is_finite(torch.zeros(3))
    assert not is_finite(float("nan"))
    assert not is_finite(torch.tensor([1.0, float("inf")]))
    assert is_finite([1.0, torch.zeros(2)])  # nested ok


def test_kill_if_non_finite_passes_through_finite():
    # Must NOT raise on finite input.
    kill_if_non_finite(torch.zeros(4), name="ok", action="raise")


def test_kill_if_non_finite_raises_on_nan():
    with pytest.raises(RuntimeError):
        kill_if_non_finite(torch.tensor([float("nan")]), name="loss", action="raise")


def test_kill_if_non_finite_sys_exit_action():
    with pytest.raises(SystemExit):
        kill_if_non_finite(float("inf"), name="loss", action="sys_exit", exit_code=3)
