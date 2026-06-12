"""Coverage for deterministic-resume RNG round-trip (train/trainer_core.py).

_capture_rng_state -> draw -> _restore_rng_state must reproduce the exact same draws,
and the captured numpy/python entries must be plain builtins so a weights_only=True
checkpoint stays loadable. Pure CPU, deterministic.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.trainer_core import _capture_rng_state, _restore_rng_state, seed_all  # noqa: E402


def _draw():
    t = torch.rand(4)
    p = [random.random() for _ in range(3)]
    try:
        import numpy as np

        n = np.random.rand(3).tolist()
    except Exception:
        n = []
    return t, p, n


def test_rng_state_round_trip_reproduces_draws():
    seed_all(123)
    _ = torch.rand(7)  # advance so the snapshot is non-trivial
    snap = _capture_rng_state()
    t1, p1, n1 = _draw()
    _restore_rng_state(snap)
    t2, p2, n2 = _draw()
    assert torch.equal(t1, t2)
    assert p1 == p2
    assert n1 == n2


def test_captured_rng_state_numpy_python_are_builtin_only():
    """numpy/python entries must be builtins (no numpy arrays) for weights_only=True loads."""
    seed_all(7)
    snap = _capture_rng_state()
    np_entry = snap["numpy"]
    assert isinstance(np_entry, list)
    assert isinstance(np_entry[1], list) and all(isinstance(v, int) for v in np_entry[1])
    py_entry = snap["python"]
    assert isinstance(py_entry, list)
    assert isinstance(py_entry[1], list) and all(isinstance(v, int) for v in py_entry[1])
