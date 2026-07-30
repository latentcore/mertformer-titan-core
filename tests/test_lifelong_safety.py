from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from layers.lifelong_safety import LifelongSafetyLayer


def test_lifelong_safety_layer_forward_and_metrics():
    layer = LifelongSafetyLayer(hidden_size=32, ema_decay=0.9, max_adaptation_gain=0.1, drift_threshold=0.05)
    x = torch.randn(4, 6, 32)
    y = layer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    m = layer.safety_metrics()
    assert "last_drift" in m
    assert "drift_threshold" in m


@pytest.mark.parametrize("decay", [0.5, 0.9, 0.99])
def test_drift_is_measured_against_the_previous_ema_not_the_updated_one(decay):
    """Regression: drift was computed AFTER the EMA absorbed the current batch.

    [2026-07-29] `_update_stats` used to update `running_mean` first and only then take
    `(mean - running_mean).abs().mean()`. Because

        running' = d * running + (1 - d) * mean
        mean - running' = (1 - d) * (mean - running)

    the reported drift was exactly (1 - ema_decay) times the real deviation -- 1% of it
    at the default ema_decay=0.99. `drift_threshold=0.35` was therefore unreachable and
    the stability-first damping branch in `forward()` was dead code.

    Pinned on a fully deterministic input: running_mean starts at zeros, so feeding a
    constant `c` must report drift == |c| on the FIRST observation, independent of
    ema_decay. Under the old ordering the same input reported |c| * (1 - decay), which is
    why parametrising over decay is the discriminating part of this test.
    """
    hidden = 8
    constant = 1.0
    layer = LifelongSafetyLayer(hidden_size=hidden, ema_decay=decay)
    layer.train()
    layer(torch.full((2, 3, hidden), constant))
    drift = float(layer.last_drift.item())
    assert drift == pytest.approx(constant, abs=1e-6), (
        f"ema_decay={decay}: drift should be the raw deviation {constant}, got {drift} "
        f"(the old buggy value would be {constant * (1 - decay)})"
    )


def test_drift_threshold_is_actually_reachable():
    """A deviation above drift_threshold must be reported as such.

    With the pre-fix scaling this required a deviation ~100x the threshold at the
    default decay, which is why the damping branch never ran in practice.
    """
    layer = LifelongSafetyLayer(hidden_size=8, ema_decay=0.99, drift_threshold=0.35)
    layer.train()
    # running_mean starts at 0, so a constant 0.5 batch is a 0.5 deviation > 0.35.
    layer(torch.full((2, 3, 8), 0.5))
    assert float(layer.last_drift.item()) > 0.35


def test_ema_still_converges_toward_the_observed_mean():
    """The reordering must not break the EMA itself -- only when drift is sampled."""
    layer = LifelongSafetyLayer(hidden_size=4, ema_decay=0.5)
    layer.train()
    for _ in range(20):
        layer(torch.full((2, 3, 4), 2.0))
    assert float(layer.running_mean.mean().item()) == pytest.approx(2.0, abs=1e-3)


def test_eval_mode_does_not_update_running_stats():
    """EMA stats must drift only during training (pre-existing [P12] guarantee)."""
    layer = LifelongSafetyLayer(hidden_size=8, ema_decay=0.9)
    layer.eval()
    before = layer.running_mean.clone()
    layer(torch.full((2, 3, 8), 5.0))
    assert torch.equal(layer.running_mean, before)
    assert float(layer.last_drift.item()) == 0.0
