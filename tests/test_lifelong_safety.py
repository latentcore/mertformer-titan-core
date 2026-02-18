from __future__ import annotations

import sys
from pathlib import Path

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

