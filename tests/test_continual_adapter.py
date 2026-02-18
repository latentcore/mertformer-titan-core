from __future__ import annotations

import sys
from pathlib import Path

import torch

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from train.continual_adapter import ContinualLearningAdapter


def test_continual_adapter_tracks_replay_and_drift():
    ad = ContinualLearningAdapter(replay_capacity=4, loss_ema_decay=0.5, drift_threshold=0.01)
    s1 = ad.update(loss=1.0, sample=torch.tensor([1, 2, 3]))
    s2 = ad.update(loss=2.0, sample=torch.tensor([4, 5, 6]))
    assert s2.step == 2
    assert s2.replay_size == 2
    assert isinstance(s2.drift_alert, bool)
    d = ad.to_dict()
    assert d["replay_size"] == 2

