from __future__ import annotations

import sys
from pathlib import Path

import pytest
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


@pytest.mark.parametrize("decay", [0.5, 0.9, 0.98])
def test_drift_is_the_loss_deviation_not_the_ema_movement(decay):
    """Regression: drift measured the EMA's own movement, scaled down by (1 - decay).

    [2026-07-29] It used to be `abs(running_loss_ema - prev_loss_ema)`. Since

        running' - prev = (1 - decay) * (loss - prev)

    that reported (1 - decay) times the real deviation -- 2% of it at the default
    decay=0.98. Combined with `drift_threshold=0.2` it required |loss - ema| > 10, i.e. a
    10-unit single-step jump, which train/train.py's NaN brake would catch long before.
    So `drift_alert` could effectively never fire and the "Continual Drift Alert" branch
    in train/train.py was dead.

    Pinned deterministically: step 1 seeds the EMA at the first loss, so on step 2 the
    drift must equal |loss2 - loss1| exactly, independent of decay. Parametrising over
    decay is what makes this test discriminating -- the old value scaled with it.
    """
    ad = ContinualLearningAdapter(replay_capacity=4, loss_ema_decay=decay, drift_threshold=1e9)
    ad.update(loss=1.0)
    state = ad.update(loss=2.0)
    assert state.running_loss_ema == pytest.approx(
        1.0 * decay + 2.0 * (1.0 - decay), abs=1e-9
    ), "the EMA update itself must be unchanged"
    ad2 = ContinualLearningAdapter(replay_capacity=4, loss_ema_decay=decay, drift_threshold=0.5)
    ad2.update(loss=1.0)
    s2 = ad2.update(loss=2.0)
    assert s2.drift_alert is True, (
        f"decay={decay}: a 1.0 loss deviation must exceed threshold 0.5; the old buggy "
        f"value would have been {1.0 * (1 - decay)}"
    )


def test_first_step_reports_no_drift():
    """Cold start seeds the EMA with the first loss, so there is nothing to deviate from."""
    ad = ContinualLearningAdapter(replay_capacity=2, loss_ema_decay=0.98, drift_threshold=1e-9)
    s1 = ad.update(loss=7.5)
    assert s1.running_loss_ema == pytest.approx(7.5)
    assert s1.drift_alert is False


def test_stable_loss_does_not_alert():
    """A flat loss stream must never trip the alert, at any decay."""
    ad = ContinualLearningAdapter(replay_capacity=8, loss_ema_decay=0.98, drift_threshold=0.2)
    alerts = []
    for _ in range(50):
        alerts.append(ad.update(loss=2.0).drift_alert)
    assert not any(alerts)


def test_replay_buffer_is_bounded():
    ad = ContinualLearningAdapter(replay_capacity=3, loss_ema_decay=0.9, drift_threshold=1.0)
    for i in range(10):
        ad.update(loss=1.0, sample=torch.tensor([i]))
    assert ad.state().replay_size == 3
    # deque(maxlen) keeps the most recent items, in insertion order.
    kept = [int(t.item()) for t in ad.replay.sample(3)]
    assert kept == [7, 8, 9]
