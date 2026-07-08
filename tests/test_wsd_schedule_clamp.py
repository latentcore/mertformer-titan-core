"""
[2026-07-08] Regression test for the WSD scheduler's unclamped cosine decay.

`get_wsd_schedule`'s decay phase computed
    progress = (current_step - stable_steps) / decay_steps
and fed it straight into `cos(pi * progress)` without clamping to [0, 1].

In a normal single run `current_step` never passes `num_training_steps`, so this never
fires. But the LambdaLR closure captures `num_training_steps`: if `TITAN_MAX_STEPS` changes
between a checkpoint-saving run and a later resume, the restored `last_epoch` is
reinterpreted against a *different* closure and `progress` can exceed 1.0 — at which point
`cos()` turns back upward and the learning rate RISES again instead of staying floored at
`min_lr_ratio`. Silent, and exactly the wrong direction for a run that is already unstable.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from train.trainer_core import get_wsd_schedule  # noqa: E402


def _lr_lambda(total_steps=1000, warmup=100, min_lr_ratio=0.01, stable_ratio=0.8):
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)
    sched = get_wsd_schedule(
        opt,
        num_warmup_steps=warmup,
        num_training_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
        stable_ratio=stable_ratio,
    )
    return sched.lr_lambdas[0]


def test_warmup_and_stable_phases_unchanged():
    fn = _lr_lambda()
    assert fn(0) == 0.0
    assert abs(fn(50) - 0.5) < 1e-12
    assert abs(fn(100) - 1.0) < 1e-12   # stable phase begins
    assert abs(fn(500) - 1.0) < 1e-12


def test_decay_phase_is_monotonically_non_increasing():
    fn = _lr_lambda()
    prev = fn(800)
    for step in range(801, 1001):
        cur = fn(step)
        assert cur <= prev + 1e-12, f"LR rose during decay at step {step}"
        prev = cur


def test_lr_stays_floored_past_num_training_steps():
    """
    The regression: with progress unclamped, step 1400 lands at progress=3.0 and
    cos(3*pi) = -1 -> 0.5*(1-1) = 0 -> floored, fine; but step 1600 gives progress=4.0 and
    cos(4*pi)=+1 -> 0.5*(1+1)=1.0 -> FULL learning rate again. Clamping pins it to the floor.
    """
    min_lr_ratio = 0.01
    fn = _lr_lambda(min_lr_ratio=min_lr_ratio)

    for step in (1000, 1200, 1400, 1600, 2000, 5000):
        assert abs(fn(step) - min_lr_ratio) < 1e-12, (
            f"step {step}: expected LR floored at {min_lr_ratio}, got {fn(step)}"
        )

    # Sanity: an unclamped cosine really would have sprung back to 1.0 at step 1600.
    stable_steps = int(1000 * 0.8)
    unclamped_progress = (1600 - stable_steps) / (1000 - stable_steps)
    unclamped = max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * unclamped_progress)))
    assert unclamped > 0.99
