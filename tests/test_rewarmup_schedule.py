"""
[2026-07-19] Tests for get_rewarmup_schedule (BACKLOG "LR re-warmup" item).

The canonical 45K run's WSD schedule decays learning_rate down to min_lr_ratio by
design. Resuming a NEW training run (SFT / DMSR ablation / additional pre-training)
from that checkpoint with the same scheduler state keeps the LR pinned at the floor
for the entire continuation run -- get_rewarmup_schedule() re-anchors to step 0 and
ramps back up before repeating the same warmup-stable-decay shape as get_wsd_schedule.

This is new, additive code (config.use_rewarmup_schedule defaults OFF); it does not
touch or change get_wsd_schedule, which the canonical 45K run still uses unmodified.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from train.trainer_core import get_rewarmup_schedule  # noqa: E402


def _lr_lambda(
    total_steps=1000,
    rewarmup_steps=100,
    start_lr_ratio=0.01,
    peak_lr_ratio=1.0,
    min_lr_ratio=0.01,
    stable_ratio=0.8,
):
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)
    sched = get_rewarmup_schedule(
        opt,
        num_rewarmup_steps=rewarmup_steps,
        num_training_steps=total_steps,
        start_lr_ratio=start_lr_ratio,
        peak_lr_ratio=peak_lr_ratio,
        min_lr_ratio=min_lr_ratio,
        stable_ratio=stable_ratio,
    )
    return sched.lr_lambdas[0]


def test_starts_at_start_lr_ratio_not_zero():
    """Unlike get_wsd_schedule's cold start at 0.0, re-warmup starts from wherever
    the base run's LR floor actually was."""
    fn = _lr_lambda(start_lr_ratio=0.01)
    assert abs(fn(0) - 0.01) < 1e-12


def test_ramps_linearly_to_peak_over_rewarmup_steps():
    fn = _lr_lambda(rewarmup_steps=100, start_lr_ratio=0.0, peak_lr_ratio=1.0)
    assert abs(fn(0) - 0.0) < 1e-12
    assert abs(fn(50) - 0.5) < 1e-9
    assert abs(fn(100) - 1.0) < 1e-9  # stable phase begins exactly at rewarmup_steps


def test_stable_phase_holds_at_peak():
    fn = _lr_lambda(total_steps=1000, rewarmup_steps=100, peak_lr_ratio=1.0)
    assert abs(fn(200) - 1.0) < 1e-9
    assert abs(fn(500) - 1.0) < 1e-9


def test_decay_phase_is_monotonically_non_increasing():
    fn = _lr_lambda(total_steps=1000, rewarmup_steps=100)
    prev = fn(800)
    for step in range(801, 1001):
        cur = fn(step)
        assert cur <= prev + 1e-12, f"LR rose during decay at step {step}"
        prev = cur


def test_lr_stays_floored_past_num_training_steps():
    """Same progress-clamp discipline as get_wsd_schedule's 2026-07-08 fix -- a
    changed num_training_steps between save and resume must not send LR back up."""
    min_lr_ratio = 0.02
    fn = _lr_lambda(total_steps=1000, rewarmup_steps=100, min_lr_ratio=min_lr_ratio)

    for step in (1000, 1200, 1600, 2000, 5000):
        assert abs(fn(step) - min_lr_ratio) < 1e-9, (
            f"step {step}: expected LR floored at {min_lr_ratio}, got {fn(step)}"
        )

    # Sanity: an unclamped cosine really would have sprung back up past step 1000.
    # progress=4.0 exactly -> cos(4*pi)=cos(0)=1 -> back to peak, unclamped.
    stable_steps = 100 + int((1000 - 100) * 0.8)
    decay_steps = 1000 - stable_steps
    step_at_progress_4 = stable_steps + 4 * decay_steps
    unclamped_progress = (step_at_progress_4 - stable_steps) / decay_steps
    unclamped = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * unclamped_progress)
    )
    assert unclamped > 0.99


def test_zero_rewarmup_steps_does_not_divide_by_zero():
    """num_rewarmup_steps=0 (degenerate config) must not crash -- max(1, ...) guard."""
    fn = _lr_lambda(total_steps=1000, rewarmup_steps=0)
    # step 0 is already >= num_rewarmup_steps=0, so it lands directly in stable phase.
    assert fn(0) == pytest.approx(1.0)


def test_config_defaults_to_off_and_matches_base_min_lr_ratio():
    """use_rewarmup_schedule must default False so the canonical 45K path is
    unaffected unless TITAN_USE_REWARMUP=1 is explicitly set; rewarmup_start_lr_ratio
    defaults to the same value as min_lr_ratio so a continuation run's cold start
    lines up with wherever the base run actually landed."""
    from config.config import MertFormerConfig

    cfg = MertFormerConfig()
    assert cfg.use_rewarmup_schedule is False
    assert cfg.rewarmup_start_lr_ratio == cfg.min_lr_ratio
