"""
[2026-07-08] Pre-45K stabilization guards.

Covers the two pure-function guards introduced by the 2026-07-08 pass:

* `utils.liquid_safeguard.update_loss_ema` / `effective_liquid_spike_threshold`
  — the Liquid spike threshold moved from a SCALE-BLIND absolute (`loss > 5.0`) to a
  scale-relative one (`loss > loss_ema * multiplier`). With the absolute threshold, the
  2026-07-02 pre-flight (start loss ~10.4) tripped the guard on every step and the Liquid
  layers were never trained.

* `utils.divergence_guard.update_divergence_guard_state`
  — a general circuit breaker for "loss is finite but climbing", the exact failure the
  NaN brake (loss never went non-finite) and the Liquid guard (only freezes Liquid params)
  both missed on 2026-07-02: 10.4 -> 15.0 over ~900 steps.
"""

from utils.divergence_guard import update_divergence_guard_state
from utils.liquid_safeguard import (
    effective_liquid_spike_threshold,
    update_liquid_spike_state,
    update_loss_ema,
)


# ---------------------------------------------------------------------------
# Loss EMA (cold-start safe)
# ---------------------------------------------------------------------------
def test_loss_ema_seeds_with_first_observation():
    """First observation seeds the EMA; it must not blend against a fictitious 0.0."""
    assert update_loss_ema(loss_value=10.4, loss_ema=0.0, observations=0, decay=0.98) == 10.4


def test_loss_ema_blends_after_cold_start():
    ema = update_loss_ema(loss_value=10.0, loss_ema=0.0, observations=0, decay=0.9)
    ema = update_loss_ema(loss_value=20.0, loss_ema=ema, observations=1, decay=0.9)
    assert abs(ema - (10.0 * 0.9 + 20.0 * 0.1)) < 1e-12


# ---------------------------------------------------------------------------
# Liquid spike threshold: absolute floor -> relative
# ---------------------------------------------------------------------------
def test_spike_threshold_uses_absolute_floor_during_ema_cold_start():
    t = effective_liquid_spike_threshold(
        loss_ema=10.4, observations=5, absolute_threshold=5.0,
        relative_multiplier=1.5, ema_warmup_steps=100,
    )
    assert t == 5.0


def test_spike_threshold_goes_relative_once_ema_is_warm():
    t = effective_liquid_spike_threshold(
        loss_ema=10.0, observations=100, absolute_threshold=5.0,
        relative_multiplier=1.5, ema_warmup_steps=100,
    )
    assert abs(t - 15.0) < 1e-12


def test_relative_threshold_stops_freezing_liquid_on_a_high_loss_scale_run():
    """
    Regression for the 2026-07-02 behavior: at a loss scale of ~10, the absolute 5.0
    threshold strikes every step and freezes Liquid forever. The relative threshold must
    NOT strike on a loss that merely sits at the run's own EMA.
    """
    loss, ema = 10.4, 10.4

    absolute = 5.0  # the old, scale-blind threshold

    counter, frozen, triggered = 0, 0, False
    for step in range(1, 4):
        counter, frozen, triggered = update_liquid_spike_state(
            loss_value=loss, threshold=absolute, counter=counter, patience=3,
            frozen_until=frozen, global_step=step, cooldown_steps=200, enabled=True,
        )
    assert triggered is True, "sanity: the old absolute threshold DID freeze Liquid here"

    relative = effective_liquid_spike_threshold(
        loss_ema=ema, observations=1000, absolute_threshold=5.0,
        relative_multiplier=1.5, ema_warmup_steps=100,
    )
    counter, frozen, triggered = 0, 0, False
    for step in range(1, 11):
        counter, frozen, triggered = update_liquid_spike_state(
            loss_value=loss, threshold=relative, counter=counter, patience=3,
            frozen_until=frozen, global_step=step, cooldown_steps=200, enabled=True,
        )
    assert triggered is False, "relative threshold must not fire on a loss at its own EMA"

    # ...but a genuine spike well above the EMA still freezes.
    counter, frozen, triggered = 0, 0, False
    for step in range(1, 4):
        counter, frozen, triggered = update_liquid_spike_state(
            loss_value=ema * 2.0, threshold=relative, counter=counter, patience=3,
            frozen_until=frozen, global_step=step, cooldown_steps=200, enabled=True,
        )
    assert triggered is True


# ---------------------------------------------------------------------------
# Loss-divergence circuit breaker
# ---------------------------------------------------------------------------
def test_divergence_guard_disarmed_without_reference():
    counter, triggered = update_divergence_guard_state(
        loss_ema=99.0, reference_ema=None, multiplier=1.5,
        breach_counter=7, patience=3, enabled=True,
    )
    assert (counter, triggered) == (0, False)


def test_divergence_guard_disabled_is_noop():
    counter, triggered = update_divergence_guard_state(
        loss_ema=99.0, reference_ema=8.0, multiplier=1.5,
        breach_counter=2, patience=3, enabled=False,
    )
    assert (counter, triggered) == (0, False)


def test_divergence_guard_ignores_nonpositive_reference():
    counter, triggered = update_divergence_guard_state(
        loss_ema=1.0, reference_ema=0.0, multiplier=1.5,
        breach_counter=0, patience=1, enabled=True,
    )
    assert (counter, triggered) == (0, False)


def test_divergence_guard_requires_sustained_breach():
    counter, triggered = 0, False
    for _ in range(2):
        counter, triggered = update_divergence_guard_state(
            loss_ema=15.0, reference_ema=8.0, multiplier=1.5,
            breach_counter=counter, patience=3, enabled=True,
        )
        assert triggered is False
    counter, triggered = update_divergence_guard_state(
        loss_ema=15.0, reference_ema=8.0, multiplier=1.5,
        breach_counter=counter, patience=3, enabled=True,
    )
    assert triggered is True


def test_divergence_guard_counter_resets_on_recovery():
    counter, _ = update_divergence_guard_state(
        loss_ema=15.0, reference_ema=8.0, multiplier=1.5,
        breach_counter=2, patience=5, enabled=True,
    )
    assert counter == 3
    counter, triggered = update_divergence_guard_state(
        loss_ema=8.1, reference_ema=8.0, multiplier=1.5,
        breach_counter=counter, patience=5, enabled=True,
    )
    assert (counter, triggered) == (0, False)


def test_divergence_guard_catches_the_2026_07_02_curve():
    """
    Replay the shape of the real diverging run: warmup-end EMA ~8.4, loss climbs to ~15.0
    and stays there. The guard must brake; a healthy run that stays at/below the reference
    must not.
    """
    reference = 8.4
    counter, triggered = 0, False
    for _ in range(60):
        counter, triggered = update_divergence_guard_state(
            loss_ema=15.0, reference_ema=reference, multiplier=1.5,
            breach_counter=counter, patience=50, enabled=True,
        )
        if triggered:
            break
    assert triggered is True

    counter, triggered = 0, False
    for _ in range(500):
        counter, triggered = update_divergence_guard_state(
            loss_ema=7.9, reference_ema=reference, multiplier=1.5,
            breach_counter=counter, patience=50, enabled=True,
        )
    assert triggered is False
