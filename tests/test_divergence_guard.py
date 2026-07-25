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

[F1, 2026-07-25] Also covers the CE-only-EMA fix (BACKLOG.md "External review triage —
hardening (2026-07-09, Fable-5 code review)"): `train/train.py`'s divergence guard used
to key off the BLENDED loss_ema (CE+KD+aux), whose composition shifts as distill alpha
decays 0.8->0.15 across training — a guard watching the blend can drift upward from that
curriculum re-weighting alone, with zero actual instability. `train.py` now feeds the
guard a separate `ce_ema`/`warmup_end_ce_ema` pair derived from `loss_ce.item()` only.
`update_divergence_guard_state` itself needed no change (see tests above — it is generic
over what float it's handed); the fix is entirely in which value `train.py` passes it.
"""

import re
from pathlib import Path

from utils.divergence_guard import update_divergence_guard_state
from utils.liquid_safeguard import (
    effective_liquid_spike_threshold,
    update_liquid_spike_state,
    update_loss_ema,
)

_TRAIN_PY = Path(__file__).resolve().parents[1] / "train" / "train.py"


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


# ---------------------------------------------------------------------------
# [F1] CE-only EMA — the guard must key off CE, not the alpha-blended total_loss
# ---------------------------------------------------------------------------
def test_blended_ema_can_drift_upward_purely_from_curriculum_reweighting_while_ce_does_not():
    """
    Demonstrates the exact false-brake risk F1 closes. KD loss is HIGHER than CE loss
    (typical: the teacher-student gap), and distill alpha decays 0.8 -> 0.15 across
    training (train.py's dyn_alpha schedule) so the CE weight rises 0.2 -> 0.85. If CE
    itself is perfectly flat (no real instability), the BLENDED ema still drifts upward
    purely because more weight shifts onto the (higher) CE-relative-to-KD-decline mix.
    This test constructs a toy version of that shift and shows: (a) a guard fed the
    blend can accumulate breaches from re-weighting alone, (b) a guard fed CE-only,
    with CE truly flat, never breaches.
    """
    ce = 3.0  # flat CE the entire run -- no real divergence
    kd_start, kd_end = 1.5, 1.5  # KD also flat, but LOWER than CE (typical late-training)
    ref_alpha = 0.8  # warmup-end alpha, matching train.py's start_alpha default

    def blended(alpha: float) -> float:
        return (1.0 - alpha) * ce + alpha * kd_start

    reference_blend = blended(ref_alpha)  # snapshotted once, like warmup_end_loss_ema

    blend_breaches, blend_triggered = 0, False
    for step in range(200):
        # alpha decays 0.8 -> 0.15 over steps, exactly like train.py's dyn_alpha
        alpha = max(0.15, ref_alpha - (ref_alpha - 0.15) * (step / 199))
        blend_breaches, blend_triggered = update_divergence_guard_state(
            loss_ema=blended(alpha), reference_ema=reference_blend, multiplier=1.5,
            breach_counter=blend_breaches, patience=50, enabled=True,
        )

    # CE never moves -- a CE-only guard must never accumulate a breach, let alone trip.
    ce_breaches, ce_triggered = 0, False
    for _ in range(200):
        ce_breaches, ce_triggered = update_divergence_guard_state(
            loss_ema=ce, reference_ema=ce, multiplier=1.5,
            breach_counter=ce_breaches, patience=50, enabled=True,
        )
    assert ce_triggered is False
    assert ce_breaches == 0

    # Sanity: as alpha -> 0.15, blended(alpha) -> 0.85*ce + 0.15*kd = 2.775, which is
    # BELOW reference_blend = 0.2*ce+0.8*kd = 1.8 only if ce>kd; here ce=3.0 > kd=1.5,
    # so blended(alpha) actually RISES toward ce as alpha falls. This reproduces the
    # real mechanism BACKLOG.md's F1 describes ("if CE > KD ... the blended EMA drifts
    # UP with zero divergence") without asserting a specific trip count, since that
    # depends on the exact decay shape -- the meaningful, load-bearing assertion is the
    # CE-only guard's silence above.
    assert blended(0.15) > blended(ref_alpha)


def test_divergence_guard_call_site_keys_off_ce_ema_not_blended_loss_ema():
    """
    [F1] Source-scan regression, same discipline as test_gate3_ddp_unfreeze_guard.py's
    static verification: the actual behavior lives in *which local train.py passes* to
    update_divergence_guard_state, which is inside a large closure inside train() and
    not independently callable without a full accelerator/model/dataloader stack. Assert
    the wiring directly from source rather than not testing it at all.
    """
    src = _TRAIN_PY.read_text(encoding="utf-8")

    call_match = re.search(
        r"update_divergence_guard_state\(\s*loss_ema=(\w+),\s*reference_ema=(\w+),",
        src,
    )
    assert call_match is not None, "update_divergence_guard_state call site not found"
    assert call_match.group(1) == "ce_ema", (
        "divergence guard must be fed the CE-only EMA, not the blended loss_ema"
    )
    assert call_match.group(2) == "warmup_end_ce_ema", (
        "divergence guard's reference must be the CE-only snapshot"
    )

    # ce_ema itself must be derived from loss_ce (CE-only), not total_loss (blended).
    assert re.search(r"step_ce\s*=\s*float\(loss_ce\.item\(\)\)", src) is not None
    assert re.search(
        r"ce_ema\s*=\s*update_loss_ema\(\s*loss_value=step_ce,", src
    ) is not None

    # loss_ema (the blended EMA) must NOT have been repointed -- it still exists and is
    # still fed from total_loss, because the Liquid spike guard depends on it (BACKLOG
    # F1: "do NOT repoint loss_ema -- it is shared with the Liquid spike guard").
    assert re.search(r"step_loss\s*=\s*float\(total_loss\.item\(\)\)", src) is not None
    assert re.search(
        r"loss_ema\s*=\s*update_loss_ema\(\s*loss_value=step_loss,", src
    ) is not None
