"""
Liquid safeguard helpers.

Pure-Python utilities to keep liquid spike tracking testable without importing
the full training stack.

[2026-07-08 pre-45K stabilization] The spike guard used to compare the raw loss
against a fixed absolute threshold (`loss > 5.0`). That threshold is SCALE-BLIND:
the 2026-07-02 laptop pre-flight started at loss ~10.4, so the guard fired on every
single step and the Liquid layers were effectively never trained. The threshold is
now computed relative to a cold-start-safe EMA of the loss
(`loss_ema * relative_multiplier`), with the old absolute value retained only as a
fallback floor for the first `ema_warmup_steps` observations, before the EMA carries
enough signal to be meaningful.

The state machine below (`update_liquid_spike_state`) is unchanged — only the
threshold handed to it changes. Its callers now compute that threshold via
`effective_liquid_spike_threshold()`.
"""

from __future__ import annotations

from typing import Tuple


def update_loss_ema(
    *,
    loss_value: float,
    loss_ema: float,
    observations: int,
    decay: float,
) -> float:
    """
    Cold-start-safe EMA of the training loss.

    Mirrors the pattern already used by train/continual_adapter.py's
    ContinualLearningAdapter: on the FIRST observation the EMA is *seeded* with that
    loss rather than blended against a fictitious 0.0 init (which would bias the early
    steps toward zero and make every early step look like a spike).

    `observations` is the number of losses seen BEFORE this call.
    """
    if observations <= 0:
        return float(loss_value)
    d = float(decay)
    return float(loss_ema) * d + float(loss_value) * (1.0 - d)


def effective_liquid_spike_threshold(
    *,
    loss_ema: float,
    observations: int,
    absolute_threshold: float,
    relative_multiplier: float,
    ema_warmup_steps: int,
) -> float:
    """
    Resolve the spike threshold actually handed to `update_liquid_spike_state`.

    Cold start (`observations` < `ema_warmup_steps`): fall back to the historical
    absolute threshold — the EMA has not seen enough samples to be trustworthy yet.

    Warmed up: return `loss_ema * relative_multiplier`, i.e. "a spike is a loss well
    above where this run actually lives", independent of the absolute loss scale.

    `observations` is the number of losses folded into `loss_ema` so far.
    """
    if int(observations) < int(ema_warmup_steps):
        return float(absolute_threshold)
    return float(loss_ema) * float(relative_multiplier)


def update_liquid_spike_state(
    *,
    loss_value: float,
    threshold: float,
    counter: int,
    patience: int,
    frozen_until: int,
    global_step: int,
    cooldown_steps: int,
    enabled: bool,
) -> Tuple[int, int, bool]:
    """
    Update liquid spike state.

    Returns (new_counter, new_frozen_until, triggered_freeze).
    """
    if not enabled:
        return counter, frozen_until, False

    if frozen_until > global_step:
        # Already in cooldown; don't accumulate strikes.
        return 0, frozen_until, False

    new_counter = counter + 1 if loss_value > threshold else 0
    if new_counter >= max(1, int(patience)):
        new_frozen_until = global_step + max(1, int(cooldown_steps))
        return 0, new_frozen_until, True

    return new_counter, frozen_until, False
