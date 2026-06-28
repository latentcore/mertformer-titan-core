"""
Liquid safeguard helpers.

Pure-Python utilities to keep liquid spike tracking testable without importing
the full training stack.
"""

from __future__ import annotations

from typing import Tuple


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
