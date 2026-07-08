"""
General loss-divergence circuit breaker (relative, finite-loss).

[2026-07-08] NOT a BACKLOG item — proposed and added by the 2026-07-08 pre-45K
stabilization pass. Flagged as an addition rather than a pre-approved backlog fix.

WHY THIS EXISTS
---------------
Before this, train/train.py had exactly two divergence brakes:

  * the NaN/Inf brake — hard safety-brake after `max_consecutive_nan` non-finite losses;
  * the Liquid spike guard — freezes only the Liquid params, never stops the run.

Neither catches "the loss is perfectly finite but climbing steadily", which is exactly
what the 2026-07-02 laptop pre-flight did: 10.4 -> 8.4 (best) -> 9.6 -> 11.8 -> 15.0,
above the random-init baseline, grad_norm sustained at 1e11-1e16, never once NaN. A
human had to eyeball the loss curve to notice. On an unattended 45K run that is days
and hundreds of dollars of compute burned after the run is already dead.

DESIGN
------
`orchestrator/failure_budget.py` already contains an unused `FailureBudget` /
`LossSlopeTracker` pair doing almost exactly this. That module is inert/out-of-scope
and stays where it is (see the repo rule: inert code gets real bugs fixed, not
promoted). This is a deliberately lighter, *deterministic* port of the same idea:

  * `FailureBudget` keys off wall-clock (`time.time()`) slope-per-hour, which makes it
    non-reproducible across machines and untestable without freezing the clock.
  * This guard keys off the run's own loss EMA compared against the EMA snapshotted at
    the END OF WARMUP — i.e. "is the loss now materially worse than where the schedule
    handed it off?" — which is scale-free, hardware-free, and exactly reproducible.

A breach must be sustained for `patience` consecutive checks before braking, so a
single noisy micro-batch cannot kill a healthy run.

Pure functions only: no torch, no config import, no I/O. Callers own the state.
"""

from __future__ import annotations

from typing import Optional, Tuple


def update_divergence_guard_state(
    *,
    loss_ema: float,
    reference_ema: Optional[float],
    multiplier: float,
    breach_counter: int,
    patience: int,
    enabled: bool,
) -> Tuple[int, bool]:
    """
    Update the relative loss-divergence guard.

    Args:
        loss_ema: current cold-start-safe EMA of the training loss.
        reference_ema: the loss EMA snapshotted at the end of warmup. `None` until
            warmup completes -> the guard is inactive (returns a reset counter).
        multiplier: brake when `loss_ema > reference_ema * multiplier`.
        breach_counter: consecutive breaches observed so far.
        patience: consecutive breaches required before the brake trips.
        enabled: master switch (cfg.use_divergence_guard).

    Returns:
        (new_breach_counter, triggered)

    A non-positive or missing `reference_ema` disables the check: a reference of <= 0
    would make the relative comparison meaningless (any positive loss "breaches" it).
    """
    if not enabled or reference_ema is None or float(reference_ema) <= 0.0:
        return 0, False

    breached = float(loss_ema) > float(reference_ema) * float(multiplier)
    new_counter = int(breach_counter) + 1 if breached else 0
    if new_counter >= max(1, int(patience)):
        return new_counter, True
    return new_counter, False
