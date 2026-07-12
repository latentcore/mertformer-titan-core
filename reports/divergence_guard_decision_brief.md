# Divergence Guard: Keep-or-Remove Decision Brief

**Purpose:** BACKLOG I.2 #18 — "divergence_guard varsayılan-AÇIK ama ön-onaylı DEĞİL — bilinçli koru/kaldır" (divergence_guard is default-ON but NOT pre-approved — a conscious keep/remove decision).

## What it is

`utils/divergence_guard.py` (ported from the previously-inert `orchestrator/failure_budget.py` slope tracker): trips `safety_brake_reason = "loss_divergence_relative"` after `divergence_guard_patience` (50) consecutive steps where the loss EMA exceeds 1.5x the EMA snapshotted at warmup end. The brake decision is all-reduced across ranks (mirrors the existing NaN-brake collective) to avoid DDP desync. Default: **ON** (`use_divergence_guard=True`); disable via `TITAN_DIVERGENCE_GUARD=0`.

## Why it's flagged as "not pre-approved"

Per BACKLOG.md's own T1.6 entry, this was added as part of the 2026-07-08 pre-45K stabilization pass but explicitly marked "NEW — not from this backlog" — meaning it was not one of the originally-scoped, pre-approved fixes for that pass. It is real, tested code (not cosmetic), but its *inclusion* in the canonical 45K run was never separately signed off the way the other T1.x items were.

## The tradeoff

- **Keep ON (default):** protects the single, expensive 45K run from silently burning GPU-hours on a diverged model — the exact failure mode the 2026-07-02 laptop preflight run itself exhibited (loss climbing rather than descending) before this pass's LR fixes. Given the run is checkpoint-bound and one-shot, an automatic stop-loss is a real safety net, not paranoia.
- **Risk of keeping ON:** the 1.5x-EMA-over-50-steps threshold is itself unvalidated at canonical scale — it was designed and reasoned about, but never fired-and-confirmed-correct against a real 3.67B training curve. A threshold that's too sensitive could trip on legitimate loss noise/warmup transients and kill a healthy run; the guard's own correctness is exactly as unverified as the LR regime it's meant to protect.
- **Risk of turning OFF:** returns to the pre-2026-07-08 status quo, where a diverging run silently burns the full GPU-hour budget with no automatic stop — which is the scenario that motivated adding the guard in the first place.

## Recommendation (not a decision — Mert's call)

Keep it ON for the 172.67M pilot run (BACKLOG T1.5) specifically **because** the pilot's whole purpose is validating the LR regime cheaply — if the guard's threshold is miscalibrated, the pilot is exactly where you want to find that out (a few dollars of GPU time, not $150-570). Before the real 45K launch, review the pilot's guard-trip behavior (did it fire correctly, falsely, or not at all) and use that as the actual pre-approval evidence, then record the keep/remove decision in DECISIONS.md the same way the guard's original addition was recorded.
