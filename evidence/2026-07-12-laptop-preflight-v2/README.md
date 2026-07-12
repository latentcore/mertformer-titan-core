# Laptop Pre-Flight v2 — 2026-07-12 (two runs, both diverged, second confirmation)

**Status: `PRE-TRAINING (UNVERIFIED)` — this is INFRASTRUCTURE evidence and a NEGATIVE finding, not a capability, benchmark, or completed run.**

## What this is

Two real RTX 5070 Laptop (8 GB) runs, both against the then-current default LR regime (`TITAN_LEARNING_RATE=3e-4`, the candidate fix from the 2026-07-08 stabilization pass and the 2026-07-02 divergence): `scripts/preflight_run.py` (36M) and `scripts/preflight_run_pilot171m.py` (171M pilot — the T1.5 Go/No-Go gate). Both self-contained launch bundles were built via `git archive` per the repo's §16.3 discipline (see the Master Protocol addendum). **Both runs diverged.** This is the second confirmation of the divergence class first seen on 2026-07-02, and it materially changed the diagnosis: full data in `metrics_summary.json` and the two scrubbed `console_*_full_scrubbed.log` files.

## 🟢 MEASURED — infrastructure (what PASSED, both runs)

- Both pipelines ran end-to-end on real target hardware (Windows 11, RTX 5070 Laptop, sm_120, 8 GB VRAM, torch cu128).
- Both overlay-verified configs matched their yaml specs exactly ("Overlay verified: cfg matches ... on every checked field").
- Both graceful-interrupted cleanly on Ctrl+C with a checkpoint write (`pilot171m_stabilization_step_563.pt` for the 171M run) — the atomic-checkpoint plumbing worked correctly under real interruption, same as 2026-07-02.
- 171M's Liquid EMA-relative spike guard (added in the 2026-07-08 stabilization pass, replacing the 2026-07-02 run's absolute `loss>5.0` threshold) never had a chance to prove itself here — the run was manually interrupted at step 563 of 3000, still short of `liquid_warmup_steps=667`, so Liquid never unfroze. 36M's Liquid spike guard DID trip correctly (3-strikes rule) and re-froze Liquid after its first unfreeze-triggered spike — this specific safety mechanism is confirmed working as designed.

## 🟢 MEASURED — negative finding (what DIVERGED, both runs)

- **36M:** diverged twice, both times correlated with a Liquid-unfreeze event (`liquid_warmup_steps=200`, an ad-hoc script-specific value that unfreezes Liquid BEFORE the LR warmup (274) and the divergence guard's arm point (also 274) complete — a real ordering bug specific to this smaller script, already avoided in the 171M pilot's carefully-derived `liquid_warmup_steps=667 > 450`).
- **171M:** diverged once, **without Liquid ever unfreezing** — ruling out Liquid-unfreeze-timing as the (sole) root cause. Onset instead correlates with LR crossing roughly 2.0e-4–2.5e-4 during warmup, in lockstep with MoE `Overflow`/`Balance(std)` degradation.
- In both runs, `clip=2.0` kept the *loss* signal partially protected even as `GradNorm` reached into the trillions — and in both runs the loss-only `divergence_guard` **never tripped**, because its trigger (`loss_ema > 1.5x` a warmup-end reference) never got exceeded even during unambiguous catastrophic instability. This is a real, newly-documented blind spot (see `BACKLOG.md`) — not fixed here.
- Full step-by-step GradNorm/loss/MoE-health tables: `metrics_summary.json`.

## Interpretation

The shared signature across both runs (GradNorm explosion + MoE routing degradation + loss-protected-by-clip + guard-never-trips), at two different scales and two different proximate triggers, points at a common underlying cause rather than two unrelated bugs: the router z-loss regularizer's effective weight was **2e-6** (a `1e-4 × 0.02` double-multiply in `layers/moe.py`/`train/train.py`, ~50x below the ~1e-3 Switch-Transformer/ST-MoE convention) — confirmed directly in both runs' own `run_config.json`. This session applied a candidate fix (`config/config.py:329`, `z_loss_coef` `1e-4` → `0.05`, landing the effective weight at 1e-3 without restructuring the double-multiply itself) — **unverified on real hardware**, since no further preflight/pilot re-run was performed before the real 45K launch (explicit user decision). The next real GPU run is the verification event, same discipline as the existing F1/F3 candidates.

## Checkpoints — NOT included (deleted at the source, per this session's storage policy)

Both runs' checkpoints were deleted from the Windows laptop by the user before/without being hashed, matching the repo's own established precedent for small validation runs (`evidence/2026-07-02-laptop-preflight/README.md`: "checkpoints excluded — `.pt` gitignored, ~310MB, diverging-run weights — referenced by SHA256 only"). No SHA256 reference is available here since the files were removed before this evidence pass began; this is consistent with the policy (the value of these runs is in the logs/diagnosis, not the diverging-run weights).

## Boundary (does NOT prove)

Not trained · not benchmark-verified · not capability · not mobile/production-ready · not frontier. Two infrastructure-evidence + negative-training-dynamics findings, against repo commit `8e8978f` (both launch bundles), processed and diagnosed against repo HEAD as of this pass.

## Reproduce

`python preflight_run.py` (36M) / `python preflight_run_pilot171m.py` (171M) from their respective self-contained launch bundles; both auto-install missing deps and stream the pinned `ultrachat_200k` dataset revision on first run.
