# 36M LM Re-Verify — 2026-07-25 (Kaggle T4×2, single-GPU) — DIVERGED, C1 grad-norm guard fired correctly

**Status: `RE-VERIFICATION RUN, FAILED infra-verdict (3/9), genuinely informative` — third real-hardware confirmation of the 2026-07-02/2026-07-12 divergence class at 36M scale, run against the full candidate fix set (LR 3e-4, z-loss 0.05, EMA-relative Liquid threshold, F1 CE-only divergence guard, F3 guard-state persistence, and the same-day C1 grad-norm co-trigger). The candidate fixes did NOT prevent divergence. The C1 safety brake — added earlier the same day as this run, commit `bc542c0` — correctly detected it and stopped the run cleanly with a valid checkpoint. This is a real, measured result, not a crash or infra bug (contrast with `evidence/` chess job note in `BACKLOG.md`, which was a genuine orchestrator bug).**

## What this is

BACKLOG's 36M LM preflight re-verification (`scripts/preflight_run.py`, zero-arg, 1830-step plan), run as job `03_lm_36m_reverify` of the same 5-job Kaggle batch as N3/N4, via `kaggle_batch_runner.py`. Single-GPU (the batch's own DDP smoke test ran first and correctly fell back to single-GPU — see `04_lm_171m_reverify`'s evidence for the shared smoke-test log). `TITAN_LEARNING_RATE=3e-4` (default, candidate not overridden further).

## 🟢 MEASURED — what happened (full detail in `console.log`, this folder)

- Preflight infra checks: all hard requirements PASS (CUDA, Tesla T4 14.6GB, bf16, disk, RAM).
- Dataset staged: 59,401 rows (29.98M trainable tokens) from `HuggingFaceH4/ultrachat_200k`, offline lock engaged.
- Training started normally (Stage 1, loss ~10.45 at step 10).
- **Step ~80-90:** gradient norm begins climbing past the soft-clip threshold (clip held at 2.0, but the underlying pre-clip norm is unbounded and logged).
- **Step ~110-130:** MoE early-imbalance alerts begin (`MaxLoad` crossing the 0.40 alarm threshold, reaching 0.72-0.76 by step 180-400) — the same MoE-imbalance signature seen in the 2026-07-12 divergence.
- **Step 200: Liquid layers unfreeze** (`🔓 UNFREEZING LIQUID LAYERS at Step 200!`). Gradient norm explodes from thousands to hundreds-of-millions within the next ~10 steps.
- **Step 274:** divergence guards arm (`ce_ema=8.5008`, `grad_norm_ema=6,701,995.21` — already ~6.7M, i.e. gradients were already deep into instability by the time the reference was captured).
- **Step ~410-416:** grad_norm_ema reaches 5.96e9. **`🛑 SAFETY BRAKE: relative grad-norm divergence — grad_norm_ema=5.9568e+09 exceeded 100.00x the warmup-end reference (6,701,995.21) for 10 consecutive optimizer steps.`** — this is the C1 co-trigger, added to `train/train.py` earlier the same calendar day (commit `bc542c0`), firing for the first time on real hardware.
- Clean shutdown: `infra_preflight_step_416.pt` and `infra_preflight_final.pt` checkpoints saved, `REPORT.md` written, output archive packaged and verified (828.8 MiB, 18 members) — **but this archive was not included in the consolidated batch output zip** (see Checkpoints section below) and is not present on this laptop.
- `preflight_run.py`'s own infra-verdict: **FAIL (3/9 infrastructure checks passed)** — this script's verdict is infra-completeness-only by design (`evaluate_verdict()`), not a training-quality judgment; a safety-brake stop before the full 1830-step plan legitimately fails most of its infra checks (final-step-reached, full-token-budget-consumed, etc.), even though the brake itself worked exactly as designed.

## 🔴 Negative finding — the candidate fixes did not prevent this divergence class

LR=3e-4, z-loss=0.05, and the EMA-relative Liquid threshold (all "candidate, unverified by dedicated GPU re-run" as of this morning) do **not**, on their own, prevent the MoE-imbalance-driven, Liquid-unfreeze-correlated divergence first seen 2026-07-02 and confirmed again 2026-07-12. This is the third real-hardware confirmation of this divergence class, now with the candidate LR/z-loss fixes actually in effect and still not sufficient. `BACKLOG.md`/`DECISIONS.md` need a real update reflecting this (see this pass's docs changes) — the LR/z-loss items move from "candidate, unverified" to "candidate, tested, insufficient alone."

## 🟢 Positive finding — the C1 grad-norm guard (same-day fix) worked exactly as designed

C1 was added to close a documented blind spot: the CE-based divergence guard (F1) can stay quiet while clip artificially bounds the *reported* loss even as *gradients* are actually exploding. This run is a genuine, same-day, real-hardware validation: the guard armed, tracked the exploding grad-norm EMA, and correctly tripped the safety brake at a 100x-over-threshold sustained divergence — producing a clean checkpoint save instead of an uncontrolled crash or a silent training-effort waste running to a NaN/inf wall. This moves C1 from "candidate, applied, unverified" to "candidate, applied, verified working on real hardware" (the underlying divergence itself is still not fixed — only the safety response to it is confirmed).

## Checkpoints — NOT retrievable from this laptop (a real orchestrator gap, disclosed honestly)

`infra_preflight_step_416.pt` and `infra_preflight_final.pt` were saved during the Kaggle run (see `console.log` lines ~532-533) and packaged into `preflight_run_output.zip` (828.8 MiB) **inside the job's own Kaggle working directory** — but `kaggle_batch_runner.py`'s `collect_job_output()` only copies a job's output into the consolidated batch zip when `status == "completed"`; this job's exit code was non-zero (the infra-verdict FAIL), so it was classified `"failed"` and excluded by that rule, even though the underlying event was a clean, intentional, informative stop with a valid checkpoint — not a crash or a genuinely half-finished run. **The checkpoint therefore cannot be hashed or referenced here** — it is not present anywhere on this laptop. It may still exist in the Kaggle notebook's own persisted Output if that has not been cleaned up; this was not checked as part of this pass. Recorded as a real design gap in `BACKLOG.md` (the completed-only collection rule should distinguish a clean safety-brake stop from a genuine failure/timeout in a future revision of `kaggle_batch_runner.py`) — not fixed retroactively in this pass, since the run already happened and there is nothing left to re-collect.

## Boundary (does NOT prove)

Not a claim that the 45K canonical run is safe or ready — divergence at 36M scale under the candidate fixes is a real negative signal that must inform the 45K launch decision, not be set aside. Not a claim about C1's behavior at 45K scale/duration (it has now been seen to work correctly exactly once, on a 36M-scale, single-GPU, T4 run). Does not change 45K readiness (`TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP / START_ALLOWED`) directly, but is directly relevant evidence for the launch-time judgment calls in `reports/launch_time_decisions_checklist.md`.

## Reproduce

`python scripts/preflight_run.py` (zero-arg) on a CUDA machine with `TITAN_LEARNING_RATE=3e-4` (current default).
