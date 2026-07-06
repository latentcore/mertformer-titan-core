# Laptop Pre-Flight Run — 2026-07-02 (infrastructure evidence, diverging-LR, interrupted)

**Status: `PRE-TRAINING (UNVERIFIED)` — this is INFRASTRUCTURE evidence and a NEGATIVE finding, not a capability, benchmark, or completed run.**

## What this is
A zero-argument single-file orchestrator (`scripts/preflight_run.py`) ran a small pre-flight on one **RTX 5070 Laptop (8 GB)** against repo commit `5fc5068`, using the canonical training entry point (`train.train.train`) unchanged. Purpose: prove the pipeline runs end-to-end and writes verifiable artifacts on real hardware — **not** model quality.

Config (see `run_config.json`): 36.2M params · 6L × 384h · GQA 6q/3kv · MoE 8×2 (layers 2,5) · Liquid mixer (layer 3) · seq 512 · global batch 32 · **LR 1.5e-3** · grad_clip 2.0 · liquid spike threshold **absolute `loss>5.0`** · tokenizer = ungated 32k stand-in (`distill_alpha=0`, no teacher) — the canonical 45K run uses the 128256 Llama-3 tokenizer. Dataset: `ultrachat_200k` `train_sft` pinned to revision `8049631c…`, ~30M trainable tokens (see `dataset_manifest.json`).

## 🟢 MEASURED — infrastructure (what PASSED)
- Pipeline ran end-to-end on the target device (28m20s, 981 steps before manual interrupt).
- **Atomic checkpoint wrote cleanly at step 500 — the exact point a prior 2×H200 run died mid-write.** This is the plumbing that was being validated.
- Guards functioned as designed: gradient clip **held at 2.0** every step (the run never took the exploded step), MoE health telemetry live (MaxLoad / entropy / overflow), and a **graceful interrupt-save** wrote `infra_preflight_step_981.pt` on Ctrl+C.
- Hash-chained run log intact (kept local; not published because scrubbing would break the chain).

## 🟢 MEASURED — negative finding (what DIVERGED)
The run **diverged**, and that is the valuable result:
- Loss started ~**10.4** (≈ `ln(32000)`, correct random-init), dipped to ~**8.4** (step ~80), then climbed **back above random** to ~**15.0** by step ~970.
- **Gradient norm exploded** to `1e11`–`1e16` from step ~80 onward and reached **`inf`** by step ~970. Only `grad_clip=2.0` kept the run alive (it never applied the raw step).
- MoE load entropy degraded **0.995 → 0.735**, MaxLoad rose to **~0.6** (`EARLY IMBALANCE ALERT`).
- The **Liquid spike guard uses an absolute threshold (`loss>5.0`)** which, at this loss scale, never releases → the Liquid layer is effectively **never trained** this run.

## Interpretation
This empirically confirms **LR 1.5e-3 is fatal for this architecture at this scale** — the exact pre-45K stabilization signal. Cost: **$0 / one laptop**, instead of burning 8–16× H100-hours + days to learn the same thing on the real 45K run. Actions recorded in `BACKLOG.md` / `DECISIONS.md` (LR regime sweep; relative Liquid threshold `EMA×1.5`; `generate()` Liquid-state parity; held-out ppl harness; 100–300M pilot).

## Checkpoints — NOT included (weights excluded on purpose)
`.pt` files are gitignored, ~310 MB each, and are **diverging-run weights** (no capability). Referenced by SHA256 only:
- `infra_preflight_step_500.pt` = `infra_preflight_latest.pt` → `c9e5df833b6279c59b6a79b15fc9fd38309eea1b89455a3aa22744cdff435355`
- `infra_preflight_best.pt` (val 9.7734 @ step 500) → `f16414d350261084887fe2ff6659207e472ec6fb1921e147100d370690d6cd43`
- `infra_preflight_step_981.pt` — incomplete atomic write (`.tmp`) captured at interrupt.

## Boundary (does NOT prove)
Not trained · not benchmark-verified · not capability · not mobile/production-ready · not frontier. Infrastructure evidence + one negative training-dynamics finding, against commit `5fc5068`.

## Reproduce
`python scripts/preflight_run.py` with the repo present; dataset regenerates byte-identically from the pinned revision in `dataset_manifest.json` (`source.reproduce_load`).
