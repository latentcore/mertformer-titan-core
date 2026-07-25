# STATUS — MertFormer Titan

Canonical, hand-maintained status entry point. Generated detail lives under `reports/`
(this file is the single source a reviewer should read first). Turkish: [STATUS_TR.md](STATUS_TR.md).

## One glance
- **Stage:** pre-training **closure-complete** — the canonical model has **NOT been trained yet**.
- **Build:** `578 passed, 5 skipped` (offline-first `pytest`). See [REPRODUCE.md](REPRODUCE.md).
- **Readiness:** `train_allowed = true` · `decision_reason_code = READY_REMOTE_BOOTSTRAP` · `start_gate = START_ALLOWED`.
- **Crash-class bugs:** none (canonical model + orchestrator import cleanly).

## Architecture (measured, not benchmarked)
- 18 layers, hidden 2048, 16 heads / 8 KV (**GQA**), 8 experts top-2 (MoE every 3rd layer), Liquid/CfC mixer at layers [4,10,16], BitNet b1.58 ternary.
- **Measured runtime params:** `3,672,982,022` (~3.67B). **Design target:** 2.64B. Both labels are deliberate — see [reports/param_accounting_report.md](reports/param_accounting_report.md).

## What is measured vs not
- **Measured:** repo-side self-tests, offline smoke harness, the 12-seed Liquid ablation (see [ABLATION.md](ABLATION.md) — verdict: **no measured accuracy benefit, ~30% slower, inconclusive at toy scale; no speed claim**).
- **Not measured (the one real gap):** the canonical 3.67B model has never been trained — so "does it learn / converge / generalize" is **unverified**. This is hardware-bound, not a code edit.

## The single remaining blocker: a real 45K GPU run
- Needs H100/H200 + compute + days. Local K4 drills (checkpoint save→restore→resume) + import smoke are green.
- Lane blockers: `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`, `online_teacher:MISSING_HF_TOKEN`.
- Launch (target hardware): `bash zero_touch_start.sh`. See [REPRODUCE.md](REPRODUCE.md).
- **No intermediate-scale SCALING run** (250M / 500M) before or after — straight to the canonical 3.67B 45K (a smaller model has its own dynamics and does not predict 45K). The only retained pre-flight is the free `TITAN_MAX_STEPS=2` import/K4 smoke of the canonical architecture (not a sub-scale model). See [DECISIONS.md](DECISIONS.md).
  - *Not in conflict with BACKLOG.md's 2026-07-02 "100–300M pilot":* that pilot (`config/model/mertformer_pilot_stabilization.yaml`, measured 172.67M) is a **stabilization-only** LR/divergence safety check carrying **no capability claim and no scaling extrapolation**. The sealed decision forbids using a mid-scale run as a *capability/scaling proxy*; it does not forbid spending an hour proving a candidate learning rate keeps grad_norm finite before committing $150–570 and days of GPU to 45K.

## Canonical surfaces
- [TRUTH_MATRIX.md](TRUTH_MATRIX.md) — claim → evidence. · [BACKLOG.md](BACKLOG.md) — deferred work.
- [GOVERNANCE.md](GOVERNANCE.md) — policies/contracts index. · [REPRODUCE.md](REPRODUCE.md) — how to verify & launch.
- [DECISIONS.md](DECISIONS.md) — deliberate decisions (incl. documented-not-changed findings).
- Generated detail: `reports/closure_57_matrix.md`, `reports/repo_closure_scorecard.md`, `reports/final_truth_matrix.md`.

## Pre-45K stabilization pass — 2026-07-08 (candidate fixes, NOT verified)
The 2026-07-02 laptop-preflight run-feedback is now **implemented in code**: LR regime (`1.5e-3` → `3e-4`, router ×1.5 differential dropped, warmup 0.10 → 0.15, all env-sweepable), EMA-relative Liquid spike threshold, `generate()` Liquid-state threading + parity test, a checkpoint-bound held-out perplexity harness (`eval/held_out_ppl.py`), a measured 172.67M stabilization pilot config, a general loss-divergence circuit breaker, and a WSD-scheduler cosine clamp.

**These are candidate fixes pending re-verification on a real GPU run.** The divergence problem is *not* solved until a clean RTX-5070 re-run says so. No new measured-capability claim is introduced; readiness, claim boundaries and the frozen architecture are unchanged. See [BACKLOG.md](BACKLOG.md) and [DECISIONS.md](DECISIONS.md) for the full record, including one addition (the divergence guard) that was **not** a pre-approved backlog item.
