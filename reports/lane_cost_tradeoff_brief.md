# Lane Cost Tradeoff Brief — `online_teacher` vs `offline_clean`

**DECIDED (2026-07-19): `online_teacher`.** Mert chose the higher-GPU-hour lane over `offline_clean` via an explicit multiple-choice decision — matches `scripts/launch_ocean_45k.sh`'s existing default, no script change needed. Reasoning accepted: corpus materialization is the larger current blocker regardless of lane, so not adding a second sequential precompute dependency on top of it is coherent. See `DECISIONS.md` "Eight launch-time decisions locked...". Both lanes still fail readiness today for unrelated reasons (`offline_clean`: `PRECOMPUTED_LOGITS_MISSING`; `online_teacher`: `MISSING_HF_TOKEN`) — this decision picked the lane, it did not clear its blockers. The analysis below is preserved as the reasoning that informed the decision, not superseded by it.

**Purpose:** BACKLOG I.1 #3 — "online_teacher mi offline_clean mi? GPU-saatini 3-6x'e katlayan tek karar" (the single decision that multiplies GPU-hours 3-6x). This brief lays out the real tradeoff so the decision is made deliberately, not by default.

## The two lanes (as implemented today)

| | `scripts/launch_8xb300.sh` (`offline_clean`) | `scripts/launch_ocean_45k.sh` (`online_teacher` / `remote_bootstrap`) |
|---|---|---|
| `TITAN_USE_PRECOMPUTED_LOGITS` | `1` | `0` |
| Teacher access | None at train time — logits precomputed once, offline, ahead of the run | Gated 70B teacher (`meta-llama/Llama-3.3-70B-Instruct`) loaded live every step |
| `TITAN_REQUIRE_GATED_TEACHER` | `1` (but only for the precompute phase, not the 45K run itself) | Implicit — the run needs live HF_TOKEN access throughout |
| Extra compute cost | One upfront precompute pass (Phase-0, `scripts/precompute_logits_topk.py` / `precompute_logits_parallel.py`) across the full 23.6B-token corpus, done once, cacheable | Teacher forward pass on every single training step, for the whole 45K run |
| Failure mode if teacher access lapses mid-run | None — offline once precomputed | Run stalls/fails if HF_TOKEN or teacher endpoint access is interrupted |
| Current blocker | `PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE` (STATUS.md) | `MISSING_HF_TOKEN` (STATUS.md) |

## Why the multiplier is real, not hypothetical

A 70B-parameter teacher forward pass is roughly comparable in FLOPs to a meaningful fraction of the 3.67B student's own forward+backward pass per step. Running it live on **every** of the 45,000 steps (`online_teacher`) means paying that teacher-inference cost 45,000 times. `offline_clean` pays it **once**, amortized across a single precompute pass, then the 45K run itself only touches cached top-k logits (cheap disk reads, no extra GPU-side teacher compute). This is the literal source of the "3-6x" multiplier depending on teacher batch size / precompute parallelism assumptions — it is not a vague heuristic, it follows directly from "N steps × teacher cost" vs "1 pass × teacher cost".

## What actually blocks each lane today

- `offline_clean`: Phase-0 precompute has not been run (no real corpus materialized, no precomputed logit shards on disk yet — see BACKLOG I.3 #21/#25). This is compute+time to unblock, not a design gap; the code path (`precompute_logits_topk.py`, `precompute_logits_parallel.py`, `validate_logit_alignment.py`) is closure-complete and tested (ADR-0005 naming-mode guard fixed this session).
- `online_teacher`: needs a live `HF_TOKEN` with access to the gated `meta-llama/Llama-3.3-70B-Instruct` repo, provisioned at launch time (BACKLOG I.4 #29), plus sustained access for the run's full duration.

## Recommendation (not a decision — Mert's call)

Given the 3-6x cost multiplier is the dominant factor in total GPU spend, and `offline_clean`'s blocker (materializing the precompute) is a one-time, boundable cost while `online_teacher`'s blocker (sustained gated-teacher access) is an ongoing dependency for the entire run duration, `offline_clean` is the lower-risk, lower-cost default **once Phase-0 precompute is actually run**. `online_teacher` remains the fallback if precompute proves infeasible (disk, time, or access constraints on the precompute step itself).

**This brief does not make the decision — it exists so the decision, whenever made, is made with the real numbers in front of it.**
