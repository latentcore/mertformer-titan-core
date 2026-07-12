# Liquid/CfC Keep-or-Drop Decision Brief

**Purpose:** BACKLOG I.2 #15 — "Liquid/CfC 45K koşusunda kalacak mı — açık PRE-45K kararı" (will Liquid/CfC stay in the 45K run — an open pre-45K decision).

## The measured evidence (already in the repo, not new)

Per [ABLATION.md](../ABLATION.md) / [STATUS.md](../STATUS.md)'s own summary: a 12-seed Liquid ablation at toy scale found **no measured accuracy benefit** from the Liquid/CfC mixer layers, and the Liquid path runs **~30% slower** than the equivalent dense path. The verdict recorded in-repo is explicit: "ölçülen doğruluk faydası yok, ~%30 daha yavaş, toy ölçekte sonuçsuz; hız iddiası yok" (no measured accuracy benefit, ~30% slower, inconclusive at toy scale; no speed claim).

## The tension

- **Argument to keep:** the toy-scale ablation is explicitly labeled inconclusive at that scale — Liquid/CfC recurrent mixing is a genuinely different inductive bias (continuous-time state dynamics) than attention/MoE, and whether it helps could plausibly only show up at the canonical 3.67B scale with real data, not a 12-seed toy ablation. Dropping it now forecloses ever finding out under the real architecture. It is also already fully implemented, tested (parity tests for `generate()` state-threading, Gate 3 DDP unfreeze guard, MPS dropout fix — all closed this session), and load-bearing in the "BitNet+MoE+Liquid+GQA" architecture identity the whole project is built around.
- **Argument to drop (for the 45K run specifically):** ~30% slower is a real, measured cost that multiplies directly against the 45K run's GPU-hour budget — at 8-10 hours on 2xH100/H200 (BACKLOG I.1 #4), a 30% slowdown is a real number of extra GPU-hours with **zero measured benefit to point to** as justification. The 45K run is a single, expensive, one-shot event (checkpoint-bound, no re-runs planned) — spending its budget on an architectural component with no evidence of payoff is a real risk, not a hypothetical one.

## What this brief does NOT resolve

Whether "Liquid/CfC is part of what MertFormer *is*" (identity/thesis argument) outweighs "Liquid/CfC has no measured benefit and a real, measured cost" (evidence argument) is a judgment call about the project's own goals, not something more data alone settles — the 12-seed ablation already IS the available data, and it's inconclusive by its own admission.

## Recommendation (not a decision — Mert's call)

If the goal of the 45K run is primarily to validate the FULL architecture-as-designed (including Liquid) end-to-end, keep it — that is what BACKLOG's own framing of the pilot ("BitNet+MoE+Liquid+GQA all present") already assumes. If the goal is to conserve GPU-hours and Liquid's payoff is genuinely unproven, dropping it for the 45K run and treating the Liquid ablation as a **separate, future, cheaper** scaling study (rather than bundling it into the one expensive canonical run) is the more evidence-conservative choice. Either way, the decision should be recorded in [DECISIONS.md](../DECISIONS.md) once made, exactly as the divergence-guard addition already was.
