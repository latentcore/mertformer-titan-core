# arXiv Paper Outline Draft — MertFormer Titan

**Purpose:** BACKLOG I.6 #54 — "arXiv makalesi — TEK güçlü makale, 50 tane değil" (ONE strong paper, not 50). Outline only; sections that need real 45K-run results are explicitly marked `[PENDING 45K RUN]` rather than filled with placeholders — per this whole project's own forbidden-language discipline, nothing here should ever be filled in with an unearned number.

## Working title

*MertFormer Titan: A BitNet-Quantized, Sparse-MoE, Liquid-Augmented Language Model — Architecture, Ablations, and an Honest Account of What Didn't Work*

(The subtitle is deliberate — the paper's differentiator vs. a typical architecture paper is that it reports a real negative result, not just the wins.)

## 1. Introduction
- The bet: can BitNet b1.58 ternary quantization + sparse MoE + Liquid/CfC recurrent mixing + GQA compose into a coherent, trainable architecture at meaningful scale.
- Explicit scope statement up front: this is a **pre-training, one-run, checkpoint-bound** paper — not a multi-run scaling study, not a benchmark leaderboard entry.

## 2. Architecture
- BitNet b1.58 ternary weights — measured param accounting (`reports/param_accounting_report.md`), 3,672,982,022 measured runtime params vs. 2.64B design target, both reported and reconciled (the 3.67B/2.64B duality is itself worth explaining, not hiding).
- Sparse MoE: 8 experts, top-2, every 3rd layer.
- Liquid/CfC mixer layers: architecture description, LiquidRouter design, and — critically — the ablation result (Section 4) that this component shows no measured benefit at the scale tested.
- GQA (historically mislabeled "MLA" internally — corrected terminology, worth a footnote on why).

## 3. Training methodology
- LR regime and its own history: the 2026-07-02 divergence at `1.5e-3`, the fix to `3e-4` + warmup/router-multiplier changes, and the pilot-scale (172.67M) stabilization gate that validated the fix **[PENDING PILOT RUN RESULT]**.
- Optimizer: GaLore + 8-bit Adam.
- Distillation setup (teacher, KD-alpha, offline vs. online logit lane — see `reports/lane_cost_tradeoff_brief.md`).
- Divergence guard, WSD scheduler clamp — reliability infrastructure, framed as such (not as capability claims).

## 4. Ablations
- **The Liquid/CfC ablation** (already measured, see `reports/blog_liquid_ablation_draft.md` for the narrative version): 12 seeds, no measured accuracy benefit, ~30% slower. This section is the paper's most defensible content right now because it needs no future run to write.
- MoE load-balance behavior (entropy trajectory, EARLY IMBALANCE ALERT pattern) — reported as an observed, unresolved limitation, not silently omitted.

## 5. Results `[PENDING 45K RUN]`
- Held-out perplexity (`eval/held_out_ppl.py`), checkpoint-bound benchmark battery (HumanEval/MBPP/GSM8K, lm-eval-harness).
- Every number in this section requires a real checkpoint SHA256 cited alongside it — no number appears without its evidence pointer, mirroring `reports/FACTS.json`'s own discipline.

## 6. Limitations
- No intermediate-scale scaling run (250M/500M) — direct jump from pilot to canonical 3.67B, and why (STATUS.md's own reasoning: a smaller model has its own dynamics and doesn't predict 45K behavior).
- Liquid/CfC's toy-scale ablation result and its explicit non-generalization to canonical scale — including a 2026-07-31 external signal (see `BACKLOG.md`/`ABLATION.md`) that the measured ~30% wall-clock cost may not hold at the canonical sequence length, since the mixer's recurrence is sequential and attention's is not.
- Single-run, non-reproducible-by-repetition nature of the 45K training (checkpoint-bound, not statistically repeated).

## 7. Responsible AI / Ethics
- Point to `reports/responsible_ai_checklist.md`, `reports/data_provenance_audit_template.md`, `reports/data_poisoning_risk_analysis.md` (all prepared this pass) rather than duplicating their content inline.

## What's genuinely ready to write NOW vs. blocked

| Section | Status |
|---|---|
| 1, 2, 3, 4, 6, 7 | Draftable now from existing repo evidence |
| 5 (Results) | Blocked on the real 45K run — placeholder structure only, no numbers |

**One strong paper, written once results exist to report — this outline exists so that when they do, the paper isn't started from zero.**
