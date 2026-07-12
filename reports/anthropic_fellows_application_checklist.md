# Anthropic Fellows Application-Readiness Checklist

**Purpose:** BACKLOG I.6 #59 — "Anthropic Fellows = birincil kısa-vade rampası" (Anthropic Fellows = primary near-term ramp). Preparation checklist grounded in the actual current repo state, not generic advice.

| Readiness item | Status | Note |
|---|---|---|
| A real, working, non-trivial ML system to point to | ✅ | This repo — 3.67B measured-param architecture, closure-complete pre-training infra |
| Evidence-discipline as a demonstrated practice, not a claim | ✅ | This entire session: 5 real bugs found+fixed with red/green test proof, repo-wide "cosmetic-fix disease" sweep, forbidden-language regime enforced by CI |
| A concrete, honest account of what DIDN'T work | ✅ | `reports/blog_liquid_ablation_draft.md` — the Liquid/CfC negative result |
| Public artifact reviewers can actually look at | ⚠ | Public gist exists (`dac0aa0c...`) with synced test-count/evidence pointers; repo itself is private |
| A trained checkpoint / real results | ❌ | The one remaining real gap — pre-45K |
| Written technical narrative (blog/paper) | ⚠ | Drafts prepared this pass (`reports/blog_liquid_ablation_draft.md`, `reports/paper_outline_draft.md`), not yet polished/published |
| Clear articulation of research taste / judgment | ⚠ | Demonstrated implicitly (the Liquid ablation decision-making, the divergence-guard tradeoff analysis) but not yet written as an explicit "here's how I think about tradeoffs" narrative |

## What to lead with (given current state)

Given no trained checkpoint exists yet, the honest, defensible pitch is: **evidence discipline and engineering judgment as the differentiator**, not capability claims. Concretely:
1. The Liquid/CfC ablation (measured negative result, honestly reported and acted on).
2. This session's bug-hunting methodology (finding sophisticated "looks-fixed-but-isn't" bugs — getattr/hasattr dead-attribute patterns, dummy-tensor fake checks — and building permanent CI gates so the SAME bug class can't silently recur).
3. The explicit claim-boundary infrastructure (`reports/FACTS.json`, forbidden-language CI gates, `⚠ DÜZELTME` correction-blockquote discipline applied to the project's OWN historical closure claims when they turned out to be wrong) — this is arguably the most Anthropic-relevant artifact in the whole repo, since it's a working demonstration of "notice when your own past claims were wrong, and fix the record rather than letting it stand."

## What NOT to lead with

Architecture novelty claims (BitNet+MoE+Liquid+GQA composition) without the 45K run to back them — the repo's OWN discipline (STATUS.md, TRUTH_MATRIX.md) already refuses to make capability claims pre-training; the application pitch should hold the same line.

## Next concrete step

Not a code task — this is a human action (BACKLOG #59 is filed under I.6, Publication/Career, "İNSAN-DIŞI" for the actual application submission). This checklist's job is to make sure the submission, whenever made, draws on real, current, defensible material.
