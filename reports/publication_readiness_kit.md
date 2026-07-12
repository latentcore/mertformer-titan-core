# Publication Readiness Kit — Model Card, Demo Storyboard, Outreach Templates

**Purpose:** consolidates BACKLOG I.6 #53 (HuggingFace model card + launch post), #55 (demo packaging + video storyboard), #57 (Reddit/HN/email outreach templates). All are `[PENDING 45K RUN]` for their actual numbers; the structure/copy is draftable now.

## 1. HuggingFace Model Card Skeleton

```markdown
---
license: other  # Proprietary & Confidential (decided 2026-06-17, see LICENSE) — HF's "other" tag + this note; not Apache-2.0
tags: [bitnet, moe, mixture-of-experts, liquid-neural-network, gqa]
---

# MertFormer Titan

## Model Description
BitNet b1.58 ternary-quantized transformer with sparse Mixture-of-Experts
(8 experts, top-2, every 3rd layer), Liquid/CfC recurrent mixer layers, and
grouped-query attention. Measured runtime parameters: 3,672,982,022 (~3.67B).

## Intended Use
[PENDING — write only after real eval results exist; do not speculate]

## Training Data
[PENDING — bigcode/the-stack-dedup, revision/sha256: see reports/data_provenance_audit_template.md]

## Evaluation Results
[PENDING 45K RUN — every number here must cite a checkpoint SHA256, per this
repo's own FACTS.json discipline. Known negative result to include regardless
of what else runs: the Liquid/CfC ablation — no measured accuracy benefit,
~30% slower, see reports/blog_liquid_ablation_draft.md]

## Limitations
[PENDING — draw from STATUS.md's own limitation record + reports/responsible_ai_checklist.md]

## Citation
[PENDING]
```

## 2. Demo Video Storyboard (2-3 min, English)

1. **0:00-0:20** — Cold open: what MertFormer Titan is (one sentence), and the honesty framing ("this is pre-training closure, not a trained-model launch" if recording before 45K completes).
2. **0:20-0:50** — Architecture walkthrough: BitNet + MoE + Liquid + GQA, shown via a real terminal session instantiating the model and printing the measured param count live (not a slide — the repo can actually do this, `train/trainer_core.preflight_param_report()`).
3. **0:50-1:30** — The Liquid ablation story, told briefly: "we built it because we believed X, measured it, and it didn't hold up — here's the number." This is the differentiator moment.
4. **1:30-2:20** — `[PENDING 45K RUN]` real generation sample + held-out perplexity number, checkpoint SHA256 visible on screen.
5. **2:20-2:45** — Where to find the repo/gist/model card, closing.

## 3. Outreach Templates

### Reddit r/MachineLearning (post title + first paragraph)
> **Title:** [D] I ablated my own architecture's core idea and it didn't help — sharing the negative result
>
> I've been building a BitNet+MoE+Liquid-CfC+GQA language model (MertFormer Titan). Ran a 12-seed ablation on the Liquid/CfC recurrent mixer layers — the component I was most excited about — and found no measured accuracy benefit, ~30% slower. Full writeup: [link]. Curious if others have seen similar null results with continuous-time recurrent mixers in transformer hybrids.

### Show HN (title + first line)
> **Title:** Show HN: MertFormer Titan — a BitNet+MoE+Liquid transformer, with an honest ablation writeup
>
> Repo/gist link, one line: "Pre-training closure complete on a 3.67B-param hybrid architecture; the interesting part might be the negative ablation result, not the architecture itself."

### Cold-email template (career outreach, e.g. Anthropic Fellows contacts)
> Subject: MertFormer Titan — evidence-discipline engineering, not a capability pitch
>
> [Name], I'm reaching out about [program]. Rather than lead with an architecture pitch, I want to point to something more relevant: [link to blog_liquid_ablation_draft.md] — a negative result I measured and reported honestly, and [link to this session's closure work] — a pass where I found and fixed several "looks-fixed-but-isn't" bugs in my own prior work and built permanent CI gates so the same bug class can't recur silently. That's the working style I'd bring.

## Status of this kit

All copy above is a starting draft, not final — every `[PENDING 45K RUN]` marker must be resolved with a real number before publication, per this repo's forbidden-language regime. Actual posting/sending is a human action (I.6 items are filed as career/publication, out of code scope).
