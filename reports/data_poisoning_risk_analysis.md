# Training-Data Poisoning Risk Analysis

**Purpose:** BACKLOG I.7 #84 — checkpoint-independent, fully analyzable now against the current pipeline design (no corpus sampling required for this analysis, unlike the provenance audit).

## Attack surface as currently designed

1. **Source-level poisoning** (malicious content injected upstream into `bigcode/the-stack-dedup` before this repo fetches it): mitigated by pinning a specific revision + sha256 (BACKLOG #24) — a supply-chain substitution AFTER the pin would change the sha256 and be caught by the re-pin verification step, not silently accepted.
2. **Fetch-time substitution** (a man-in-the-middle or compromised mirror serving different content under the same nominal revision): mitigated by the sha256 pin itself (content-addressed, not just revision-tag-addressed) — this is the standard defense against exactly this attack class.
3. **Teacher-logit poisoning** (if `online_teacher` lane is used — BACKLOG #3/`lane_cost_tradeoff_brief.md` — a compromised or adversarially-fine-tuned teacher could inject biased distillation targets): **not currently mitigated**. The teacher is a trusted, gated HuggingFace model (`meta-llama/Llama-3.3-70B-Instruct`), and the pipeline has no independent verification that its outputs are unmanipulated beyond trusting the HF-hosted weights' own integrity. This is a real, currently-unaddressed gap — flagged here rather than silently assumed away.
4. **Mixture-freeze violation** (a second, unvetted source added after Stage-3 freeze — see `reports/stage3_mixture_freeze_proposal.md`): mitigated by the freeze proposal + DECISIONS.md recording discipline, PROVIDED the freeze process is actually followed at launch time (process discipline, not a technical control).

## What this analysis does NOT cover

- Statistical data-poisoning detection (e.g. outlier/anomaly detection within the corpus itself) — this would require sampling the actual corpus content, same limitation as `reports/data_provenance_audit_template.md`.
- Backdoor/trigger-pattern detection in the trained checkpoint itself (post-hoc, checkpoint-bound — not possible pre-45K).

## Recommendation

The sha256-pinning discipline already in place (BACKLOG #24) is a real, standard, adequate mitigation for source-level and fetch-time poisoning — no additional action needed there. The teacher-integrity gap (#3 above) is real and currently open; whether it's worth addressing (e.g. spot-checking teacher outputs against a known-good reference before trusting them as distillation targets) is a genuine, unresolved question that should be weighed against the `online_teacher` vs `offline_clean` lane decision itself — `offline_clean`'s one-time precompute pass is inherently easier to spot-check than `online_teacher`'s continuous live generation, which is one more argument in `offline_clean`'s favor beyond the cost multiplier already covered in `reports/lane_cost_tradeoff_brief.md`.
