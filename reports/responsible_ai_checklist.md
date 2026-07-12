# Responsible-AI Checklist + Go/No-Go

**Purpose:** BACKLOG I.7 #74 — checkpoint-independent, fillable now. This is a checklist, not a completed audit — items are marked per their actual current status, not aspirationally.

| Area | Status | Evidence / next step |
|---|---|---|
| Training-data provenance/consent | ⚠ Partial | `reports/data_provenance_audit_template.md` (this pass) — corpus is `bigcode/the-stack-dedup` (code, permissively-licensed per its own upstream filtering); formal audit still to be run |
| Training-data poisoning risk | ✅ Analyzed | `reports/data_poisoning_risk_analysis.md` (this pass) |
| Bias/fairness | ⚠ Probe-ready, not run | `eval/bias_fairness_probe.py` (this pass) — SKIPPED pending checkpoint |
| Toxicity | ⚠ Probe-ready, not run | `eval/toxicity_probe.py` (this pass) — SKIPPED pending checkpoint |
| Hallucination rate | ⚠ Probe-ready, not run | `eval/hallucination_rate_probe.py` (this pass) — SKIPPED pending checkpoint |
| Calibration/abstention | ⚠ Probe-ready, not run | `eval/calibration_ece.py` (this pass) — SKIPPED pending checkpoint |
| Adversarial robustness | ⚠ Probe-ready, not run | `eval/adversarial_prompt_robustness.py` (this pass) — SKIPPED pending checkpoint |
| Membership inference / privacy | ⚠ Probe-ready, methodology-only | `eval/membership_inference_probe.py` (this pass) — synthetic-split sanity check only, no real train-set access |
| Prompt-injection / input sanitization | ✅ Audited this pass | see BACKLOG #82 disposition — `scripts/chat.py`/`mertformer_sdk/api.py` reviewed |
| Carbon footprint | ✅ Calculator ready | `scripts/estimate_carbon_footprint.py` (this pass) — needs real GPU-hours once run completes |
| Claim-boundary discipline | ✅ Established, enforced by CI | `scripts/check_doc_claim_consistency.py`, `scripts/check_facts_drift.py` (this pass), forbidden-language regime |
| Independent external review | ❌ Not started | `reports/independent_signoff_template.md` (this pass) — blank template, real pentest is human/external |
| Watermarking | ❌ Not decided | see `reports/post_45k_research_agenda.md` #80 |
| Model-inversion risk | ⚠ Probe-ready | shares methodology with membership-inference probe above |
| SDK rate limiting | N/A | see BACKLOG #82/#83 disposition — local-inference SDK, no server surface, rate-limiting does not apply |

## Go/No-Go framing

**This checklist is not itself a go/no-go gate for the 45K run** — none of the ✅/⚠ items above block training (they are evaluation/governance items, not training-safety items; the actual training-safety gates are `train_allowed`/`start_gate.py`/the divergence guard, which are separate and already enforced). This checklist IS a go/no-go gate for any **release/publication** claim (model card, HuggingFace upload, blog post) — no ⚠/❌ item above should be silently treated as ✅ in outward-facing material.
