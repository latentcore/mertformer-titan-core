# Training-Data Provenance/Consent Audit Template

**Purpose:** BACKLOG I.7 #75 — checkpoint-independent, partially fillable now against the currently-pinned corpus source.

## Current corpus (as pinned, per Pass-4 commit `21366d6`)

| Field | Value | Status |
|---|---|---|
| Source dataset | `bigcode/the-stack-dedup` | pinned by revision + sha256 |
| License basis | Upstream filters to permissively-licensed repositories (per BigCode's own stated methodology) | inherited, not independently re-verified by this repo |
| Consent mechanism | Upstream opt-out process (BigCode's own `am-i-in-the-stack` tooling) | inherited, not independently re-verified by this repo |
| PII scrubbing | Upstream PII redaction pass (per BigCode's stated pipeline) | inherited, not independently re-verified by this repo |
| Re-pin discipline | Revision + sha256 re-pinned on the training machine (BACKLOG #24, closed) | ✅ done |

## What "independently re-verified" would require (not done, scoped here)

1. Sample-audit a statistically meaningful random subset of the pinned corpus snapshot directly (not trust the upstream description alone) for: license-header presence/consistency, residual PII patterns, obviously-excluded-category leakage (per BigCode's own exclusion list).
2. Cross-check the pinned revision's sha256 against BigCode's own published manifest at the time of the Pass-4 re-pin, to confirm no supply-chain substitution occurred between BigCode's publication and this repo's fetch.
3. Document the sample size, methodology, and findings here, replacing this template with actual audit results.

## Why this is a template, not a completed audit

A real provenance/consent audit requires either (a) direct sampling of the actual multi-terabyte corpus snapshot (not available in this offline development environment — the corpus is fetched at train time, not present in this repo) or (b) trusting BigCode's own published methodology without independent verification. Neither is something this pass can complete without the actual corpus present. This template exists so the audit has a defined shape and can be filled in as soon as the corpus is materialized (BACKLOG #21), rather than being invented from scratch at that point.
