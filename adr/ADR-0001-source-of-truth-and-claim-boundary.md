# ADR-0001 Source of Truth and Claim Boundary

- Date: `2026-04-08`
- Owner: `repo maintainer`
- Status: `accepted`

## Context
- The repo already contains many closure, release, and evidence documents.
- Without a clear authority order, future edits can re-open solved surfaces or let prose outrun evidence.
- This closure pass needs a stable rule for what is authoritative now versus what remains post-run or external.

## Alternatives Considered
- Keep using README plus scattered reports as informal truth.
- Freeze only the raw closure matrix and let humans interpret the rest.
- Establish a generated source-of-truth map plus claim registry plus known-limits boundary.

## Decision
- Use `AGENTS.md` plus the generated governance pack as the active authority chain for repo-side closure.
- Treat `reports/source_of_truth_map.md`, `reports/final_truth_constitution.md`, `reports/final_truth_matrix.md`, `reports/known_limits_v1.md`, and `reports/repo_closure_scorecard.md` as the current repo-side truth layer. (`reports/final_truth_constitution.md` is generated locally by the closure ladder — `bash scripts/verify_all.sh` — and is intentionally not version-controlled; regenerate it locally if absent when browsing on GitHub.)
- Separate repo-side closure truth from post-run evidence, external validation, legal sign-off, and release claims.

## Tradeoffs
- More governance files exist and must stay synchronized.
- In return, the repo gets a repeatable truth boundary instead of ad-hoc narrative drift.

## Rollback Impact
- Removing this ADR would re-open ambiguity around what can be claimed from the current working tree.
- A rollback would require a superseding ADR and a regenerated governance pack.

## Compatibility Impact
- Compatible with the current verify flow because the governance pack is generated and refreshed through existing scripts.

## Benchmark Impact
- None directly.
- Indirectly reduces the risk of presenting internal proxy signals as final measured benchmark truth.

## Product Impact
- Clarifies that current repo-side readiness and chess delivery hardening are real, while trained quality claims remain post-run.

## Ops Impact
- Operators get a stable authority order and a durable resume point for future closure passes.

## Legal and Security Impact
- Supports safer claim discipline by keeping legal, security, and external claims outside blind automation.
