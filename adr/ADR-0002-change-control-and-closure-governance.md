# ADR-0002 Change Control and Closure Governance

- Date: `2026-04-08`
- Owner: `repo maintainer`
- Status: `accepted`

## Context
- The repo has strong verification and closeout scripts, but several closure-governance surfaces were still implicit.
- The current pass aims to permanently convert the repo-side closure count from eleven to twenty-four.
- That conversion is only defensible if the missing items become generated or otherwise durable repo truth, not one-off notes.

## Alternatives Considered
- Manually write the missing policy documents without integrating them into verify.
- Keep the missing items as chat-level interpretation.
- Generate the missing closure-governance docs through `scripts/build_closure_governance_pack.py` and treat them as authoritative outputs.

## Decision
- The repo-side closure surfaces for this pass are generated or pinned as durable artifacts.
- `scripts/build_closure_governance_pack.py` is the canonical generator for freeze, update-first, directory contract, automation boundary, change control, known limits, support policy, matrices, ADR index, and the repo-side scorecard.
- `scripts/verify_all.sh` remains the canonical refresh path for these outputs.

## Tradeoffs
- The repo now has a larger generated governance surface to maintain.
- In exchange, closure status becomes reproducible, auditable, and less dependent on memory or conversation context.

## Rollback Impact
- Reverting this decision would collapse the permanent repo-side closure count back into ambiguous manual interpretation.

## Compatibility Impact
- Compatible with current scripts, runbooks, and report refresh flow.
- Future governance changes should be done by extending the generator, not by ad-hoc edits across many files.

## Benchmark Impact
- No direct benchmark effect.
- Helps preserve the line between process-green state and trained benchmark evidence.

## Product Impact
- Strengthens handoff quality and release-readiness communication.

## Ops Impact
- Gives operators a stable SOP ladder and maintenance policy for closure-critical work.

## Legal and Security Impact
- Reinforces the rule that strategic, legal, and security decisions remain human-reviewed.
