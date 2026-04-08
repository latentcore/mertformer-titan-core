# ADR-0003 Chess One-Click Delivery Runtime Contract

- Date: `2026-04-08`
- Owner: `repo maintainer`
- Status: `accepted`

## Context
- The chess onefile lane is the strongest repo-side product proof currently present in the working tree.
- It now includes Windows delivery hardening, runtime containment, Stockfish auto-fetch, and handoff artifacts.
- The repo needs an explicit decision that this lane is repo-side green while still separating it from post-run strength claims.

## Alternatives Considered
- Keep chess delivery status implicit in tests and scattered reports.
- Treat chess delivery as equivalent to measured strength proof.
- Freeze the runtime and delivery contract while keeping Elo and final strength claims outside the current repo-side closure.

## Decision
- The chess onefile lane is treated as repo-side delivery-ready and closure-ready.
- Runtime containment, canonical onefile sync, Stockfish auto-fetch, teaching contract, and Windows delivery artifacts are part of the repo-side closure scorecard.
- Real strength claims, trained checkpoints, and claim-grade benchmark outputs remain post-run evidence and are not implied by delivery readiness.

## Tradeoffs
- The repo can confidently ship the lane operationally, but it must continue to speak carefully about model strength.

## Rollback Impact
- Removing this decision would weaken handoff confidence and blur the meaning of the current chess readiness reports.

## Compatibility Impact
- Compatible with current tests, delivery scripts, and reports.

## Benchmark Impact
- Preserves the distinction between readiness signals and real measured strength.

## Product Impact
- Makes the chess lane a concrete proof-of-system deliverable without over-claiming quality.

## Ops Impact
- Operators can move directly to the actual run on target hardware with less ambiguity.

## Legal and Security Impact
- No direct change; this ADR mainly tightens truth boundaries around productized chess delivery.
