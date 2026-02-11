# Security & Safety Policy

## Scope
This document describes safety boundaries for MertFormer Titan and its training/evaluation pipeline.

## Usage Boundaries
- Do not use the model for malicious, harmful, or unlawful purposes.
- Do not upload sensitive or regulated data to external services during training.

## Kill-Switch & Failure Budget
- Numerical instability is handled via kill-switch drills and failure budget guards.
- See: `scripts/nan_kill_test.py`, `orchestrator/failure_budget.py`.

## Data Handling
- Respect dataset licenses and provenance.
- Do not redistribute datasets unless explicitly permitted by license.

## Reporting
- Record incidents in `postmortems/` using the provided template.
- Update mitigation steps after resolution.

## Status
This policy is a **baseline template** and will be updated after production runs.
