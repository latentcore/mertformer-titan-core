# Final Truth Constitution

## Current Pass Objective
- Close the repository for the 45K architecture validation run.
- Prefer verifiable outputs over speculative redesign.
- Keep the repo honest about what is implemented now versus what only becomes true after the real run.

## Claim Modes
- `measured`: current artifact-backed fact.
- `target`: planned or estimated behavior, not yet verified.
- `vision`: long-range direction outside current evidence scope.
- `policy`: repository rule or restriction.

## Status Modes
- `DONE_NOW`: implemented now with exact repo evidence.
- `PREPARED_FOR_POSTRUN`: infrastructure exists, but final proof appears only after the real 45K run.
- `PHASE2`: explicitly deferred and not required for 45K readiness.
- `OUT_OF_SCOPE`: not part of the current closure mandate.
- `EXTERNAL_DEPENDENCY`: blocked by external data, credentials, compute, or the real run itself.

## Hard Rules
- No claim without evidence.
- Do not say `45K-ready` unless the current readiness report says `TRAIN_ALLOWED`.
- A closure-critical item is only `done` when code path, canonical command, verification, and artifact/report evidence all exist together.
- Docs-only closure is forbidden.
- Do not convert scaffolds, placeholders, historical snapshots, or plans into completed work.
- Do not use historical audit files as current truth unless the current source-of-truth files explicitly point back to them.
- Keep measured vs target vs vision language explicit in README, model card, policy files, and prompts.

## Code-Truth Maturity Labels
- `reference_safe`: correctness-first reference or scaffold path that is safe for parity/debug use, not for production-depth speed claims.
- `tested_fallback`: deterministic or bounded implementation with test coverage, but not a release-grade performance claim surface.
- `optimized_production`: measured and release-grade optimized path backed by claim-grade evidence.

## Surface Lifecycle Classes
- `frozen`: rules, schemas, naming, source-of-truth order, and release-truth constraints.
- `maintained`: verification gates, manifests, and reproducibility or handoff surfaces that must stay current but not churn without reason.
- `living`: training, benchmark, kernel, export, product, chess, packaging, security, legal, and pilot implementation surfaces.

## Research-Lane Rule
- `3000+ Elo`, `20 ms/move`, `10000x speedup`, AGI/ASI language, and long-context moonshots remain research lanes unless measured evidence explicitly upgrades them.

## Release-Truth Gates
- `bash scripts/verify_all.sh`
- `bash scripts/one_command_full_sop.sh`
- `bash scripts/final_one_shot.sh`
- `python3 scripts/build_train_readiness_contract.py --allow-not-ready`
- `python3 scripts/build_closure_governance_pack.py`
- `bash zero_touch_start.sh --check-only`

## 45K Readiness Rule
The current repo-side readiness verdict is `TRAIN_ALLOWED` via the offline-clean lane. The remaining exact blocker is the optional `online_teacher:MISSING_HF_TOKEN` lane when gated teacher access is intentionally requested.

## Post-Run Rule
Trained weights, checkpoints, benchmark summaries, demo bundle, evidence pack, and measured deployment outputs are not current facts until the real 45K run produces them.
