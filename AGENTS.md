# AGENTS.md

Project closure constitution for contributors and coding agents.

## Mission
- Treat this repository as a project-closure system for the 45K architecture validation run.
- Optimize for operational reliability, evidence quality, and clean handoff.
- Do not treat this repository as a research-expansion sandbox during closure work.

## Non-Goals
- No large architecture rewrite.
- No new product direction.
- No speculative AGI/ASI claims.
- No benchmark, latency, energy, or deployment claim without evidence.
- No feature creep for TurboQuant, multimodal, or unrelated experimental paths.

## Protected Core
- Preserve the BitNet + MoE + Liquid core.
- Preserve the MoE 8e2 decision unless a verified blocker explicitly requires a change.
- Keep experimental feature flags off on the main 45K path unless evidence and tests justify otherwise.
- Do not silently change teacher, tokenizer, or dataset policy.

## Canonical Commands
- Verification gate: `bash scripts/verify_all.sh`
- One-command closure flow: `bash scripts/one_command_full_sop.sh`
- Max closure refresh: `bash scripts/final_one_shot.sh`
- Readiness decision refresh: `python3 scripts/build_train_readiness_contract.py --allow-not-ready`
- Governance pack refresh: `python3 scripts/build_closure_governance_pack.py`

## Source-of-Truth Order
1. `AGENTS.md`
2. `reports/source_of_truth_map.md`
3. `reports/final_truth_constitution.md`
4. `reports/final_backlog_classification.md`
5. `reports/train_readiness_decision.md`
6. `README.md` / `README_TR.md`
7. `MODEL_CARD.md` / `MODEL_CARD_TR.md`
8. `USAGE_GUIDE.md` / `TRAINING_PLAN.md`

Historical snapshots, archived reports, and legacy audits are supporting context only unless the current source-of-truth files explicitly point back to them.

## Acceptance Criteria
- Code path exists and is wired into a real command or report.
- Tests or verification gates cover the changed behavior where practical.
- Machine-readable and human-readable artifacts exist for critical closure surfaces.
- Documentation points to the same canonical command and same current truth.
- No desktop absolute paths leak into tracked artifacts.
- Final readiness output keeps exact blocker reason codes.

## Required Writing Rules
- `measured`: backed by a current artifact, manifest, benchmark, or log.
- `target`: planned or estimated behavior, not yet verified.
- `vision`: long-range direction outside current evidence scope.
- `DONE_NOW`: implemented now with exact repo evidence.
- `PREPARED_FOR_POSTRUN`: infrastructure is in place but the final evidence only appears after the real 45K run.
- `PHASE2`: intentionally deferred and not required for 45K readiness.
- `EXTERNAL_DEPENDENCY`: blocked by external data, credentials, hardware, or the real run itself.

## Hard Gates
- Do not say `45K-ready` unless `reports/train_readiness_decision.md` is `TRAIN_ALLOWED`.
- Do not say `trained`, `benchmark-verified`, `mobile-ready`, `production-ready`, or `secure` unless there is direct evidence for that exact claim.
- Do not collapse planned work, scaffolds, or placeholders into completed work.

## Files To Avoid Touching Without Need
- `reports/snapshots/`
- `mertformer-titan-dealroom-private/`
- release zip artifacts unless the closure flow is regenerating them
- benchmark history snapshots unless the closure flow is explicitly updating them

## Handoff Rules
- Keep repo-internal closure evidence under `reports/`.
- Keep repo-external operator handoff on the Desktop when the closure flow generates it.
- If blockers remain, list the exact blocker codes and the next safe continuation boundary.
