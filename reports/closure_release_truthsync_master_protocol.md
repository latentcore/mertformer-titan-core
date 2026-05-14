# Closure Release Truth-Sync Master Protocol Pointer

## Role
This file records the external operator protocol as supporting governance context.
It does not replace the repository source-of-truth order.

## Authority Boundary
The active authority order remains:
1. `AGENTS.md`
2. `reports/source_of_truth_map.md`
3. `reports/final_truth_constitution.md`
4. `reports/final_backlog_classification.md`
5. `reports/train_readiness_decision.md`
6. `README.md` / `README_TR.md`
7. `MODEL_CARD.md` / `MODEL_CARD_TR.md`
8. `USAGE_GUIDE.md` / `TRAINING_PLAN.md`

## External Protocol
- Local operator document: `/Users/mertyunlu/Documents/MertFormer_Kapanış_Release_TruthSync_Master_Protokolu_2026-05-12.md`
- Protocol version: `MertFormer Build30 Closure / Release / Truth-Sync Master Protocol v1`
- Created: `2026-05-12 15:05 +03`
- Last cited in this repo pass: `2026-05-14`

## Repo Interpretation
- The protocol is treated as an operator runbook and closure checklist.
- It is supporting context for verify/SOP/final closure sequencing.
- It cannot override claim-boundary rules, source-of-truth order, or readiness reason codes.
- Any repo-changing action still requires the canonical commands and generated reports in this repository.

## Canonical Commands
- Verification gate: `bash scripts/verify_all.sh`
- One-command closure flow: `bash scripts/one_command_full_sop.sh`
- Max closure refresh: `bash scripts/final_one_shot.sh`
- Readiness decision refresh: `python3 scripts/build_train_readiness_contract.py --allow-not-ready`
- Governance pack refresh: `python3 scripts/build_closure_governance_pack.py`
