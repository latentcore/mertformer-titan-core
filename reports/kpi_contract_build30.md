# KPI Contract (Technical) — Build 30 V2

## Document Control
- Date (UTC): 2026-03-13
- Scope: Technical GO KPI contract for pre-claim operation
- Status: `ACTIVE`

## KPI Scope
This KPI contract defines objective pass criteria for technical GO only.

## KPI Set
1. Gate integrity
   - Command: `TITAN_OFFLINE=1 bash scripts/verify_all.sh`
   - Pass criterion: command ends with `[verify] OK`
2. Operator fail-fast
   - Command: `python3 scripts/operator_mode_gate.py --no-pytest --overfit-dataset datasets/validation.jsonl`
   - Pass criterion: no blocking failure
3. Smoke train (M4-safe)
   - Command: `python3 scripts/train_smoke.py --steps 20 --seq-len 64 --batch-size 2 --device mps --cleanup`
   - Pass criterion: smoke completes without runtime failure
4. Smoke metrics artifact
   - Command: `python3 scripts/smoke_train_benchmark.py --steps 20 --seq-len 64 --batch-size 2 --device mps`
   - Pass criterion: `reports/benchmarks/smoke_train_metrics.json` updated
5. Doc claim consistency
   - Command: `python3 scripts/check_doc_claim_consistency.py`
   - Pass criterion: consistency check passes
6. CfC/MoE tolerance check
   - Command: `python3 scripts/cfc_moe_tolerance_check.py --out reports/cfc_moe_tolerance_report.json`
   - Pass criterion: report `ok=true` and max loss diff <= 1%

## Measured vs Target Policy
- Measured: values produced by the commands above.
- Target: any projected performance/cost metrics not backed by command output.

## External Pending (Not part of Technical GO)
- Legal counsel sign-off
- Paid pilot/LOI closure
- Independent pentest/commercial compliance final sign-off

## Decision Rule
Technical GO is granted only if all KPI commands pass in a single run window.
