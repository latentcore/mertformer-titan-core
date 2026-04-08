# Quality Gate Matrix

| Lane | Minimum Gate | Evidence Surface |
| --- | --- | --- |
| repo closure | `bash scripts/verify_all.sh` | `reports/release_manifest.json`, `reports/policy_sync_report.json` |
| train readiness | `python3 scripts/build_train_readiness_contract.py --allow-not-ready` | `reports/train_readiness_decision.json` |
| closure governance | `python3 scripts/build_closure_governance_pack.py` | `reports/final_truth_matrix.md`, `reports/repo_closure_scorecard.md` |
| code truth | `python3 scripts/build_code_truth_audit.py` | `reports/code_truth_delta_audit.md` |
| chess onefile readiness | `python3 scripts/build_chess_training_readiness_report.py` | `reports/chess_training_readiness_report.md` |
| windows delivery | `python3 scripts/export_chess_5080_share.py` plus delivery tests | `reports/target_machine_handoff_manifest.md` |
| max closeout | `bash scripts/final_one_shot.sh` | release and handoff refresh artifacts |

## KPI and SLA Reference
- `reports/kpi_pack_v1.md`
- `reports/commercial_handover/sla_kpi_90_180.md`
