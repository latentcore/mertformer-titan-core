# KPI Pack v1 (Build 30)

This package defines the 12 KPI contract used for pilot go/no-go and enterprise review.

## Output Contract
- CLI: `mertformer kpi-report --out reports/kpi_report_v1.json`
- Schema: `interfaces/kpi_report_v1.schema.json`
- Report Schema Name: `kpi_report_v1`

## KPI Set (12)
1. verify_all pass
2. secret scan pass
3. pytest pass
4. preflight pass
5. operator gate pass
6. pilot schema present
7. release artifacts presence (zip + 2 age)
8. swarm omega readiness (45-agent)
9. onnx smoke pass
10. smoke benchmark availability
11. kaggle compare availability
12. claim eligibility gate

## Interpretation
- `readiness_score` is `pass_count / total_count`.
- `>= 0.90` is release-ready for controlled pilot.
- `< 0.90` requires action on warning checks.

## Notes
- KPI is evidence-based and does not claim model quality without trained checkpoints.
- ONNX check can be enabled with `--onnx-check`.
