# Expected Artifacts List

## Always Refreshed
- `reports/run_contract.md`
- `reports/expected_artifacts_list.md`
- `reports/exit_code_standard.md`
- `reports/post_train_automation_contract.md`
- `reports/post_train_state_machine.md`
- `interfaces/run_manifest_v1.schema.json`

## Check-Only / Start-Gate Path
- `reports/start_gate_report.json`
- `reports/train_readiness_decision.json`
- `reports/train_readiness_decision.md`
- `reports/training_readiness_manifest.json`
- `reports/final_orchestrator_status.json`

## Full Train Path (after success)
- `logs/production_run.log` or equivalent training stdout capture
- checkpoints under `cfg.save_dir`
- refreshed readiness/runtime manifests from the training path
- `reports/post_train_autorun_status.json`
- `reports/demo_bundle_manifest.json`
- `reports/final_evidence_pack.md`

## Post-Only Path
- `reports/post_train_autorun_status.json`
- `reports/demo_bundle_manifest.json`
- `reports/demo_bundle.md`
- `reports/final_evidence_pack.md`
