# Chess Evidence Contract

## Core measured artifacts
A serious chess onefile run should preserve:
- `reports/run_summary.json`
- `reports/model_card.json`
- `reports/eval_card.json`
- `reports/feature_flag_report.json`
- `reports/run_status_manifest.json`
- `reports/postrun_analysis_manifest.json`
- `reports/artifact_truth_matrix.json`
- `reports/run_contract.json`
- `reports/release_snapshot.json`
- `reports/evidence_pack_stub.json`
- `reports/final_truth_registry.json`
- `reports/claim_registry.json`
- `reports/known_limits.json`
- `reports/support_matrix.json`
- `reports/release_gate_summary.json`
- `reports/rc_stub.json`
- `reports/golden_stub.json`
- `reports/handoff_pack_manifest.json`
- `reports/operator_handoff_summary.json`
- `reports/external_repro_stub.json`
- `reports/pilot_stub.json`
- `reports/security_stub.json`
- `reports/legal_stub.json`
- `reports/operator_handbook_stub.json`
- `reports/dr_evidence_stub.json`
- `reports/backup_retention_stub.json`
- `reports/blind_handoff_stub.json`
- `reports/release_notes_stub.json`
- `reports/freeze_manifest_stub.json`
- `reports/changelog_snapshot.json`
- `reports/maintenance_policy_stub.json`
- `reports/export_truth_stub.json`
- `reports/device_validation_stub.json`
- `reports/packaging_closure_stub.json`
- `reports/installer_validation_stub.json`
- `reports/benchmark_raw_outputs_stub.json`
- `reports/benchmark_compare_report_stub.json`
- `reports/benchmark_summary_stub.json`
- `reports/benchmark_manifest_stub.json`
- `reports/artifact_manifest_with_hashes.json`
- `logs/run_log.jsonl`

## Internal-only artifacts
The following are useful and real, but remain internal unless separately validated:
- `reports/selfplay_report.json`
- `reports/inference_mode_tournament_report.json`
- `reports/replay_buffer_manifest.json`
- `reports/curated_position_suite_report.json`
- `reports/stockfish_match_report.json`

## Not enough by themselves
None of the following alone proves external chess strength:
- demo replay
- self-play
- inference-mode tournament
- internal proxy score
