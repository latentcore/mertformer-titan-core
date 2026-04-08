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
