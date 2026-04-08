# Chess 4060 24h All-On Experimental Runbook

## Purpose
Run the onefile chess stack on a single RTX 4060 for a bounded 24-hour experimental training pass with the widest safe feature surface enabled.

Profiles:
- `strength_4060_24h_all_on_experimental`
- `strength_4060_24h_omni_max`

## Recommended Commands
Baseline all-on experimental:
```bash
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h_all_on_experimental
```

Higher-pressure omni-max variant:
```bash
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h_omni_max
```

Selective rollback of a risky surface:
```bash
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h_omni_max --disable-features use_qinn,use_world_model_head
```

## Enabled Surfaces
The all-on family is expected to activate:
- MoE / expert paging / cross-expert sync
- Liquid / liquid adapter / QINN
- flash-attn inference + hierarchical KV cache
- workspace / neuromodulatory gain / latent ODE / Hebbian / neuro-symbolic / world-model / lifelong safety
- gradient checkpointing
- auxiliary chess heads: `phase_head`, `wdl_head`, `legality_head`

## Mandatory Evidence After Run
Expect these reports under the run directory:
- `reports/run_summary.json`
- `reports/run_summary.md`
- `reports/model_card.json`
- `reports/eval_card.json`
- `reports/feature_flag_report.json`
- `reports/feature_flag_report.md`
- `reports/mirror_parity_report.json`
- `reports/curated_position_suite_report.json`
- `reports/legal_move_safety.json`
- `reports/raw_vs_masked_policy_metrics.json`
- `reports/observability_report.json`
- `reports/selfplay_report.json`
- `reports/inference_mode_tournament_report.json`
- `reports/replay_buffer_manifest.json`
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
- `logs/run_log.jsonl`

## Operator Gates
Before launch:
- Confirm disk headroom for checkpoints and bundles.
- Confirm CUDA device visibility.
- Confirm `stockfish` path or auto-fetch allowance if gauntlet benchmarks are expected.
- Keep experimental flags explicit in the command or profile name.

During run:
- Watch for `fatal_exception`, `oom_event`, and repeated `midrun_snapshot_stockfish` failures in `logs/run_log.jsonl`.
- If the run is unstable, prefer disabling `use_qinn` or `use_world_model_head` first.

After run:
- Treat replay/demo output as demonstration material only.
- Treat self-play, tournament, and replay-buffer artifacts as internal diagnostics unless separately validated.
- Treat benchmark outputs as internal unless externally reproduced.
- Preserve the feature report alongside the checkpoint bundle so the exact head/feature mix remains auditable.
