# PROJECT_STRUCTURE

Generated automatically from tracked files with inline role comments.

```text
mertformer-titan-core/  # project root (git ls-files inventory)
├── .github/  # directory
│   ├── ISSUE_TEMPLATE/  # directory
│   │   ├── bug_report.md  # documentation/report file
│   │   └── feature_request.md  # documentation/report file
│   ├── workflows/  # directory
│   │   └── ci.yml  # YAML configuration file
│   ├── CODEOWNERS  # artifact
│   └── PULL_REQUEST_TEMPLATE.md  # documentation/report file
├── ablations/  # directory
│   ├── bitlinear_off/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── dense_only/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── no_liquid/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── no_moe/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── results.md  # documentation/report file
│   └── results_TR.md  # Turkish document counterpart
├── adr/  # directory
│   ├── ADR-0001-source-of-truth-and-claim-boundary.md  # documentation/report file
│   ├── ADR-0002-change-control-and-closure-governance.md  # documentation/report file
│   ├── ADR-0003-chess-oneclick-delivery-runtime-contract.md  # documentation/report file
│   ├── ADR-0004-blocker-fix-pass-core-override.md  # documentation/report file
│   └── ADR-0005-parallel-precompute-orchestration.md  # documentation/report file
├── applications/  # directory
│   └── anthropic/  # directory
│       ├── PACKET_POINTER_20260419.md  # documentation/report file
│       ├── README.md  # primary documentation (EN)
│       ├── application_strategy.md  # documentation/report file
│       ├── interview_prep.md  # documentation/report file
│       ├── measured_evidence_summary.md  # documentation/report file
│       ├── mertformer_anthropic_packet_20260419.zip.sha256  # artifact checksum
│       ├── performance_engineer_fallback.md  # documentation/report file
│       ├── project_summary.md  # documentation/report file
│       ├── science_of_scaling_cv_seed.md  # documentation/report file
│       ├── strongest_stories.md  # documentation/report file
│       ├── tokens_variant_notes.md  # documentation/report file
│       └── why_anthropic_science_of_scaling.md  # documentation/report file
├── apps/  # directory
│   └── chess_gui/  # directory
│       ├── checkpoints/  # directory
│       │   └── README.md  # primary documentation (EN)
│       ├── logs/  # directory
│       │   └── README.md  # primary documentation (EN)
│       ├── .gitignore  # git ignore policy
│       ├── README.md  # primary documentation (EN)
│       ├── launch_mertformer_chess_gui.command  # artifact
│       └── play_mertformer_chess_web.py  # Python module/script (module for play mertformer chess web)
├── artifacts/  # directory
│   └── mertformer_release.zip.sha256  # artifact checksum
├── assets/  # directory
│   ├── sources/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── header.png  # media asset
│   ├── snake_demo_preview.gif  # media asset
│   ├── snake_demo_proof.mp4  # media asset
│   └── synaptic_map.png  # media asset
├── checklists/  # directory
│   ├── README.md  # primary documentation (EN)
│   ├── chess_4060_24h.md  # documentation/report file
│   ├── chess_4060_24h_TR.md  # Turkish document counterpart
│   ├── chess_4060_24h_all_on_experimental.md  # documentation/report file
│   └── chess_4060_24h_all_on_experimental_TR.md  # Turkish document counterpart
├── config/  # directory
│   ├── export/  # directory
│   │   └── onnx_mobile.yaml  # YAML configuration file
│   ├── model/  # directory
│   │   ├── mertformer_max_arch.yaml  # YAML configuration file
│   │   ├── mertformer_moe.yaml  # YAML configuration file
│   │   └── mertformer_small.yaml  # YAML configuration file
│   ├── train/  # directory
│   │   ├── finetune.yaml  # YAML configuration file
│   │   └── pretrain.yaml  # YAML configuration file
│   ├── __init__.py  # Python module/script (config package initializer and exports)
│   ├── base.yaml  # YAML configuration file
│   └── config.py  # Python module/script (runtime configuration model and validation helpers)
├── configs/  # directory
│   ├── README.md  # primary documentation (EN)
│   ├── README_TR.md  # Turkish document counterpart
│   └── chess_onefile_profile_contract.md  # documentation/report file
├── datasets/  # directory
│   ├── INTERNAL_POLICY.md  # documentation/report file
│   ├── INTERNAL_POLICY_TR.md  # Turkish document counterpart
│   ├── LICENSES.md  # documentation/report file
│   ├── LICENSES_TR.md  # Turkish document counterpart
│   ├── README.md  # primary documentation (EN)
│   ├── README_TR.md  # Turkish document counterpart
│   ├── SOURCES.md  # documentation/report file
│   ├── SOURCES_TR.md  # Turkish document counterpart
│   ├── filters.yaml  # YAML configuration file
│   ├── golden_assertions.jsonl  # JSONL data/log artifact
│   ├── golden_samples.jsonl  # JSONL data/log artifact
│   ├── hashes.json  # JSON data artifact
│   ├── inventory.json  # JSON data artifact
│   ├── inventory.md  # documentation/report file
│   ├── inventory_TR.md  # Turkish document counterpart
│   └── validation.jsonl  # JSONL data/log artifact
├── docs/  # directory
│   ├── CHAIN_MAP.md  # documentation/report file
│   ├── CHAIN_MAP_TR.md  # Turkish document counterpart
│   ├── CHESS_ONEFILE_MASTER_TRUTH.md  # documentation/report file
│   ├── CHESS_ONEFILE_MASTER_TRUTH_TR.md  # Turkish document counterpart
│   ├── KAGGLE_PILOT.md  # documentation/report file
│   ├── MERTFORMER_5080_ARTIFACT_INTAKE_CHECKLIST_TR.md  # Turkish document counterpart
│   ├── MERTFORMER_5080_FINAL_ONEFILE_TRUTH.md  # documentation/report file
│   ├── MERTFORMER_5080_FINAL_ONEFILE_TRUTH_TR.md  # Turkish document counterpart
│   ├── PROJECT_MASTER_TRUTH.md  # documentation/report file
│   ├── PROJECT_MASTER_TRUTH_TR.md  # Turkish document counterpart
│   └── QUICKSTART_CPU.md  # documentation/report file
├── documents/  # directory
│   ├── README_TR_before_final_simplification.md  # documentation/report file
│   ├── README_before_final_simplification.md  # documentation/report file
│   └── README_snapshot_source.md  # documentation/report file
├── economics/  # directory
│   ├── cost_model.md  # documentation/report file
│   ├── cost_model_TR.md  # Turkish document counterpart
│   ├── efficiency_report.md  # documentation/report file
│   ├── efficiency_report_TR.md  # Turkish document counterpart
│   └── flops_estimator.py  # Python module/script (module for flops estimator)
├── eval/  # directory
│   ├── agentic_suite.py  # Python module/script (evaluation routine for agentic suite)
│   ├── generalization_suite.py  # Python module/script (evaluation routine for generalization suite)
│   ├── golden.py  # Python module/script (evaluation routine for golden)
│   ├── gsm8k.py  # Python module/script (evaluation routine for gsm8k)
│   ├── humaneval.py  # Python module/script (evaluation routine for humaneval)
│   └── report_builder.py  # Python module/script (evaluation routine for report builder)
├── evidence/  # directory
│   ├── build30_t4_onecell/  # directory
│   │   ├── EVIDENCE_POINTER.md  # documentation/report file
│   │   ├── run_summary.json  # JSON data artifact
│   │   ├── sha256.txt  # text artifact
│   │   └── smoke_test_result.md  # documentation/report file
│   ├── README.md  # primary documentation (EN)
│   └── chess_evidence_contract.md  # documentation/report file
├── experiments/  # directory
│   └── exp_001_baseline/  # directory
│       ├── config.yaml  # YAML configuration file
│       ├── metrics.json  # JSON data artifact
│       ├── notes.md  # documentation/report file
│       └── notes_TR.md  # Turkish document counterpart
├── interfaces/  # directory
│   ├── backlog_item_v1.schema.json  # JSON schema artifact
│   ├── closure_57_matrix_v1.schema.json  # JSON schema artifact
│   ├── inference_contract.md  # documentation/report file
│   ├── inference_contract_TR.md  # Turkish document counterpart
│   ├── kpi_report_v1.schema.json  # JSON schema artifact
│   ├── pilot_report_v1.schema.json  # JSON schema artifact
│   ├── run_manifest_v1.schema.json  # JSON schema artifact
│   ├── tokenizer_spec.json  # JSON data artifact
│   └── workspace_hygiene_manifest_v1.schema.json  # JSON schema artifact
├── knowledge/  # directory
│   ├── README.md  # primary documentation (EN)
│   └── chess_onefile_glossary.md  # documentation/report file
├── layers/  # directory
│   ├── __init__.py  # Python module/script (layers package initializer and exports)
│   ├── bitlinear.py  # Python module/script (BitLinear low-bit linear layer implementation)
│   ├── bitnet_patch.py  # Python module/script (BitNet quantization patch and runtime hooks)
│   ├── cognitive_extensions.py  # Python module/script (optional cognitive extension blocks)
│   ├── ffn.py  # Python module/script (feed-forward network blocks (dense and sparse paths))
│   ├── lifelong_safety.py  # Python module/script (lifelong safety guard layer)
│   ├── liquid.py  # Python module/script (liquid neural dynamics layers)
│   ├── mertformer_block.py  # Python module/script (core transformer block composition)
│   ├── mla.py  # Python module/script (multi-head latent attention implementation)
│   ├── moe.py  # Python module/script (mixture-of-experts routing and expert execution)
│   ├── qinn.py  # Python module/script (QINN experimental regulation layer (feature-flag))
│   └── world_model_head.py  # Python module/script (world-model auxiliary head)
├── limits/  # directory
│   ├── scaling_breakpoints.md  # documentation/report file
│   ├── scaling_breakpoints_TR.md  # Turkish document counterpart
│   └── stress_curves.png  # media asset
├── logs/  # directory
│   ├── README.md  # primary documentation (EN)
│   └── README_TR.md  # Turkish document counterpart
├── mertformer_sdk/  # directory
│   ├── kernels/  # directory
│   │   ├── cpp/  # directory
│   │   │   ├── __init__.py  # Python module/script (cpp package initializer and exports)
│   │   │   ├── bitnet_cpu.cpp  # C++ source file
│   │   │   └── loader.py  # Python module/script (SDK component for loader)
│   │   ├── metal/  # directory
│   │   │   ├── __init__.py  # Python module/script (metal package initializer and exports)
│   │   │   └── engine.py  # Python module/script (SDK component for engine)
│   │   ├── npu/  # directory
│   │   │   ├── __init__.py  # Python module/script (npu package initializer and exports)
│   │   │   └── engine.py  # Python module/script (SDK component for engine)
│   │   ├── vulkan/  # directory
│   │   │   ├── __init__.py  # Python module/script (vulkan package initializer and exports)
│   │   │   └── engine.py  # Python module/script (SDK component for engine)
│   │   ├── __init__.py  # Python module/script (kernels package initializer and exports)
│   │   ├── dispatcher.py  # Python module/script (SDK component for dispatcher)
│   │   ├── onnx_custom_op.py  # Python module/script (SDK component for onnx custom op)
│   │   ├── triton_fused_bitlinear.py  # Python module/script (Triton fused BitLinear CUDA kernel surface)
│   │   └── triton_ternary.py  # Python module/script (SDK component for triton ternary)
│   ├── utils/  # directory
│   │   ├── __init__.py  # Python module/script (utils package initializer and exports)
│   │   ├── bitpack.py  # Python module/script (SDK component for bitpack)
│   │   └── onnx_meta.py  # Python module/script (SDK component for onnx meta)
│   ├── __init__.py  # Python module/script (mertformer_sdk package initializer and exports)
│   ├── api.py  # Python module/script (SDK component for api)
│   ├── cli.py  # Python module/script (SDK component for cli)
│   ├── export.py  # Python module/script (SDK component for export)
│   ├── kpi.py  # Python module/script (SDK component for kpi)
│   └── pilot.py  # Python module/script (SDK component for pilot)
├── model/  # directory
│   ├── __init__.py  # Python module/script (model package initializer and exports)
│   └── transformers.py  # Python module/script (MertFormer model assembly and forward graph)
├── orchestrator/  # directory
│   ├── __init__.py  # Python module/script (orchestrator package initializer and exports)
│   ├── agent_registry.py  # Python module/script (orchestrator runtime component for agent registry)
│   ├── alignment_contracts.py  # Python module/script (orchestrator runtime component for alignment contracts)
│   ├── audio_sense.py  # Python module/script (orchestrator runtime component for audio sense)
│   ├── cognitive.py  # Python module/script (orchestrator runtime component for cognitive)
│   ├── cognitive_loop.py  # Python module/script (orchestrator runtime component for cognitive loop)
│   ├── compute_orchestrator.py  # Python module/script (orchestrator runtime component for compute orchestrator)
│   ├── core.py  # Python module/script (orchestrator runtime component for core)
│   ├── distillation_manager.py  # Python module/script (orchestrator runtime component for distillation manager)
│   ├── experience_store.py  # Python module/script (orchestrator runtime component for experience store)
│   ├── failure_budget.py  # Python module/script (orchestrator runtime component for failure budget)
│   ├── governance.py  # Python module/script (orchestrator runtime component for governance)
│   ├── hardware.py  # Python module/script (orchestrator runtime component for hardware)
│   ├── memory.py  # Python module/script (orchestrator runtime component for memory)
│   ├── paths.py  # Python module/script (orchestrator runtime component for paths)
│   ├── planner.py  # Python module/script (orchestrator runtime component for planner)
│   ├── reasoning_engine.py  # Python module/script (orchestrator runtime component for reasoning engine)
│   ├── self_audit.py  # Python module/script (orchestrator runtime component for self audit)
│   ├── self_improvement_guard.py  # Python module/script (orchestrator runtime component for self improvement guard)
│   ├── sense_engine.py  # Python module/script (orchestrator runtime component for sense engine)
│   ├── swarm_runtime.py  # Python module/script (orchestrator runtime component for swarm runtime)
│   ├── telemetry.py  # Python module/script (orchestrator runtime component for telemetry)
│   ├── tool_executor.py  # Python module/script (orchestrator runtime component for tool executor)
│   ├── tool_registry.py  # Python module/script (orchestrator runtime component for tool registry)
│   ├── verifier.py  # Python module/script (orchestrator runtime component for verifier)
│   └── web_sense.py  # Python module/script (orchestrator runtime component for web sense)
├── packages/  # directory
│   └── MertFormer_Titan_OnyxStorm_v2.0_B30_Release.zip.sha256  # artifact checksum
├── policy/  # directory
│   └── allow_deny_policy.yaml  # YAML configuration file
├── postmortems/  # directory
│   ├── README.md  # primary documentation (EN)
│   ├── README_TR.md  # Turkish document counterpart
│   ├── _template.md  # documentation/report file
│   ├── _template_TR.md  # Turkish document counterpart
│   ├── example_001.md  # documentation/report file
│   └── example_001_TR.md  # Turkish document counterpart
├── prompts/  # directory
│   ├── changelog.md  # documentation/report file
│   ├── changelog_TR.md  # Turkish document counterpart
│   └── system_v1.txt  # text artifact
├── registry/  # directory
│   └── mertformer_v0.1.json  # JSON data artifact
├── releases/  # directory
│   ├── README.md  # primary documentation (EN)
│   └── chess_release_contract.md  # documentation/report file
├── reports/  # directory
│   ├── ablations/  # directory
│   │   ├── liquid_ablation_final_20260615/  # directory
│   │   │   ├── plots/  # directory
│   │   │   │   ├── fig1_train_loss_log.png  # media asset
│   │   │   │   ├── fig2_loss_and_accuracy.png  # media asset
│   │   │   │   └── fig3_per_seed_scatter.png  # media asset
│   │   │   ├── MANIFEST.json  # JSON data artifact
│   │   │   └── final_summary.json  # JSON data artifact
│   │   ├── README.md  # primary documentation (EN)
│   │   ├── liquid_ablation_kaggle_20260614.json  # JSON data artifact
│   │   ├── liquid_ablation_pilot_curve.png  # media asset
│   │   └── liquid_ablation_results.json  # JSON data artifact
│   ├── benchmarks/  # directory
│   │   ├── linkedin_sweetspot/  # directory
│   │   │   ├── README.md  # primary documentation (EN)
│   │   │   ├── README_TR.md  # Turkish document counterpart
│   │   │   ├── run_20260318_144125_artifact_index.json  # JSON data artifact
│   │   │   ├── run_20260318_144125_compare.csv  # CSV data artifact
│   │   │   ├── run_20260318_144125_compare.json  # JSON data artifact
│   │   │   ├── run_20260318_144125_compare.md  # documentation/report file
│   │   │   ├── run_20260318_144125_health.txt  # text artifact
│   │   │   ├── run_20260318_144125_run_log.jsonl  # JSONL data/log artifact
│   │   │   ├── run_20260318_144125_step_metrics.csv  # CSV data artifact
│   │   │   ├── run_20260318_144125_summary.json  # JSON data artifact
│   │   │   └── zip_manifest.json  # JSON data artifact
│   │   ├── math_fastproof/  # directory
│   │   │   ├── README.md  # primary documentation (EN)
│   │   │   ├── README_TR.md  # Turkish document counterpart
│   │   │   ├── run_20260315_050133_artifact_index.json  # JSON data artifact
│   │   │   ├── run_20260315_050133_compare.csv  # CSV data artifact
│   │   │   ├── run_20260315_050133_compare.json  # JSON data artifact
│   │   │   ├── run_20260315_050133_compare.md  # documentation/report file
│   │   │   ├── run_20260315_050133_health.txt  # text artifact
│   │   │   ├── run_20260315_050133_run_log.jsonl  # JSONL data/log artifact
│   │   │   ├── run_20260315_050133_step_metrics.csv  # CSV data artifact
│   │   │   ├── run_20260315_050133_summary.json  # JSON data artifact
│   │   │   └── zip_manifest.json  # JSON data artifact
│   │   ├── text_understanding/  # directory
│   │   │   ├── README.md  # primary documentation (EN)
│   │   │   ├── README_TR.md  # Turkish document counterpart
│   │   │   ├── run_20260315_180151_artifact_index.json  # JSON data artifact
│   │   │   ├── run_20260315_180151_compare.csv  # CSV data artifact
│   │   │   ├── run_20260315_180151_compare.json  # JSON data artifact
│   │   │   ├── run_20260315_180151_compare.md  # documentation/report file
│   │   │   ├── run_20260315_180151_health.txt  # text artifact
│   │   │   ├── run_20260315_180151_run_log.jsonl  # JSONL data/log artifact
│   │   │   └── run_20260315_180151_summary.json  # JSON data artifact
│   │   ├── README.md  # primary documentation (EN)
│   │   ├── README_TR.md  # Turkish document counterpart
│   │   ├── agentic_suite_build30.json  # JSON data artifact
│   │   ├── generalization_suite_build30.json  # JSON data artifact
│   │   ├── internal_smoke_summary.json  # JSON data artifact
│   │   ├── kaggle_compare_build30.csv  # CSV data artifact
│   │   ├── kaggle_compare_build30.json  # JSON data artifact
│   │   ├── kaggle_compare_build30.md  # documentation/report file
│   │   ├── smoke_train_metrics.json  # JSON data artifact
│   │   ├── summary.json  # JSON data artifact
│   │   └── summary.md  # documentation/report file
│   ├── commercial_handover/  # directory
│   │   ├── contract_terms_checklist.md  # documentation/report file
│   │   ├── contract_terms_checklist_TR.md  # Turkish document counterpart
│   │   ├── handover_scope.md  # documentation/report file
│   │   ├── handover_scope_TR.md  # Turkish document counterpart
│   │   ├── known_issues.md  # documentation/report file
│   │   ├── known_issues_TR.md  # Turkish document counterpart
│   │   ├── ownership_and_role.md  # documentation/report file
│   │   ├── ownership_and_role_TR.md  # Turkish document counterpart
│   │   ├── sla_kpi_90_180.md  # documentation/report file
│   │   └── sla_kpi_90_180_TR.md  # Turkish document counterpart
│   ├── outreach/  # directory
│   │   ├── github_release_post.md  # documentation/report file
│   │   ├── huggingface_launch.md  # documentation/report file
│   │   ├── liquid_ablation_pilot_note_2026-06-15.md  # documentation/report file
│   │   ├── mertformer_titan_1_page_evidence_packet_2026-05-22.md  # documentation/report file
│   │   ├── mertformer_titan_executive_brief_2026-05-22.md  # documentation/report file
│   │   └── reddit_post.md  # documentation/report file
│   ├── pilots/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── snapshots/  # directory
│   │   └── 2026-02-24/  # directory
│   │       ├── claim_matrix_v2_2026-02-24.json  # JSON data artifact
│   │       ├── commercial_scenarios_v1_2026-02-24.json  # JSON data artifact
│   │       ├── evidence_snapshot_2026-02-24.json  # JSON data artifact
│   │       ├── mertformer_master_decision_report_TR_2026-02-24.md  # documentation/report file
│   │       ├── readiness_scorecard_v1_2026-02-24.json  # JSON data artifact
│   │       ├── report_interface_schema_v1.json  # JSON schema artifact
│   │       └── web_validation_sources_2026-02-24.md  # documentation/report file
│   ├── FACTS.json  # JSON data artifact
│   ├── adr_index.md  # documentation/report file
│   ├── architecture_honesty_audit.md  # documentation/report file
│   ├── artifacts_zip_denylist_audit.json  # JSON data artifact
│   ├── asset_stack.md  # documentation/report file
│   ├── asset_stack_TR.md  # Turkish document counterpart
│   ├── automation_boundary_policy.md  # documentation/report file
│   ├── backlog_operating_contract.md  # documentation/report file
│   ├── backup_restore_report.json  # JSON data artifact
│   ├── bench_cpp_report.json  # JSON data artifact
│   ├── bench_metal_report.json  # JSON data artifact
│   ├── bench_npu_report.json  # JSON data artifact
│   ├── bench_vulkan_report.json  # JSON data artifact
│   ├── bench_zero_copy_report.json  # JSON data artifact
│   ├── benchmark_compare_report.json  # JSON data artifact
│   ├── benchmark_compare_report.md  # documentation/report file
│   ├── benchmark_contract.md  # documentation/report file
│   ├── canonical_entrypoint.md  # documentation/report file
│   ├── cfc_moe_tolerance_report.json  # JSON data artifact
│   ├── change_control_sop.md  # documentation/report file
│   ├── checkpoint_contract.md  # documentation/report file
│   ├── checkpoint_hash_manifest.json  # JSON data artifact
│   ├── checkpoint_restore_report.json  # JSON data artifact
│   ├── chess_gui_onefile_sync_report.json  # JSON data artifact
│   ├── chess_gui_onefile_sync_report.md  # documentation/report file
│   ├── chess_onefile_extension_report.json  # JSON data artifact
│   ├── chess_onefile_extension_report.md  # documentation/report file
│   ├── chess_proof_teaching_case_study.md  # documentation/report file
│   ├── chess_teaching_contract_report.json  # JSON data artifact
│   ├── chess_teaching_contract_report.md  # documentation/report file
│   ├── chess_training_readiness_report.json  # JSON data artifact
│   ├── chess_training_readiness_report.md  # documentation/report file
│   ├── claim_number_audit.json  # JSON data artifact
│   ├── claim_registry.json  # JSON data artifact
│   ├── cleanroom_verification.md  # documentation/report file
│   ├── cleanroom_verification_TR.md  # Turkish document counterpart
│   ├── cleanup_scoped_closure_junk_report.json  # JSON data artifact
│   ├── cli_smoke_log.md  # documentation/report file
│   ├── cli_smoke_log_TR.md  # Turkish document counterpart
│   ├── closure_57_matrix.json  # JSON data artifact
│   ├── closure_57_matrix.md  # documentation/report file
│   ├── closure_57_matrix_TR.md  # Turkish document counterpart
│   ├── closure_release_truthsync_master_protocol.md  # documentation/report file
│   ├── closure_report_build30_v2.md  # documentation/report file
│   ├── closure_risk_register.md  # documentation/report file
│   ├── cloud_readiness_report.md  # documentation/report file
│   ├── code_truth_contract.md  # documentation/report file
│   ├── code_truth_delta_audit.json  # JSON data artifact
│   ├── code_truth_delta_audit.md  # documentation/report file
│   ├── codex_deep_audit_DE.md  # documentation/report file
│   ├── codex_deep_audit_DE_TR.md  # Turkish document counterpart
│   ├── codex_deep_audit_EN.md  # documentation/report file
│   ├── codex_deep_audit_EN_TR.md  # Turkish document counterpart
│   ├── codex_deep_audit_TR.md  # Turkish document counterpart
│   ├── commercial_handover_pack.md  # documentation/report file
│   ├── contamination_report_build30.md  # documentation/report file
│   ├── current_delta_addendum_2026_05_15.md  # documentation/report file
│   ├── customer_ready_definition.md  # documentation/report file
│   ├── data_pipeline_contract.md  # documentation/report file
│   ├── data_pipeline_provenance.json  # JSON data artifact
│   ├── data_pipeline_token_probe.json  # JSON data artifact
│   ├── dataset_health.md  # documentation/report file
│   ├── dataset_health_TR.md  # Turkish document counterpart
│   ├── dataset_health_final.md  # documentation/report file
│   ├── dataset_lineage_final.json  # JSON data artifact
│   ├── dealroom_reference.json  # JSON data artifact
│   ├── demo_bundle.md  # documentation/report file
│   ├── demo_bundle_manifest.json  # JSON data artifact
│   ├── deprecated_surface_report.md  # documentation/report file
│   ├── determinism_report.json  # JSON data artifact
│   ├── differential_backend_report.json  # JSON data artifact
│   ├── doc_alignment_report.json  # JSON data artifact
│   ├── doc_alignment_report.md  # documentation/report file
│   ├── doc_ownership_matrix.md  # documentation/report file
│   ├── docs_dedup_canonical_list.md  # documentation/report file
│   ├── docs_packages_hash_manifest.json  # JSON data artifact
│   ├── drone_sitl_demo.md  # documentation/report file
│   ├── drone_sitl_demo_TR.md  # Turkish document counterpart
│   ├── dry_run_report.json  # JSON data artifact
│   ├── dry_run_report.md  # documentation/report file
│   ├── duplicate_source_of_truth_report.md  # documentation/report file
│   ├── duplicate_zip_guard_report.json  # JSON data artifact
│   ├── edge_readiness_plan.md  # documentation/report file
│   ├── efficiency_convergence_analysis.md  # documentation/report file
│   ├── efficiency_convergence_analysis_TR.md  # Turkish document counterpart
│   ├── energy_baseline.json  # JSON data artifact
│   ├── entrypoint_deprecation_map.md  # documentation/report file
│   ├── execution_trace.json  # JSON data artifact
│   ├── exit_code_standard.md  # documentation/report file
│   ├── expected_artifacts_list.md  # documentation/report file
│   ├── export_validation_report.json  # JSON data artifact
│   ├── external_readability_checklist.md  # documentation/report file
│   ├── fallback_policy_report.json  # JSON data artifact
│   ├── feature_flag_governance.md  # documentation/report file
│   ├── feature_flag_governance_TR.md  # Turkish document counterpart
│   ├── file_state_inventory.json  # JSON data artifact
│   ├── final_artifact_manifest.json  # JSON data artifact
│   ├── final_backlog_classification.json  # JSON data artifact
│   ├── final_backlog_classification.md  # documentation/report file
│   ├── final_backlog_coverage_diff.md  # documentation/report file
│   ├── final_backlog_missing_items.md  # documentation/report file
│   ├── final_checksum_manifest.json  # JSON data artifact
│   ├── final_commands.md  # documentation/report file
│   ├── final_evidence_pack.md  # documentation/report file
│   ├── final_freeze_manifest.json  # JSON data artifact
│   ├── final_freeze_manifest.md  # documentation/report file
│   ├── final_master_plan_freeze.md  # documentation/report file
│   ├── final_orchestrator_status.json  # JSON data artifact
│   ├── final_orchestrator_status.md  # documentation/report file
│   ├── final_repo_audit.md  # documentation/report file
│   ├── final_sync_matrix.md  # documentation/report file
│   ├── final_sync_matrix_TR.md  # Turkish document counterpart
│   ├── final_truth_constitution.md  # documentation/report file
│   ├── final_truth_matrix.md  # documentation/report file
│   ├── folder_drift_report.json  # JSON data artifact
│   ├── folder_structure_policy.md  # documentation/report file
│   ├── founders_hub_application.md  # documentation/report file
│   ├── founders_hub_application_TR.md  # Turkish document counterpart
│   ├── github_policy_report.json  # JSON data artifact
│   ├── go_nogo_signoff_onepager.md  # documentation/report file
│   ├── go_nogo_signoff_onepager_TR.md  # Turkish document counterpart
│   ├── go_status_matrix.md  # documentation/report file
│   ├── go_status_matrix_TR.md  # Turkish document counterpart
│   ├── gtm_master_plan.md  # documentation/report file
│   ├── hardening_bundle_summary.json  # JSON data artifact
│   ├── immutable_evidence_register.json  # JSON data artifact
│   ├── immutable_evidence_register.md  # documentation/report file
│   ├── investable_definition.md  # documentation/report file
│   ├── investor_deck.pptx  # artifact
│   ├── investor_deck_TR.pptx  # artifact
│   ├── ip_licensing_split.md  # documentation/report file
│   ├── ip_licensing_split_TR.md  # Turkish document counterpart
│   ├── kaggle_onefile_closure_verify.json  # JSON data artifact
│   ├── kernel_fuzz_report.json  # JSON data artifact
│   ├── known_limits_v1.md  # documentation/report file
│   ├── kpi_contract_build30.md  # documentation/report file
│   ├── kpi_pack_v1.md  # documentation/report file
│   ├── kpi_pack_v1_TR.md  # Turkish document counterpart
│   ├── kpi_report_v1.json  # JSON data artifact
│   ├── latency_baseline.json  # JSON data artifact
│   ├── legal_cleanroom_signoff_internal.md  # documentation/report file
│   ├── legal_ip_pack.md  # documentation/report file
│   ├── license_gate_report.json  # JSON data artifact
│   ├── linkcheck_report.json  # JSON data artifact
│   ├── local_50step_proof_report.json  # JSON data artifact
│   ├── logger_contract.md  # documentation/report file
│   ├── logits_integrity_report.md  # documentation/report file
│   ├── master_closure_matrix.json  # JSON data artifact
│   ├── master_closure_matrix.md  # documentation/report file
│   ├── master_operating_plan.md  # documentation/report file
│   ├── md_lint_report.json  # JSON data artifact
│   ├── model_health.md  # documentation/report file
│   ├── model_health_TR.md  # Turkish document counterpart
│   ├── model_health_final.md  # documentation/report file
│   ├── ocean_2xh200_1024_first_launch_profile.md  # documentation/report file
│   ├── ocean_pre45k_h200_20260514_clean_summary.json  # JSON data artifact
│   ├── ocean_pre45k_h200_20260514_partial_evidence.md  # documentation/report file
│   ├── offline_4060_demo_evidence.md  # documentation/report file
│   ├── offline_4060_demo_summary.json  # JSON data artifact
│   ├── offline_assistant_case_study.md  # documentation/report file
│   ├── one_command_full_sop.log  # text/log artifact (single-command end-to-end SOP raw log; overwritten each run)
│   ├── one_command_full_sop_summary.md  # documentation/report file (single-command end-to-end SOP summary; overwritten each run)
│   ├── one_pager.md  # documentation/report file
│   ├── one_pager_TR.md  # Turkish document counterpart
│   ├── outreach_compute_sponsorship_messages.md  # documentation/report file
│   ├── owner_matrix.md  # documentation/report file
│   ├── ownership_proof_bundle.json  # JSON data artifact
│   ├── package_smoke_report.json  # JSON data artifact
│   ├── package_validation_report.md  # documentation/report file
│   ├── param_accounting_report.md  # documentation/report file
│   ├── phase2_carryover.md  # documentation/report file
│   ├── pilot_acceptance_signoff.md  # documentation/report file
│   ├── pilot_acceptance_signoff_TR.md  # Turkish document counterpart
│   ├── pilot_offer_packages.md  # documentation/report file
│   ├── pilot_offer_packages_TR.md  # Turkish document counterpart
│   ├── pilot_readiness_kit.md  # documentation/report file
│   ├── pilot_readiness_kit_TR.md  # Turkish document counterpart
│   ├── plot_contract.md  # documentation/report file
│   ├── poc_protocol.md  # documentation/report file
│   ├── poc_protocol_TR.md  # Turkish document counterpart
│   ├── post_45k_decision_tree.md  # documentation/report file
│   ├── post_train_automation_contract.md  # documentation/report file
│   ├── post_train_autorun_status.json  # JSON data artifact
│   ├── post_train_autorun_status.md  # documentation/report file
│   ├── post_train_state_machine.md  # documentation/report file
│   ├── presentation_readiness_final.md  # documentation/report file
│   ├── proje_zip_rebuild_manifest_v2.json  # JSON data artifact
│   ├── proje_zip_rebuild_manifest_v2.md  # documentation/report file
│   ├── quality_gate_matrix.md  # documentation/report file
│   ├── ram_guard_report.json  # JSON data artifact
│   ├── release_closure_lock_report.json  # JSON data artifact
│   ├── release_closure_note.md  # documentation/report file
│   ├── release_snapshot.md  # documentation/report file
│   ├── release_snapshot_TR.md  # Turkish document counterpart
│   ├── rented_machine_bringup.md  # documentation/report file
│   ├── repo_closure_scorecard.json  # JSON data artifact
│   ├── repo_closure_scorecard.md  # documentation/report file
│   ├── repo_directory_contract.md  # documentation/report file
│   ├── repo_external_handoff.md  # documentation/report file
│   ├── report_accuracy_audit.md  # documentation/report file
│   ├── report_accuracy_audit_TR.md  # Turkish document counterpart
│   ├── report_truth_matrix.md  # documentation/report file
│   ├── repro_build_report.json  # JSON data artifact
│   ├── resume_compat_report.json  # JSON data artifact
│   ├── review_checklist.md  # documentation/report file
│   ├── review_checklist_TR.md  # Turkish document counterpart
│   ├── run_contract.md  # documentation/report file
│   ├── runbook_validation_report.json  # JSON data artifact
│   ├── sales_funnel_90d.md  # documentation/report file
│   ├── sales_funnel_90d_TR.md  # Turkish document counterpart
│   ├── sanitizer_report.json  # JSON data artifact
│   ├── sbom.cdx.json  # JSON data artifact
│   ├── scoped_external_intake_matrix.json  # JSON data artifact
│   ├── scoped_external_intake_matrix.md  # documentation/report file
│   ├── security_compliance.md  # documentation/report file
│   ├── security_compliance_TR.md  # Turkish document counterpart
│   ├── smoke_run_report.json  # JSON data artifact
│   ├── snapshot_manifest_dealroom.json  # JSON data artifact
│   ├── snapshot_manifest_main.json  # JSON data artifact
│   ├── source_of_truth_map.md  # documentation/report file
│   ├── stale_script_report.md  # documentation/report file
│   ├── start_gate_operator_decision.json  # JSON data artifact
│   ├── start_gate_operator_decision.md  # documentation/report file
│   ├── start_gate_report.json  # JSON data artifact
│   ├── startup_selfcheck_report.json  # JSON data artifact
│   ├── static_analysis_report.json  # JSON data artifact
│   ├── strategic_value.md  # documentation/report file
│   ├── strategic_value_TR.md  # Turkish document counterpart
│   ├── support_maintenance_policy.md  # documentation/report file
│   ├── surface_lifecycle_matrix.md  # documentation/report file
│   ├── system_hardware.md  # documentation/report file
│   ├── system_hardware_TR.md  # Turkish document counterpart
│   ├── system_memory_policy.md  # documentation/report file
│   ├── system_stats.jsonl  # JSONL data/log artifact
│   ├── systems_performance_case_study.md  # documentation/report file
│   ├── target_machine_handoff_manifest.json  # JSON data artifact
│   ├── target_machine_handoff_manifest.md  # documentation/report file
│   ├── teacher_decision_record.md  # documentation/report file
│   ├── teacher_output_license_assessment.md  # documentation/report file
│   ├── technical_snapshot.md  # documentation/report file
│   ├── technical_snapshot_TR.md  # Turkish document counterpart
│   ├── test_verification_matrix.md  # documentation/report file
│   ├── test_verification_matrix_TR.md  # Turkish document counterpart
│   ├── thermal_baseline.json  # JSON data artifact
│   ├── tokenizer_parity_fix.md  # documentation/report file
│   ├── tokenizer_sync_final_report.md  # documentation/report file
│   ├── train_readiness_decision.json  # JSON data artifact
│   ├── train_readiness_decision.md  # documentation/report file
│   ├── training_outputs_bundle_manifest.json  # JSON data artifact
│   ├── training_outputs_bundle_manifest.md  # documentation/report file
│   ├── training_readiness_manifest.json  # JSON data artifact
│   ├── training_surface_audit_2026_05_15.json  # JSON data artifact
│   ├── training_surface_audit_2026_05_15.md  # documentation/report file
│   ├── turk_telekom_call_faq.md  # documentation/report file
│   ├── unicode_path_guard_report.json  # JSON data artifact
│   ├── update_first_policy.md  # documentation/report file
│   ├── verified_matrix.md  # documentation/report file
│   ├── verified_matrix_TR.md  # Turkish document counterpart
│   ├── workspace_hygiene_manifest.json  # JSON data artifact
│   ├── workspace_hygiene_manifest.md  # documentation/report file
│   ├── xla_smoke_report.json  # JSON data artifact
│   ├── zip_audit_artifacts.json  # JSON data artifact
│   └── zip_audit_packages.json  # JSON data artifact
├── repro/  # directory
│   ├── accelerate_8xgpu.yaml  # YAML configuration file
│   ├── accelerate_default.yaml  # YAML configuration file
│   ├── cuda.lock  # artifact
│   ├── env.lock  # artifact
│   ├── pip_freeze.txt  # text artifact
│   ├── python.md  # documentation/report file
│   ├── python_TR.md  # Turkish document counterpart
│   ├── seed_policy.md  # documentation/report file
│   └── seed_policy_TR.md  # Turkish document counterpart
├── runbooks/  # directory
│   ├── README.md  # primary documentation (EN)
│   ├── chess_4060_24h.md  # documentation/report file
│   ├── chess_4060_24h_TR.md  # Turkish document counterpart
│   ├── chess_4060_24h_all_on_experimental.md  # documentation/report file
│   └── chess_4060_24h_all_on_experimental_TR.md  # Turkish document counterpart
├── scripts/  # directory
│   ├── reports/  # directory
│   │   ├── model_health.md  # documentation/report file
│   │   └── model_health_TR.md  # Turkish document counterpart
│   ├── runs/  # directory
│   │   └── preflight/  # directory
│   │       └── config_snapshot.json  # JSON data artifact
│   ├── tools/  # directory
│   │   ├── claim_number_audit.py  # Python module/script (automation script for claim number audit)
│   │   └── denylist_scan_zip.py  # Python module/script (automation script for denylist scan zip)
│   ├── README.md  # primary documentation (EN)
│   ├── README_TR.md  # Turkish document counterpart
│   ├── __init__.py  # Python module/script (scripts package initializer and exports)
│   ├── apply_github_policy.sh  # shell automation script
│   ├── benchmark_liquid_impls.py  # Python module/script (automation script for benchmark liquid impls)
│   ├── benchmarks_internal.py  # Python module/script (automation script for benchmarks internal)
│   ├── bitnet_kernel_benchmark_standalone.py  # Python module/script (automation script for bitnet kernel benchmark standalone)
│   ├── bootstrap_venv.sh  # shell automation script
│   ├── build_artifacts_release_zip.sh  # shell automation script
│   ├── build_chess_5080_windows_delivery.py  # Python module/script (automation script for build chess 5080 windows delivery)
│   ├── build_chess_onefile_extension_report.py  # Python module/script (automation script for build chess onefile extension report)
│   ├── build_chess_teaching_contract_report.py  # Python module/script (automation script for build chess teaching contract report)
│   ├── build_chess_training_readiness_report.py  # Python module/script (automation script for build chess training readiness report)
│   ├── build_closure_governance_pack.py  # Python module/script (automation script for build closure governance pack)
│   ├── build_code_truth_audit.py  # Python module/script (automation script for build code truth audit)
│   ├── build_investor_deck.py  # Python module/script (automation script for build investor deck)
│   ├── build_master_closure_matrix.py  # Python module/script (automation script for build master closure matrix)
│   ├── build_max_closure_handoff.py  # Python module/script (automation script for build max closure handoff)
│   ├── build_mertformer_5080_final_delivery.py  # Python module/script (automation script for build mertformer 5080 final delivery)
│   ├── build_offline_closure_pack.py  # Python module/script (automation script for build offline closure pack)
│   ├── build_scoped_external_intake_matrix.py  # Python module/script (automation script for build scoped external intake matrix)
│   ├── build_summary_pdf.py  # Python module/script (automation script for build summary pdf)
│   ├── build_target_machine_handoff_bundle.py  # Python module/script (automation script for build target machine handoff bundle)
│   ├── build_train_readiness_contract.py  # Python module/script (automation script for build train readiness contract)
│   ├── build_training_outputs_bundle.py  # Python module/script (automation script for build training outputs bundle)
│   ├── build_validation_set.py  # Python module/script (automation script for build validation set)
│   ├── build_workspace_hygiene_manifest.py  # Python module/script (automation script for build workspace hygiene manifest)
│   ├── cfc_moe_tolerance_check.py  # Python module/script (automation script for cfc moe tolerance check)
│   ├── chat.py  # Python module/script (automation script for chat)
│   ├── check_57_matrix.py  # Python module/script (automation script for check 57 matrix)
│   ├── check_doc_claim_consistency.py  # Python module/script (automation script for check doc claim consistency)
│   ├── check_facts_consistency.py  # Python module/script (automation script for check facts consistency)
│   ├── check_tokenizer_sync.py  # Python module/script (automation script for check tokenizer sync)
│   ├── check_translation_pointer_policy.py  # Python module/script (automation script for check translation pointer policy)
│   ├── checkpoint_restore_drill.py  # Python module/script (automation script for checkpoint restore drill)
│   ├── chess_5080_onefile.py  # Python module/script (automation script for chess 5080 onefile)
│   ├── chess_onefile_contract.py  # Python module/script (automation script for chess onefile contract)
│   ├── clean_runtime_artifacts.sh  # shell automation script
│   ├── cleanroom_verify.sh  # shell automation script
│   ├── cleanup_scoped_closure_junk.py  # Python module/script (automation script for cleanup scoped closure junk)
│   ├── data_pipeline.py  # Python module/script (automation script for data pipeline)
│   ├── dealroom_sync.py  # Python module/script (automation script for dealroom sync)
│   ├── decrypt_mertformer_result_package.py  # Python module/script (automation script for decrypt mertformer result package)
│   ├── docs_inventory.py  # Python module/script (markdown inventory and folder policy reporter)
│   ├── download_tr_tokenizer.py  # Python module/script (automation script for download tr tokenizer)
│   ├── drone_sitl_demo.py  # Python module/script (automation script for drone sitl demo)
│   ├── duplicate_zip_guard.py  # Python module/script (automation script for duplicate zip guard)
│   ├── eval.py  # Python module/script (automation script for eval)
│   ├── export_chess_5080_share.py  # Python module/script (automation script for export chess 5080 share)
│   ├── extract_dataset_refs.py  # Python module/script (automation script for extract dataset refs)
│   ├── failure_budget_drill.py  # Python module/script (automation script for failure budget drill)
│   ├── final_one_shot.sh  # shell automation script
│   ├── final_orchestrator.py  # Python module/script (automation script for final orchestrator)
│   ├── generate_bench_reports.py  # Python module/script (automation script for generate bench reports)
│   ├── generate_energy_baselines.py  # Python module/script (automation script for generate energy baselines)
│   ├── generate_sbom.py  # Python module/script (automation script for generate sbom)
│   ├── golden_eval.py  # Python module/script (automation script for golden eval)
│   ├── golden_score.py  # Python module/script (automation script for golden score)
│   ├── hardening_bundle.py  # Python module/script (automation script for hardening bundle)
│   ├── hash_manifest_to_json.py  # Python module/script (automation script for hash manifest to json)
│   ├── kaggle_onecell_t4_build30.py  # Python module/script (automation script for kaggle onecell t4 build30)
│   ├── kaggle_onefile_closure_build30.py  # Python module/script (automation script for kaggle onefile closure build30)
│   ├── kaggle_onefile_demo_build30.py  # Python module/script (automation script for kaggle onefile demo build30)
│   ├── kaggle_onefile_demo_build30_colab_math_fastproof.py  # Python module/script (automation script for kaggle onefile demo build30 colab math fastproof)
│   ├── kaggle_onefile_demo_build30_text_understanding.py  # Python module/script (automation script for kaggle onefile demo build30 text understanding)
│   ├── kaggle_train_compare_build30.py  # Python module/script (automation script for kaggle train compare build30)
│   ├── launch_8xb300.sh  # shell automation script
│   ├── launch_ocean_45k.sh  # shell automation script
│   ├── linkcheck_gate.py  # Python module/script (automation script for linkcheck gate)
│   ├── liquid_train_impl_benchmark.py  # Python module/script (automation script for liquid train impl benchmark)
│   ├── logbook_build.py  # Python module/script (automation script for logbook build)
│   ├── logbook_verify.py  # Python module/script (automation script for logbook verify)
│   ├── mac_simulation.py  # Python module/script (automation script for mac simulation)
│   ├── macos_keepawake.sh  # shell automation script
│   ├── mathfp_interactive_chat.py  # Python module/script (automation script for mathfp interactive chat)
│   ├── md_build30_sweep.py  # Python module/script (automation script for md build30 sweep)
│   ├── md_integrity_check.py  # Python module/script (automation script for md integrity check)
│   ├── md_quality_gate.py  # Python module/script (automation script for md quality gate)
│   ├── mertformer_5080_final_onefile.py  # Python module/script (automation script for mertformer 5080 final onefile)
│   ├── mini_titan_poc.py  # Python module/script (automation script for mini titan poc)
│   ├── mobile_export.py  # Python module/script (automation script for mobile export)
│   ├── nan_kill_test.py  # Python module/script (automation script for nan kill test)
│   ├── offline_4060_demo_train.py  # Python module/script (automation script for offline 4060 demo train)
│   ├── one_command_full_sop.sh  # shell automation script
│   ├── operator_mode_gate.py  # Python module/script (automation script for operator mode gate)
│   ├── overfit_gate.py  # Python module/script (automation script for overfit gate)
│   ├── plot_training_log.py  # Python module/script (automation script for plot training log)
│   ├── post_run_processor.py  # Python module/script (automation script for post run processor)
│   ├── post_train_autorun.py  # Python module/script (automation script for post train autorun)
│   ├── precompute_logits_parallel.py  # Python module/script (automation script for precompute logits parallel)
│   ├── precompute_logits_topk.py  # Python module/script (automation script for precompute logits topk)
│   ├── ram_guard.py  # Python module/script (automation script for ram guard)
│   ├── record_dataset_hashes.py  # Python module/script (automation script for record dataset hashes)
│   ├── release_build30.sh  # shell automation script
│   ├── release_closure_lock.sh  # shell automation script
│   ├── repro_build_check.py  # Python module/script (automation script for repro build check)
│   ├── resume_compat_check.py  # Python module/script (automation script for resume compat check)
│   ├── run_and_clean_pycache.py  # Python module/script (run command + guaranteed post-run cache sweep; add --include-venv-caches for venv cache cleanup)
│   ├── run_liquid_ablation.py  # Python module/script (automation script for run liquid ablation)
│   ├── scaling_audit_math.py  # Python module/script (automation script for scaling audit math)
│   ├── secret_scan.py  # Python module/script (automation script for secret scan)
│   ├── smart_runner.py  # Python module/script (automation script for smart runner)
│   ├── smoke_train_benchmark.py  # Python module/script (automation script for smoke train benchmark)
│   ├── start_gate.py  # Python module/script (automation script for start gate)
│   ├── sync_chess_gui_onefile.py  # Python module/script (automation script for sync chess gui onefile)
│   ├── sync_manifest.py  # Python module/script (release manifest and project-structure sync generator)
│   ├── sync_test_stat_claims.py  # Python module/script (pytest pass/skipped claim synchronizer for tracked docs)
│   ├── test_onnx_export.py  # Python module/script (automation script for test onnx export)
│   ├── titan_onnx_stress_test.py  # Python module/script (automation script for titan onnx stress test)
│   ├── titan_preflight.py  # Python module/script (automation script for titan preflight)
│   ├── train_smoke.py  # Python module/script (automation script for train smoke)
│   ├── train_tpu_turbo.py  # Python module/script (automation script for train tpu turbo)
│   ├── unicode_path_guard.py  # Python module/script (automation script for unicode path guard)
│   ├── update_investor_deck.py  # Python module/script (automation script for update investor deck)
│   ├── update_system_hardware.py  # Python module/script (automation script for update system hardware)
│   ├── validate_logit_alignment.py  # Python module/script (automation script for validate logit alignment)
│   ├── verify_all.sh  # shell automation script
│   ├── verify_datasets.py  # Python module/script (automation script for verify datasets)
│   ├── verify_onnx_local.py  # Python module/script (automation script for verify onnx local)
│   ├── version_checker.py  # Python module/script (automation script for version checker)
│   ├── write_cuda_lock.py  # Python module/script (automation script for write cuda lock)
│   ├── xray.py  # Python module/script (automation script for xray)
│   └── zip_denylist_audit.py  # Python module/script (automation script for zip denylist audit)
├── telemetry/  # directory
│   └── metrics_schema.json  # JSON schema artifact
├── tests/  # directory
│   ├── test_57_matrix_gate.py  # Python module/script (automated test module for 57 matrix gate)
│   ├── test_agi_cognitive.py  # Python module/script (automated test module for agi cognitive)
│   ├── test_architecture_integrity.py  # Python module/script (automated test module for architecture integrity)
│   ├── test_benchmark_tokenizer_id.py  # Python module/script (automated test module for benchmark tokenizer id)
│   ├── test_bitnet_patch_skip.py  # Python module/script (automated test module for bitnet patch skip)
│   ├── test_build_chess_5080_windows_delivery.py  # Python module/script (automated test module for build chess 5080 windows delivery)
│   ├── test_build_chess_onefile_extension_report.py  # Python module/script (automated test module for build chess onefile extension report)
│   ├── test_build_chess_teaching_contract_report.py  # Python module/script (automated test module for build chess teaching contract report)
│   ├── test_build_chess_training_readiness_report.py  # Python module/script (automated test module for build chess training readiness report)
│   ├── test_build_closure_governance_pack.py  # Python module/script (automated test module for build closure governance pack)
│   ├── test_build_code_truth_audit.py  # Python module/script (automated test module for build code truth audit)
│   ├── test_build_max_closure_handoff.py  # Python module/script (automated test module for build max closure handoff)
│   ├── test_build_target_machine_handoff_bundle.py  # Python module/script (automated test module for build target machine handoff bundle)
│   ├── test_build_training_outputs_bundle.py  # Python module/script (automated test module for build training outputs bundle)
│   ├── test_build_validation_set.py  # Python module/script (automated test module for build validation set)
│   ├── test_build_workspace_hygiene_manifest.py  # Python module/script (automated test module for build workspace hygiene manifest)
│   ├── test_check_doc_claim_consistency.py  # Python module/script (automated test module for check doc claim consistency)
│   ├── test_checkpoint_tokenizer_id.py  # Python module/script (automated test module for checkpoint tokenizer id)
│   ├── test_chess_5080_onefile.py  # Python module/script (automated test module for chess 5080 onefile)
│   ├── test_chess_gui_contract.py  # Python module/script (automated test module for chess gui contract)
│   ├── test_chess_onefile_curated_suites.py  # Python module/script (automated test module for chess onefile curated suites)
│   ├── test_cognitive_extensions.py  # Python module/script (automated test module for cognitive extensions)
│   ├── test_comprehensive.py  # Python module/script (automated test module for comprehensive)
│   ├── test_config_contract.py  # Python module/script (automated test module for config contract)
│   ├── test_continual_adapter.py  # Python module/script (automated test module for continual adapter)
│   ├── test_cpp_kernel_loader.py  # Python module/script (automated test module for cpp kernel loader)
│   ├── test_dispatcher_extended.py  # Python module/script (automated test module for dispatcher extended)
│   ├── test_distillation_topk.py  # Python module/script (automated test module for distillation topk)
│   ├── test_drone_sitl_demo.py  # Python module/script (automated test module for drone sitl demo)
│   ├── test_duplicate_zip_guard.py  # Python module/script (automated test module for duplicate zip guard)
│   ├── test_eval_generation_eos.py  # Python module/script (automated test module for eval generation eos)
│   ├── test_eval_suites.py  # Python module/script (automated test module for eval suites)
│   ├── test_export_chess_5080_share.py  # Python module/script (automated test module for export chess 5080 share)
│   ├── test_export_metadata.py  # Python module/script (automated test module for export metadata)
│   ├── test_final_orchestrator_cli.py  # Python module/script (automated test module for final orchestrator cli)
│   ├── test_flops_estimator.py  # Python module/script (automated test module for flops estimator)
│   ├── test_freeze_policy.py  # Python module/script (automated test module for freeze policy)
│   ├── test_gradient_checkpoint_moe.py  # Python module/script (automated test module for gradient checkpoint moe)
│   ├── test_gsm8k_policy.py  # Python module/script (automated test module for gsm8k policy)
│   ├── test_kaggle_compare_script.py  # Python module/script (automated test module for kaggle compare script)
│   ├── test_kaggle_onefile_closure_build30.py  # Python module/script (automated test module for kaggle onefile closure build30)
│   ├── test_kaggle_onefile_colab_math_fastproof.py  # Python module/script (automated test module for kaggle onefile colab math fastproof)
│   ├── test_kaggle_onefile_compile_guard.py  # Python module/script (automated test module for kaggle onefile compile guard)
│   ├── test_kaggle_onefile_config.py  # Python module/script (automated test module for kaggle onefile config)
│   ├── test_kaggle_onefile_feature_coverage.py  # Python module/script (automated test module for kaggle onefile feature coverage)
│   ├── test_kaggle_onefile_zero_shot_unseen.py  # Python module/script (automated test module for kaggle onefile zero shot unseen)
│   ├── test_kd_mask.py  # Python module/script (automated test module for kd mask)
│   ├── test_kernel_dispatcher.py  # Python module/script (automated test module for kernel dispatcher)
│   ├── test_kernel_equivalence.py  # Python module/script (automated test module for kernel equivalence)
│   ├── test_kill_if_non_finite.py  # Python module/script (automated test module for kill if non finite)
│   ├── test_kpi_report_cli.py  # Python module/script (automated test module for kpi report cli)
│   ├── test_lifelong_safety.py  # Python module/script (automated test module for lifelong safety)
│   ├── test_liquid_safeguard.py  # Python module/script (automated test module for liquid safeguard)
│   ├── test_mertformer_5080_final_onefile.py  # Python module/script (automated test module for mertformer 5080 final onefile)
│   ├── test_mla_regressions.py  # Python module/script (automated test module for mla regressions)
│   ├── test_model.py  # Python module/script (automated test module for model)
│   ├── test_onnx_custom_op_contract.py  # Python module/script (automated test module for onnx custom op contract)
│   ├── test_onnx_export_path.py  # Python module/script (automated test module for onnx export path)
│   ├── test_onnx_metadata_hook.py  # Python module/script (automated test module for onnx metadata hook)
│   ├── test_optimizer_build.py  # Python module/script (automated test module for optimizer build)
│   ├── test_orchestrator_swarm_runtime.py  # Python module/script (automated test module for orchestrator swarm runtime)
│   ├── test_packed_projection_equivalence.py  # Python module/script (automated test module for packed projection equivalence)
│   ├── test_packing.py  # Python module/script (automated test module for packing)
│   ├── test_post_run_processor.py  # Python module/script (automated test module for post run processor)
│   ├── test_post_train_autorun_cli.py  # Python module/script (automated test module for post train autorun cli)
│   ├── test_precompute_parallel.py  # Python module/script (automated test module for precompute parallel)
│   ├── test_precompute_train_integration.py  # Python module/script (automated test module for precompute train integration)
│   ├── test_qinn_orthogonality.py  # Python module/script (automated test module for qinn orthogonality)
│   ├── test_resume_policy.py  # Python module/script (automated test module for resume policy)
│   ├── test_rng_resume.py  # Python module/script (automated test module for rng resume)
│   ├── test_scoped_external_tools.py  # Python module/script (automated test module for scoped external tools)
│   ├── test_sdk_api.py  # Python module/script (automated test module for sdk api)
│   ├── test_sdk_pilot_cli.py  # Python module/script (automated test module for sdk pilot cli)
│   ├── test_secret_scan.py  # Python module/script (automated test module for secret scan)
│   ├── test_start_gate.py  # Python module/script (automated test module for start gate)
│   ├── test_sync_chess_gui_onefile.py  # Python module/script (automated test module for sync chess gui onefile)
│   ├── test_telemetry_logger_contract.py  # Python module/script (automated test module for telemetry logger contract)
│   ├── test_titan_preflight_contract.py  # Python module/script (automated test module for titan preflight contract)
│   ├── test_tokenizer_parity.py  # Python module/script (automated test module for tokenizer parity)
│   ├── test_train_loop_sanity.py  # Python module/script (automated test module for train loop sanity)
│   ├── test_train_loss_eos_mask.py  # Python module/script (automated test module for train loss eos mask)
│   ├── test_triad_omega_api.py  # Python module/script (automated test module for triad omega api)
│   ├── test_triton_fused_bitlinear_cuda.py  # Python module/script (automated test module for triton fused bitlinear cuda)
│   ├── test_triton_fused_bitlinear_import.py  # Python module/script (automated test module for triton fused bitlinear import)
│   ├── test_validate_logit_alignment.py  # Python module/script (automated test module for validate logit alignment)
│   └── test_world_model_head.py  # Python module/script (automated test module for world model head)
├── tokenizer/  # directory
│   ├── tr/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── drift_report.md  # documentation/report file
│   ├── drift_report_TR.md  # Turkish document counterpart
│   ├── stats.md  # documentation/report file
│   ├── stats_TR.md  # Turkish document counterpart
│   └── tokenizer.json  # JSON data artifact
├── tools/  # directory
│   ├── contracts/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── sandbox/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   └── README_TR.md  # Turkish document counterpart
│   ├── abuse_tests.md  # documentation/report file
│   └── abuse_tests_TR.md  # Turkish document counterpart
├── train/  # directory
│   ├── __init__.py  # Python module/script (train package initializer and exports)
│   ├── continual_adapter.py  # Python module/script (continual learning adapter path for training)
│   ├── packing.py  # Python module/script (module for packing)
│   ├── train.py  # Python module/script (main training loop entrypoint)
│   ├── trainer_core.py  # Python module/script (module for trainer core)
│   ├── trainer_data.py  # Python module/script (module for trainer data)
│   └── trainer_eval.py  # Python module/script (module for trainer eval)
├── training_dynamics/  # directory
│   ├── cold_vs_warm.md  # documentation/report file
│   └── cold_vs_warm_TR.md  # Turkish document counterpart
├── utils/  # directory
│   ├── __init__.py  # Python module/script (utils package initializer and exports)
│   ├── dataset_registry.py  # Python module/script (module for dataset registry)
│   ├── liquid_safeguard.py  # Python module/script (module for liquid safeguard)
│   ├── logger.py  # Python module/script (module for logger)
│   ├── safety.py  # Python module/script (module for safety)
│   └── tokenizer_resolver.py  # Python module/script (module for tokenizer resolver)
├── .gitignore  # git ignore policy
├── .pre-commit-config.yaml  # YAML configuration file
├── ABLATION.md  # documentation/report file
├── ABLATION_TR.md  # Turkish document counterpart
├── AGENTS.md  # documentation/report file
├── ARCHITECTURE.md  # documentation/report file
├── CHANGELOG.md  # documentation/report file
├── CHANGELOG_TR.md  # Turkish document counterpart
├── CHESS_5080_POC_INTERNAL.md  # documentation/report file
├── CHESS_5080_POC_INTERNAL_TR.md  # Turkish document counterpart
├── CITATION.cff  # citation metadata
├── CONTRIBUTING.md  # documentation/report file
├── CONTRIBUTING_TR.md  # Turkish document counterpart
├── DECISIONS.md  # documentation/report file
├── DECISIONS_TR.md  # Turkish document counterpart
├── Dockerfile  # container build baseline
├── IMPLEMENTATION_PLAN.md  # documentation/report file
├── IMPLEMENTATION_PLAN_TR.md  # Turkish document counterpart
├── INTERNAL_AGI_GAP.md  # documentation/report file
├── INTERNAL_AGI_GAP_TR.md  # Turkish document counterpart
├── LICENSE  # license terms (EN)
├── LICENSE_TR  # license terms (TR)
├── MISSION.md  # documentation/report file
├── MISSION_TR.md  # Turkish document counterpart
├── MODEL_CARD.md  # documentation/report file
├── MODEL_CARD_TR.md  # Turkish document counterpart
├── MODEL_LICENSE.md  # documentation/report file
├── MODEL_LICENSE_TR.md  # Turkish document counterpart
├── NOTICE  # artifact
├── OFFLINE_4060_DEMO.md  # documentation/report file
├── PITCH.md  # documentation/report file
├── PITCH_TR.md  # Turkish document counterpart
├── README.md  # primary documentation (EN)
├── README_CHECKLIST.md  # documentation/report file
├── README_CHECKLIST_TR.md  # Turkish document counterpart
├── README_SUMMARY.md  # documentation/report file
├── README_SUMMARY.pdf  # artifact
├── README_SUMMARY_TR.md  # Turkish document counterpart
├── README_SUMMARY_TR.pdf  # artifact
├── README_TR.md  # Turkish document counterpart
├── SDK_GUIDE.md  # documentation/report file
├── SDK_GUIDE_TR.md  # Turkish document counterpart
├── SECURITY.md  # documentation/report file
├── SECURITY_TR.md  # Turkish document counterpart
├── START_HERE.md  # documentation/report file
├── TASK.md  # documentation/report file
├── TASK_TR.md  # Turkish document counterpart
├── TECHNICAL_REPORT.md  # documentation/report file
├── TECHNICAL_REPORT_TR.md  # Turkish document counterpart
├── TRAINING_PLAN.md  # documentation/report file
├── TRAINING_PLAN_TR.md  # Turkish document counterpart
├── TROUBLESHOOTING.md  # documentation/report file
├── TROUBLESHOOTING_TR.md  # Turkish document counterpart
├── USAGE_GUIDE.md  # documentation/report file
├── USAGE_GUIDE_TR.md  # Turkish document counterpart
├── USE_POLICY.md  # documentation/report file
├── USE_POLICY_TR.md  # Turkish document counterpart
├── V2_BACKLOG_SEED.md  # documentation/report file
├── WHITE_PAPER_LIQUIDROUTER.md  # documentation/report file
├── WHITE_PAPER_LIQUIDROUTER_TR.md  # Turkish document counterpart
├── conftest.py  # Python module/script (module for conftest)
├── constraints.txt  # text artifact
├── launch_mertformer_kaggle_closure.command  # artifact
├── pyproject.toml  # project metadata
├── requirements.txt  # text artifact
├── run.sh  # shell automation script
├── snake_demo.py  # Python module/script (module for snake demo)
└── zero_touch_start.sh  # shell automation script
```
