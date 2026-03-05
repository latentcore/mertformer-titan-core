# PROJECT_STRUCTURE

Generated automatically from tracked files with inline role comments.

```text
mertformer-titan-core/  # project root (git ls-files inventory)
├── .github/  # directory
│   ├── workflows/  # directory
│   │   └── ci.yml  # YAML configuration file
│   └── CODEOWNERS  # artifact
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
│   ├── __init__.py  # Python module/script
│   ├── base.yaml  # YAML configuration file
│   └── config.py  # Python module/script
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
├── economics/  # directory
│   ├── cost_model.md  # documentation/report file
│   ├── cost_model_TR.md  # Turkish document counterpart
│   ├── efficiency_report.md  # documentation/report file
│   ├── efficiency_report_TR.md  # Turkish document counterpart
│   └── flops_estimator.py  # Python module/script
├── eval/  # directory
│   ├── agentic_suite.py  # Python module/script
│   ├── generalization_suite.py  # Python module/script
│   ├── golden.py  # Python module/script
│   ├── gsm8k.py  # Python module/script
│   ├── humaneval.py  # Python module/script
│   └── report_builder.py  # Python module/script
├── experiments/  # directory
│   └── exp_001_baseline/  # directory
│       ├── config.yaml  # YAML configuration file
│       ├── metrics.json  # JSON data artifact
│       ├── notes.md  # documentation/report file
│       └── notes_TR.md  # Turkish document counterpart
├── interfaces/  # directory
│   ├── closure_57_matrix_v1.schema.json  # JSON schema artifact
│   ├── inference_contract.md  # documentation/report file
│   ├── inference_contract_TR.md  # Turkish document counterpart
│   ├── kpi_report_v1.schema.json  # JSON schema artifact
│   ├── pilot_report_v1.schema.json  # JSON schema artifact
│   └── tokenizer_spec.json  # JSON data artifact
├── layers/  # directory
│   ├── __init__.py  # Python module/script
│   ├── bitlinear.py  # Python module/script
│   ├── bitnet_patch.py  # Python module/script
│   ├── cognitive_extensions.py  # Python module/script
│   ├── ffn.py  # Python module/script
│   ├── lifelong_safety.py  # Python module/script
│   ├── liquid.py  # Python module/script
│   ├── mertformer_block.py  # Python module/script
│   ├── mla.py  # Python module/script
│   ├── moe.py  # Python module/script
│   ├── qinn.py  # Python module/script
│   └── world_model_head.py  # Python module/script
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
│   │   │   ├── __init__.py  # Python module/script
│   │   │   ├── bitnet_cpu.cpp  # C++ source file
│   │   │   └── loader.py  # Python module/script
│   │   ├── metal/  # directory
│   │   │   ├── __init__.py  # Python module/script
│   │   │   └── engine.py  # Python module/script
│   │   ├── npu/  # directory
│   │   │   ├── __init__.py  # Python module/script
│   │   │   └── engine.py  # Python module/script
│   │   ├── vulkan/  # directory
│   │   │   ├── __init__.py  # Python module/script
│   │   │   └── engine.py  # Python module/script
│   │   ├── __init__.py  # Python module/script
│   │   ├── dispatcher.py  # Python module/script
│   │   ├── onnx_custom_op.py  # Python module/script
│   │   └── triton_ternary.py  # Python module/script
│   ├── utils/  # directory
│   │   ├── __init__.py  # Python module/script
│   │   ├── bitpack.py  # Python module/script
│   │   └── onnx_meta.py  # Python module/script
│   ├── __init__.py  # Python module/script
│   ├── api.py  # Python module/script
│   ├── cli.py  # Python module/script
│   ├── export.py  # Python module/script
│   ├── kpi.py  # Python module/script
│   └── pilot.py  # Python module/script
├── model/  # directory
│   ├── __init__.py  # Python module/script
│   └── transformers.py  # Python module/script
├── orchestrator/  # directory
│   ├── __init__.py  # Python module/script
│   ├── agent_registry.py  # Python module/script
│   ├── alignment_contracts.py  # Python module/script
│   ├── audio_sense.py  # Python module/script
│   ├── cognitive.py  # Python module/script
│   ├── cognitive_loop.py  # Python module/script
│   ├── compute_orchestrator.py  # Python module/script
│   ├── core.py  # Python module/script
│   ├── distillation_manager.py  # Python module/script
│   ├── experience_store.py  # Python module/script
│   ├── failure_budget.py  # Python module/script
│   ├── governance.py  # Python module/script
│   ├── hardware.py  # Python module/script
│   ├── memory.py  # Python module/script
│   ├── paths.py  # Python module/script
│   ├── planner.py  # Python module/script
│   ├── reasoning_engine.py  # Python module/script
│   ├── self_audit.py  # Python module/script
│   ├── self_improvement_guard.py  # Python module/script
│   ├── sense_engine.py  # Python module/script
│   ├── swarm_runtime.py  # Python module/script
│   ├── telemetry.py  # Python module/script
│   ├── tool_executor.py  # Python module/script
│   ├── tool_registry.py  # Python module/script
│   ├── verifier.py  # Python module/script
│   └── web_sense.py  # Python module/script
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
├── reports/  # directory
│   ├── benchmarks/  # directory
│   │   ├── README.md  # primary documentation (EN)
│   │   ├── README_TR.md  # Turkish document counterpart
│   │   ├── agentic_suite_build30.json  # JSON data artifact
│   │   ├── generalization_suite_build30.json  # JSON data artifact
│   │   ├── internal_smoke_summary.json  # JSON data artifact
│   │   ├── kaggle_compare_build30.csv  # CSV data artifact
│   │   ├── kaggle_compare_build30.json  # JSON data artifact
│   │   ├── kaggle_compare_build30.md  # documentation/report file
│   │   ├── smoke_train_metrics.json  # JSON data artifact
│   │   └── summary.json  # JSON data artifact
│   ├── commercial_handover/  # directory
│   │   ├── contract_terms_checklist.md  # documentation/report file
│   │   ├── handover_scope.md  # documentation/report file
│   │   ├── known_issues.md  # documentation/report file
│   │   ├── ownership_and_role.md  # documentation/report file
│   │   └── sla_kpi_90_180.md  # documentation/report file
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
│   ├── artifacts_zip_denylist_audit.json  # JSON data artifact
│   ├── asset_stack.md  # documentation/report file
│   ├── asset_stack_TR.md  # Turkish document counterpart
│   ├── backup_restore_report.json  # JSON data artifact
│   ├── bench_cpp_report.json  # JSON data artifact
│   ├── bench_metal_report.json  # JSON data artifact
│   ├── bench_npu_report.json  # JSON data artifact
│   ├── bench_vulkan_report.json  # JSON data artifact
│   ├── bench_zero_copy_report.json  # JSON data artifact
│   ├── cleanroom_verification.md  # documentation/report file
│   ├── cleanroom_verification_TR.md  # Turkish document counterpart
│   ├── cli_smoke_log.md  # documentation/report file
│   ├── cli_smoke_log_TR.md  # Turkish document counterpart
│   ├── closure_57_matrix.json  # JSON data artifact
│   ├── closure_57_matrix.md  # documentation/report file
│   ├── closure_57_matrix_TR.md  # Turkish document counterpart
│   ├── codex_deep_audit_DE.md  # documentation/report file
│   ├── codex_deep_audit_DE_TR.md  # Turkish document counterpart
│   ├── codex_deep_audit_EN.md  # documentation/report file
│   ├── codex_deep_audit_EN_TR.md  # Turkish document counterpart
│   ├── codex_deep_audit_TR.md  # Turkish document counterpart
│   ├── contamination_report_build30.md  # documentation/report file
│   ├── dataset_health.md  # documentation/report file
│   ├── dataset_health_TR.md  # Turkish document counterpart
│   ├── dealroom_reference.json  # JSON data artifact
│   ├── determinism_report.json  # JSON data artifact
│   ├── differential_backend_report.json  # JSON data artifact
│   ├── docs_dedup_canonical_list.md  # documentation/report file
│   ├── docs_packages_hash_manifest.json  # JSON data artifact
│   ├── drone_sitl_demo.md  # documentation/report file
│   ├── drone_sitl_demo_TR.md  # Turkish document counterpart
│   ├── duplicate_zip_guard_report.json  # JSON data artifact
│   ├── efficiency_convergence_analysis.md  # documentation/report file
│   ├── efficiency_convergence_analysis_TR.md  # Turkish document counterpart
│   ├── energy_baseline.json  # JSON data artifact
│   ├── execution_trace.json  # JSON data artifact
│   ├── fallback_policy_report.json  # JSON data artifact
│   ├── final_repo_audit.md  # documentation/report file
│   ├── final_sync_matrix.md  # documentation/report file
│   ├── final_sync_matrix_TR.md  # Turkish document counterpart
│   ├── folder_drift_report.json  # JSON data artifact
│   ├── folder_structure_policy.md  # documentation/report file
│   ├── founders_hub_application.md  # documentation/report file
│   ├── founders_hub_application_TR.md  # Turkish document counterpart
│   ├── github_policy_report.json  # JSON data artifact
│   ├── go_nogo_signoff_onepager.md  # documentation/report file
│   ├── go_nogo_signoff_onepager_TR.md  # Turkish document counterpart
│   ├── go_status_matrix.md  # documentation/report file
│   ├── go_status_matrix_TR.md  # Turkish document counterpart
│   ├── hardening_bundle_summary.json  # JSON data artifact
│   ├── investor_deck.pptx  # artifact
│   ├── investor_deck_TR.pptx  # artifact
│   ├── ip_licensing_split.md  # documentation/report file
│   ├── ip_licensing_split_TR.md  # Turkish document counterpart
│   ├── kernel_fuzz_report.json  # JSON data artifact
│   ├── kpi_contract_build30.md  # documentation/report file
│   ├── kpi_pack_v1.md  # documentation/report file
│   ├── kpi_pack_v1_TR.md  # Turkish document counterpart
│   ├── kpi_report_v1.json  # JSON data artifact
│   ├── latency_baseline.json  # JSON data artifact
│   ├── legal_cleanroom_signoff_internal.md  # documentation/report file
│   ├── license_gate_report.json  # JSON data artifact
│   ├── linkcheck_report.json  # JSON data artifact
│   ├── md_lint_report.json  # JSON data artifact
│   ├── model_health.md  # documentation/report file
│   ├── model_health_TR.md  # Turkish document counterpart
│   ├── one_pager.md  # documentation/report file
│   ├── one_pager_TR.md  # Turkish document counterpart
│   ├── ownership_proof_bundle.json  # JSON data artifact
│   ├── pilot_acceptance_signoff.md  # documentation/report file
│   ├── pilot_acceptance_signoff_TR.md  # Turkish document counterpart
│   ├── pilot_offer_packages.md  # documentation/report file
│   ├── pilot_offer_packages_TR.md  # Turkish document counterpart
│   ├── pilot_readiness_kit.md  # documentation/report file
│   ├── pilot_readiness_kit_TR.md  # Turkish document counterpart
│   ├── poc_protocol.md  # documentation/report file
│   ├── poc_protocol_TR.md  # Turkish document counterpart
│   ├── proje_zip_rebuild_manifest_v2.json  # JSON data artifact
│   ├── proje_zip_rebuild_manifest_v2.md  # documentation/report file
│   ├── ram_guard_report.json  # JSON data artifact
│   ├── release_closure_lock_report.json  # JSON data artifact
│   ├── release_closure_note.md  # documentation/report file
│   ├── release_snapshot.md  # documentation/report file
│   ├── release_snapshot_TR.md  # Turkish document counterpart
│   ├── report_accuracy_audit.md  # documentation/report file
│   ├── report_accuracy_audit_TR.md  # Turkish document counterpart
│   ├── repro_build_report.json  # JSON data artifact
│   ├── review_checklist.md  # documentation/report file
│   ├── review_checklist_TR.md  # Turkish document counterpart
│   ├── runbook_validation_report.json  # JSON data artifact
│   ├── sales_funnel_90d.md  # documentation/report file
│   ├── sales_funnel_90d_TR.md  # Turkish document counterpart
│   ├── sanitizer_report.json  # JSON data artifact
│   ├── sbom.cdx.json  # JSON data artifact
│   ├── security_compliance.md  # documentation/report file
│   ├── security_compliance_TR.md  # Turkish document counterpart
│   ├── snapshot_manifest_dealroom.json  # JSON data artifact
│   ├── snapshot_manifest_main.json  # JSON data artifact
│   ├── start_gate_report.json  # JSON data artifact
│   ├── startup_selfcheck_report.json  # JSON data artifact
│   ├── static_analysis_report.json  # JSON data artifact
│   ├── strategic_value.md  # documentation/report file
│   ├── strategic_value_TR.md  # Turkish document counterpart
│   ├── system_hardware.md  # documentation/report file
│   ├── system_hardware_TR.md  # Turkish document counterpart
│   ├── system_stats.jsonl  # JSONL data/log artifact
│   ├── teacher_output_license_assessment.md  # documentation/report file
│   ├── technical_snapshot.md  # documentation/report file
│   ├── technical_snapshot_TR.md  # Turkish document counterpart
│   ├── thermal_baseline.json  # JSON data artifact
│   ├── unicode_path_guard_report.json  # JSON data artifact
│   ├── verified_matrix.md  # documentation/report file
│   └── verified_matrix_TR.md  # Turkish document counterpart
├── repro/  # directory
│   ├── accelerate_default.yaml  # YAML configuration file
│   ├── cuda.lock  # artifact
│   ├── env.lock  # artifact
│   ├── pip_freeze.txt  # text artifact
│   ├── python.md  # documentation/report file
│   ├── python_TR.md  # Turkish document counterpart
│   ├── seed_policy.md  # documentation/report file
│   └── seed_policy_TR.md  # Turkish document counterpart
├── scripts/  # directory
│   ├── reports/  # directory
│   │   ├── model_health.md  # documentation/report file
│   │   └── model_health_TR.md  # Turkish document counterpart
│   ├── runs/  # directory
│   │   └── preflight/  # directory
│   │       └── config_snapshot.json  # JSON data artifact
│   ├── README.md  # primary documentation (EN)
│   ├── README_TR.md  # Turkish document counterpart
│   ├── __init__.py  # Python module/script
│   ├── apply_github_policy.sh  # shell automation script
│   ├── benchmarks_internal.py  # Python module/script
│   ├── bitnet_kernel_benchmark_standalone.py  # Python module/script
│   ├── bootstrap_venv.sh  # shell automation script
│   ├── build_investor_deck.py  # Python module/script
│   ├── build_summary_pdf.py  # Python module/script
│   ├── build_validation_set.py  # Python module/script
│   ├── chat.py  # Python module/script
│   ├── check_57_matrix.py  # Python module/script
│   ├── check_doc_claim_consistency.py  # Python module/script
│   ├── check_tokenizer_sync.py  # Python module/script
│   ├── check_translation_pointer_policy.py  # Python module/script
│   ├── checkpoint_restore_drill.py  # Python module/script
│   ├── clean_runtime_artifacts.sh  # shell automation script
│   ├── cleanroom_verify.sh  # shell automation script
│   ├── data_pipeline.py  # Python module/script
│   ├── dealroom_sync.py  # Python module/script
│   ├── docs_inventory.py  # Python module/script
│   ├── download_tr_tokenizer.py  # Python module/script
│   ├── drone_sitl_demo.py  # Python module/script
│   ├── duplicate_zip_guard.py  # Python module/script
│   ├── eval.py  # Python module/script
│   ├── extract_dataset_refs.py  # Python module/script
│   ├── failure_budget_drill.py  # Python module/script
│   ├── final_one_shot.sh  # shell automation script
│   ├── generate_bench_reports.py  # Python module/script
│   ├── generate_energy_baselines.py  # Python module/script
│   ├── generate_sbom.py  # Python module/script
│   ├── golden_eval.py  # Python module/script
│   ├── golden_score.py  # Python module/script
│   ├── hardening_bundle.py  # Python module/script
│   ├── hash_manifest_to_json.py  # Python module/script
│   ├── kaggle_onefile_demo_build30.py  # Python module/script
│   ├── kaggle_onefile_demo_build30_colab_math_fastproof.py  # Python module/script
│   ├── kaggle_train_compare_build30.py  # Python module/script
│   ├── linkcheck_gate.py  # Python module/script
│   ├── logbook_build.py  # Python module/script
│   ├── mac_simulation.py  # Python module/script
│   ├── md_build30_sweep.py  # Python module/script
│   ├── md_integrity_check.py  # Python module/script
│   ├── md_quality_gate.py  # Python module/script
│   ├── mini_titan_poc.py  # Python module/script
│   ├── mobile_export.py  # Python module/script
│   ├── nan_kill_test.py  # Python module/script
│   ├── operator_mode_gate.py  # Python module/script
│   ├── overfit_gate.py  # Python module/script
│   ├── ram_guard.py  # Python module/script
│   ├── record_dataset_hashes.py  # Python module/script
│   ├── release_build30.sh  # shell automation script
│   ├── release_closure_lock.sh  # shell automation script
│   ├── repro_build_check.py  # Python module/script
│   ├── scaling_audit_math.py  # Python module/script
│   ├── secret_scan.py  # Python module/script
│   ├── smart_runner.py  # Python module/script
│   ├── smoke_train_benchmark.py  # Python module/script
│   ├── start_gate.py  # Python module/script
│   ├── sync_manifest.py  # Python module/script
│   ├── test_onnx_export.py  # Python module/script
│   ├── titan_onnx_stress_test.py  # Python module/script
│   ├── titan_preflight.py  # Python module/script
│   ├── train_smoke.py  # Python module/script
│   ├── train_tpu_turbo.py  # Python module/script
│   ├── unicode_path_guard.py  # Python module/script
│   ├── update_system_hardware.py  # Python module/script
│   ├── verify_all.sh  # shell automation script
│   ├── verify_datasets.py  # Python module/script
│   ├── verify_onnx_local.py  # Python module/script
│   ├── version_checker.py  # Python module/script
│   ├── write_cuda_lock.py  # Python module/script
│   ├── xray.py  # Python module/script
│   └── zip_denylist_audit.py  # Python module/script
├── telemetry/  # directory
│   └── metrics_schema.json  # JSON schema artifact
├── tests/  # directory
│   ├── test_57_matrix_gate.py  # Python module/script
│   ├── test_agi_cognitive.py  # Python module/script
│   ├── test_architecture_integrity.py  # Python module/script
│   ├── test_cognitive_extensions.py  # Python module/script
│   ├── test_comprehensive.py  # Python module/script
│   ├── test_continual_adapter.py  # Python module/script
│   ├── test_cpp_kernel_loader.py  # Python module/script
│   ├── test_dispatcher_extended.py  # Python module/script
│   ├── test_drone_sitl_demo.py  # Python module/script
│   ├── test_eval_suites.py  # Python module/script
│   ├── test_export_metadata.py  # Python module/script
│   ├── test_kaggle_compare_script.py  # Python module/script
│   ├── test_kaggle_onefile_colab_math_fastproof.py  # Python module/script
│   ├── test_kaggle_onefile_compile_guard.py  # Python module/script
│   ├── test_kaggle_onefile_config.py  # Python module/script
│   ├── test_kaggle_onefile_feature_coverage.py  # Python module/script
│   ├── test_kaggle_onefile_zero_shot_unseen.py  # Python module/script
│   ├── test_kernel_dispatcher.py  # Python module/script
│   ├── test_kernel_equivalence.py  # Python module/script
│   ├── test_kpi_report_cli.py  # Python module/script
│   ├── test_lifelong_safety.py  # Python module/script
│   ├── test_mla_regressions.py  # Python module/script
│   ├── test_model.py  # Python module/script
│   ├── test_onnx_custom_op_contract.py  # Python module/script
│   ├── test_onnx_export_path.py  # Python module/script
│   ├── test_onnx_metadata_hook.py  # Python module/script
│   ├── test_orchestrator_swarm_runtime.py  # Python module/script
│   ├── test_sdk_api.py  # Python module/script
│   ├── test_sdk_pilot_cli.py  # Python module/script
│   ├── test_train_loop_sanity.py  # Python module/script
│   ├── test_triad_omega_api.py  # Python module/script
│   └── test_world_model_head.py  # Python module/script
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
│   ├── __init__.py  # Python module/script
│   ├── continual_adapter.py  # Python module/script
│   └── train.py  # Python module/script
├── training_dynamics/  # directory
│   ├── cold_vs_warm.md  # documentation/report file
│   └── cold_vs_warm_TR.md  # Turkish document counterpart
├── utils/  # directory
│   ├── __init__.py  # Python module/script
│   ├── dataset_registry.py  # Python module/script
│   ├── logger.py  # Python module/script
│   └── safety.py  # Python module/script
├── .gitignore  # git ignore policy
├── CHANGELOG.md  # documentation/report file
├── CHANGELOG_TR.md  # Turkish document counterpart
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
├── MODEL_CARD.md  # documentation/report file
├── MODEL_CARD_TR.md  # Turkish document counterpart
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
├── TASK.md  # documentation/report file
├── TASK_TR.md  # Turkish document counterpart
├── TECHNICAL_REPORT.md  # documentation/report file
├── TECHNICAL_REPORT_TR.md  # Turkish document counterpart
├── TRAINING_PLAN.md  # documentation/report file
├── TRAINING_PLAN_TR.md  # Turkish document counterpart
├── USAGE_GUIDE.md  # documentation/report file
├── USAGE_GUIDE_TR.md  # Turkish document counterpart
├── USE_POLICY.md  # documentation/report file
├── USE_POLICY_TR.md  # Turkish document counterpart
├── V2_BACKLOG_SEED.md  # documentation/report file
├── WHITE_PAPER_LIQUIDROUTER.md  # documentation/report file
├── WHITE_PAPER_LIQUIDROUTER_TR.md  # Turkish document counterpart
├── pyproject.toml  # project metadata
├── requirements.txt  # text artifact
├── run.sh  # shell automation script
└── snake_demo.py  # Python module/script
```
