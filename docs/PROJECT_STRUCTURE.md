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
│   ├── __init__.py  # Python module/script (config package initializer and exports)
│   ├── base.yaml  # YAML configuration file
│   └── config.py  # Python module/script (runtime configuration model and validation helpers)
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
│   └── CHAIN_MAP_TR.md  # Turkish document counterpart
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
│   ├── cfc_moe_tolerance_report.json  # JSON data artifact
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
│   ├── one_command_full_sop.log  # text/log artifact (single-command end-to-end SOP raw log; overwritten each run)
│   ├── one_command_full_sop_summary.md  # documentation/report file (single-command end-to-end SOP summary; overwritten each run)
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
│   ├── presentation_readiness_final.md  # documentation/report file
│   ├── proje_zip_rebuild_manifest_v2.json  # JSON data artifact
│   ├── proje_zip_rebuild_manifest_v2.md  # documentation/report file
│   ├── ram_guard_report.json  # JSON data artifact
│   ├── release_closure_lock_report.json  # JSON data artifact
│   ├── release_closure_note.md  # documentation/report file
│   ├── release_snapshot.md  # documentation/report file
│   ├── release_snapshot_TR.md  # Turkish document counterpart
│   ├── report_accuracy_audit.md  # documentation/report file
│   ├── report_accuracy_audit_TR.md  # Turkish document counterpart
│   ├── report_truth_matrix.md  # documentation/report file
│   ├── repro_build_report.json  # JSON data artifact
│   ├── resume_compat_report.json  # JSON data artifact
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
│   ├── training_readiness_manifest.json  # JSON data artifact
│   ├── unicode_path_guard_report.json  # JSON data artifact
│   ├── verified_matrix.md  # documentation/report file
│   ├── verified_matrix_TR.md  # Turkish document counterpart
│   ├── zip_audit_artifacts.json  # JSON data artifact
│   └── zip_audit_packages.json  # JSON data artifact
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
│   ├── __init__.py  # Python module/script (scripts package initializer and exports)
│   ├── apply_github_policy.sh  # shell automation script
│   ├── benchmarks_internal.py  # Python module/script (automation script for benchmarks internal)
│   ├── bitnet_kernel_benchmark_standalone.py  # Python module/script (automation script for bitnet kernel benchmark standalone)
│   ├── bootstrap_venv.sh  # shell automation script
│   ├── build_artifacts_release_zip.sh  # shell automation script
│   ├── build_investor_deck.py  # Python module/script (automation script for build investor deck)
│   ├── build_summary_pdf.py  # Python module/script (automation script for build summary pdf)
│   ├── build_validation_set.py  # Python module/script (automation script for build validation set)
│   ├── cfc_moe_tolerance_check.py  # Python module/script (automation script for cfc moe tolerance check)
│   ├── chat.py  # Python module/script (automation script for chat)
│   ├── check_57_matrix.py  # Python module/script (automation script for check 57 matrix)
│   ├── check_doc_claim_consistency.py  # Python module/script (automation script for check doc claim consistency)
│   ├── check_tokenizer_sync.py  # Python module/script (automation script for check tokenizer sync)
│   ├── check_translation_pointer_policy.py  # Python module/script (automation script for check translation pointer policy)
│   ├── checkpoint_restore_drill.py  # Python module/script (automation script for checkpoint restore drill)
│   ├── clean_runtime_artifacts.sh  # shell automation script
│   ├── cleanroom_verify.sh  # shell automation script
│   ├── data_pipeline.py  # Python module/script (automation script for data pipeline)
│   ├── dealroom_sync.py  # Python module/script (automation script for dealroom sync)
│   ├── docs_inventory.py  # Python module/script (markdown inventory and folder policy reporter)
│   ├── download_tr_tokenizer.py  # Python module/script (automation script for download tr tokenizer)
│   ├── drone_sitl_demo.py  # Python module/script (automation script for drone sitl demo)
│   ├── duplicate_zip_guard.py  # Python module/script (automation script for duplicate zip guard)
│   ├── eval.py  # Python module/script (automation script for eval)
│   ├── extract_dataset_refs.py  # Python module/script (automation script for extract dataset refs)
│   ├── failure_budget_drill.py  # Python module/script (automation script for failure budget drill)
│   ├── final_one_shot.sh  # shell automation script
│   ├── generate_bench_reports.py  # Python module/script (automation script for generate bench reports)
│   ├── generate_energy_baselines.py  # Python module/script (automation script for generate energy baselines)
│   ├── generate_sbom.py  # Python module/script (automation script for generate sbom)
│   ├── golden_eval.py  # Python module/script (automation script for golden eval)
│   ├── golden_score.py  # Python module/script (automation script for golden score)
│   ├── hardening_bundle.py  # Python module/script (automation script for hardening bundle)
│   ├── hash_manifest_to_json.py  # Python module/script (automation script for hash manifest to json)
│   ├── kaggle_onefile_demo_build30.py  # Python module/script (automation script for kaggle onefile demo build30)
│   ├── kaggle_onefile_demo_build30_colab_math_fastproof.py  # Python module/script (automation script for kaggle onefile demo build30 colab math fastproof)
│   ├── kaggle_train_compare_build30.py  # Python module/script (automation script for kaggle train compare build30)
│   ├── linkcheck_gate.py  # Python module/script (automation script for linkcheck gate)
│   ├── logbook_build.py  # Python module/script (automation script for logbook build)
│   ├── mac_simulation.py  # Python module/script (automation script for mac simulation)
│   ├── md_build30_sweep.py  # Python module/script (automation script for md build30 sweep)
│   ├── md_integrity_check.py  # Python module/script (automation script for md integrity check)
│   ├── md_quality_gate.py  # Python module/script (automation script for md quality gate)
│   ├── mini_titan_poc.py  # Python module/script (automation script for mini titan poc)
│   ├── mobile_export.py  # Python module/script (automation script for mobile export)
│   ├── nan_kill_test.py  # Python module/script (automation script for nan kill test)
│   ├── one_command_full_sop.sh  # shell automation script
│   ├── operator_mode_gate.py  # Python module/script (automation script for operator mode gate)
│   ├── overfit_gate.py  # Python module/script (automation script for overfit gate)
│   ├── plot_training_log.py  # Python module/script (automation script for plot training log)
│   ├── ram_guard.py  # Python module/script (automation script for ram guard)
│   ├── record_dataset_hashes.py  # Python module/script (automation script for record dataset hashes)
│   ├── release_build30.sh  # shell automation script
│   ├── release_closure_lock.sh  # shell automation script
│   ├── repro_build_check.py  # Python module/script (automation script for repro build check)
│   ├── resume_compat_check.py  # Python module/script (automation script for resume compat check)
│   ├── run_and_clean_pycache.py  # Python module/script (run command + guaranteed post-run cache sweep; add --include-venv-caches for venv cache cleanup)
│   ├── scaling_audit_math.py  # Python module/script (automation script for scaling audit math)
│   ├── secret_scan.py  # Python module/script (automation script for secret scan)
│   ├── smart_runner.py  # Python module/script (automation script for smart runner)
│   ├── smoke_train_benchmark.py  # Python module/script (automation script for smoke train benchmark)
│   ├── start_gate.py  # Python module/script (automation script for start gate)
│   ├── sync_manifest.py  # Python module/script (release manifest and project-structure sync generator)
│   ├── test_onnx_export.py  # Python module/script (automation script for test onnx export)
│   ├── titan_onnx_stress_test.py  # Python module/script (automation script for titan onnx stress test)
│   ├── titan_preflight.py  # Python module/script (automation script for titan preflight)
│   ├── train_smoke.py  # Python module/script (automation script for train smoke)
│   ├── train_tpu_turbo.py  # Python module/script (automation script for train tpu turbo)
│   ├── unicode_path_guard.py  # Python module/script (automation script for unicode path guard)
│   ├── update_investor_deck.py  # Python module/script (automation script for update investor deck)
│   ├── update_system_hardware.py  # Python module/script (automation script for update system hardware)
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
│   ├── test_cognitive_extensions.py  # Python module/script (automated test module for cognitive extensions)
│   ├── test_comprehensive.py  # Python module/script (automated test module for comprehensive)
│   ├── test_continual_adapter.py  # Python module/script (automated test module for continual adapter)
│   ├── test_cpp_kernel_loader.py  # Python module/script (automated test module for cpp kernel loader)
│   ├── test_dispatcher_extended.py  # Python module/script (automated test module for dispatcher extended)
│   ├── test_drone_sitl_demo.py  # Python module/script (automated test module for drone sitl demo)
│   ├── test_eval_suites.py  # Python module/script (automated test module for eval suites)
│   ├── test_export_metadata.py  # Python module/script (automated test module for export metadata)
│   ├── test_kaggle_compare_script.py  # Python module/script (automated test module for kaggle compare script)
│   ├── test_kaggle_onefile_colab_math_fastproof.py  # Python module/script (automated test module for kaggle onefile colab math fastproof)
│   ├── test_kaggle_onefile_compile_guard.py  # Python module/script (automated test module for kaggle onefile compile guard)
│   ├── test_kaggle_onefile_config.py  # Python module/script (automated test module for kaggle onefile config)
│   ├── test_kaggle_onefile_feature_coverage.py  # Python module/script (automated test module for kaggle onefile feature coverage)
│   ├── test_kaggle_onefile_zero_shot_unseen.py  # Python module/script (automated test module for kaggle onefile zero shot unseen)
│   ├── test_kernel_dispatcher.py  # Python module/script (automated test module for kernel dispatcher)
│   ├── test_kernel_equivalence.py  # Python module/script (automated test module for kernel equivalence)
│   ├── test_kpi_report_cli.py  # Python module/script (automated test module for kpi report cli)
│   ├── test_lifelong_safety.py  # Python module/script (automated test module for lifelong safety)
│   ├── test_mla_regressions.py  # Python module/script (automated test module for mla regressions)
│   ├── test_model.py  # Python module/script (automated test module for model)
│   ├── test_onnx_custom_op_contract.py  # Python module/script (automated test module for onnx custom op contract)
│   ├── test_onnx_export_path.py  # Python module/script (automated test module for onnx export path)
│   ├── test_onnx_metadata_hook.py  # Python module/script (automated test module for onnx metadata hook)
│   ├── test_orchestrator_swarm_runtime.py  # Python module/script (automated test module for orchestrator swarm runtime)
│   ├── test_sdk_api.py  # Python module/script (automated test module for sdk api)
│   ├── test_sdk_pilot_cli.py  # Python module/script (automated test module for sdk pilot cli)
│   ├── test_train_loop_sanity.py  # Python module/script (automated test module for train loop sanity)
│   ├── test_triad_omega_api.py  # Python module/script (automated test module for triad omega api)
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
│   └── train.py  # Python module/script (main training loop entrypoint)
├── training_dynamics/  # directory
│   ├── cold_vs_warm.md  # documentation/report file
│   └── cold_vs_warm_TR.md  # Turkish document counterpart
├── utils/  # directory
│   ├── __init__.py  # Python module/script (utils package initializer and exports)
│   ├── dataset_registry.py  # Python module/script (module for dataset registry)
│   ├── logger.py  # Python module/script (module for logger)
│   └── safety.py  # Python module/script (module for safety)
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
├── MODEL_LICENSE.md  # documentation/report file
├── MODEL_LICENSE_TR.md  # Turkish document counterpart
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
├── TROUBLESHOOTING.md  # documentation/report file
├── TROUBLESHOOTING_TR.md  # Turkish document counterpart
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
└── snake_demo.py  # Python module/script (module for snake demo)
```
