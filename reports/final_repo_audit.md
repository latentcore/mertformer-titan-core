# Final Repo Audit

Generated UTC: 2026-03-04T01:08:54Z

## Scope
- One-shot closure workflow executed via [final] start_gate
{"ok": true}
[final] unicode_path_guard
OK: unicode path guard; hits=0
[final] sbom
SBOM written: reports/sbom.cdx.json
[final] repro_build
{"ok": true, "target_count": 6}
[final] energy_baselines
[final] hardening_bundle
{"generated_utc": "2026-03-04T01:09:24.663503+00:00", "ok": true, "reports": {"static_analysis": true, "sanitizer": true, "kernel_fuzz": true, "determinism": true, "differential": true, "license": true, "startup": true, "fallback": true, "backup_restore": true, "runbook": true}}
{"generated_utc": "2026-03-04T01:09:25.184724+00:00", "command": ".titan-venv/bin/python scripts/hardening_bundle.py", "warn_threshold_gb": 10.5, "slow_threshold_gb": 12.0, "hard_threshold_gb": 13.0, "peak_used_gb": 10.798, "terminated_by_guard": false, "events": [{"event": "warn", "used_gb": 10.798019409179688, "ts": 1772586564.674571}], "return_code": 0, "duration_sec": 1.021, "ok": true}
[final] bench_reports
[final] md_quality
md_quality: scope=release_core files=7 errors=0 warnings=14
[final] linkcheck
linkcheck: scope=release_core files=7 missing=0
[final] docs_inventory
{"ok": true, "md_files": 247, "dup_groups": 21}
[final] duplicate_zip_guard
OK: duplicate zip guard; groups=0
[final] sync_manifest
{"manifest_entries": 650, "ok": true}
[final] dealroom_sync
{"ok": true}
[final] demo_bundle
{"ok": true, "video": "artifacts/demo_v1.mp4", "used_fallback": true}
updating: SECURITY_TR.md (deflated 40%)
updating: artifacts/ (stored 0%)
updating: artifacts/demo_v1.mp4 (deflated 88%)
updating: tools/ (stored 0%)
updating: tools/abuse_tests_TR.md (deflated 41%)
updating: tools/abuse_tests.md (deflated 42%)
updating: tools/contracts/ (stored 0%)
updating: tools/contracts/README.md (deflated 40%)
updating: tools/contracts/README_TR.md (deflated 36%)
updating: tools/sandbox/ (stored 0%)
updating: tools/sandbox/README.md (deflated 35%)
updating: tools/sandbox/README_TR.md (deflated 35%)
updating: USE_POLICY.md (deflated 42%)
updating: PITCH_TR.md (deflated 35%)
updating: training_dynamics/ (stored 0%)
updating: training_dynamics/cold_vs_warm_TR.md (deflated 8%)
updating: training_dynamics/cold_vs_warm.md (deflated 11%)
updating: SDK_GUIDE_TR.md (deflated 54%)
updating: layers/ (stored 0%)
updating: layers/qinn.py (deflated 59%)
updating: layers/ffn.py (deflated 61%)
updating: layers/cognitive_extensions.py (deflated 74%)
updating: layers/liquid.py (deflated 69%)
updating: layers/bitlinear.py (deflated 67%)
updating: layers/__init__.py (stored 0%)
updating: layers/world_model_head.py (deflated 70%)
updating: layers/bitnet_patch.py (deflated 62%)
updating: layers/moe.py (deflated 74%)
updating: layers/mla.py (deflated 71%)
updating: layers/lifelong_safety.py (deflated 66%)
updating: layers/mertformer_block.py (deflated 69%)
updating: LICENSE (deflated 52%)
updating: MertFormer_Titan_Dealroom_2026-02-23/ (stored 0%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/ (stored 0%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/final_operator_checklist.md (deflated 26%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/privacy_visibility_audit_v1.json (deflated 66%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/rewrite_target_list_v1.json (deflated 21%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/rewrite_coverage_report_v1.json (deflated 33%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/deal_manifest_v1.json (deflated 23%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/consistency_check_v1.md (deflated 63%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/release_signoff_v1.md (deflated 24%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/claim_boundary_audit_v1.md (deflated 34%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/exposure_policy_v1.md (deflated 37%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/commit_chain_manifest_v2.json (deflated 58%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/deal_manifest_v2.json (deflated 93%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/ownership_proof_bundle_v2.json (deflated 56%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/_pre_rewrite_hashes.json (deflated 51%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/security_scan_result_v1.json (deflated 25%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/private_only_certificate_v1.md (deflated 28%)
updating: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/originality_audit_v1.json (deflated 91%)
updating: MertFormer_Titan_Dealroom_2026-02-23/03_TECH_EVIDENCE/ (stored 0%)
updating: MertFormer_Titan_Dealroom_2026-02-23/03_TECH_EVIDENCE/compile_cudagraph_stall_fix_note.md (deflated 37%)
updating: MertFormer_Titan_Dealroom_2026-02-23/03_TECH_EVIDENCE/closure_57_matrix_TR.md (deflated 36%)
updating: MertFormer_Titan_Dealroom_2026-02-23/03_TECH_EVIDENCE/live_validation_snapshot.md (deflated 41%)
updating: MertFormer_Titan_Dealroom_2026-02-23/03_TECH_EVIDENCE/feature_coverage_matrix_v2.md (deflated 42%)
updating: MertFormer_Titan_Dealroom_2026-02-23/03_TECH_EVIDENCE/closure_57_matrix_EN.md (deflated 39%)
updating: MertFormer_Titan_Dealroom_2026-02-23/03_TECH_EVIDENCE/report_accuracy_audit_EN.md (deflated 36%)
updating: MertFormer_Titan_Dealroom_2026-02-23/03_TECH_EVIDENCE/kpi_report_v1.json (deflated 55%)
updating: MertFormer_Titan_Dealroom_2026-02-23/03_TECH_EVIDENCE/report_accuracy_audit_TR.md (deflated 35%)
updating: MertFormer_Titan_Dealroom_2026-02-23/03_TECH_EVIDENCE/runtime_disclosure.md (deflated 33%)
updating: MertFormer_Titan_Dealroom_2026-02-23/02_EXECUTIVE/ (stored 0%)
updating: MertFormer_Titan_Dealroom_2026-02-23/02_EXECUTIVE/quick_decision_TR.md (deflated 34%)
updating: MertFormer_Titan_Dealroom_2026-02-23/02_EXECUTIVE/quick_decision_EN.md (deflated 35%)
updating: MertFormer_Titan_Dealroom_2026-02-23/02_EXECUTIVE/one_pager_EN.md (deflated 37%)
updating: MertFormer_Titan_Dealroom_2026-02-23/02_EXECUTIVE/one_pager_TR.md (deflated 35%)
updating: MertFormer_Titan_Dealroom_2026-02-23/02_EXECUTIVE/v1_closure_sync_v2.md (deflated 21%)
updating: MertFormer_Titan_Dealroom_2026-02-23/09_APPENDIX/ (stored 0%)
updating: MertFormer_Titan_Dealroom_2026-02-23/09_APPENDIX/technical_report_EN.md (deflated 39%)
updating: MertFormer_Titan_Dealroom_2026-02-23/09_APPENDIX/technical_report_TR.md (deflated 36%)
updating: MertFormer_Titan_Dealroom_2026-02-23/09_APPENDIX/liquidrouter_whitepaper_TR.md (deflated 35%)
updating: MertFormer_Titan_Dealroom_2026-02-23/09_APPENDIX/liquidrouter_whitepaper_EN.md (deflated 36%)
updating: MertFormer_Titan_Dealroom_2026-02-23/09_APPENDIX/model_card_EN.md (deflated 32%)
updating: MertFormer_Titan_Dealroom_2026-02-23/09_APPENDIX/v1_closure_sync_v2.md (deflated 21%)
updating: MertFormer_Titan_Dealroom_2026-02-23/09_APPENDIX/model_card_TR.md (deflated 31%)
updating: MertFormer_Titan_Dealroom_2026-02-23/08_GITHUB_GIST/ (stored 0%)
updating: MertFormer_Titan_Dealroom_2026-02-23/08_GITHUB_GIST/gist_payload_TR_EN.md (deflated 39%)
updating: MertFormer_Titan_Dealroom_2026-02-23/08_GITHUB_GIST/superseded_gists.md (deflated 25%)
updating: MertFormer_Titan_Dealroom_2026-02-23/08_GITHUB_GIST/v1_closure_sync_v2.md (deflated 21%)
updating: MertFormer_Titan_Dealroom_2026-02-23/08_GITHUB_GIST/secret_gist_url.txt (deflated 1%)
updating: MertFormer_Titan_Dealroom_2026-02-23/04_SECURITY_LEGAL/ (stored 0%)
updating: MertFormer_Titan_Dealroom_2026-02-23/04_SECURITY_LEGAL/LICENSE_TR.txt (deflated 50%)
updating: MertFormer_Titan_Dealroom_2026-02-23/04_SECURITY_LEGAL/LICENSE_EN.txt (deflated 52%)
updating: MertFormer_Titan_Dealroom_2026-02-23/04_SECURITY_LEGAL/teacher_output_license_assessment.md (deflated 36%)
updating: MertFormer_Titan_Dealroom_2026-02-23/04_SECURITY_LEGAL/security_compliance_TR.md (deflated 38%)
updating: MertFormer_Titan_Dealroom_2026-02-23/04_SECURITY_LEGAL/security_compliance_EN.md (deflated 38%)
updating: MertFormer_Titan_Dealroom_2026-02-23/04_SECURITY_LEGAL/v1_closure_sync_v2.md (deflated 21%)
updating: MertFormer_Titan_Dealroom_2026-02-23/04_SECURITY_LEGAL/legal_cleanroom_signoff_internal.md (deflated 36%)
updating: MertFormer_Titan_Dealroom_2026-02-23/06_INVESTOR_ASSETS/ (stored 0%)
updating: MertFormer_Titan_Dealroom_2026-02-23/06_INVESTOR_ASSETS/investor_deck_TR.pptx (deflated 13%)
updating: MertFormer_Titan_Dealroom_2026-02-23/06_INVESTOR_ASSETS/v1_closure_sync_v2.md (deflated 21%)
updating: MertFormer_Titan_Dealroom_2026-02-23/06_INVESTOR_ASSETS/founders_hub_application_EN.md (deflated 37%)
updating: MertFormer_Titan_Dealroom_2026-02-23/06_INVESTOR_ASSETS/private_placement_brief_TR.md (deflated 35%)
updating: MertFormer_Titan_Dealroom_2026-02-23/06_INVESTOR_ASSETS/investor_deck_EN.pptx (deflated 13%)
updating: MertFormer_Titan_Dealroom_2026-02-23/06_INVESTOR_ASSETS/private_placement_brief_EN.md (deflated 37%)
updating: MertFormer_Titan_Dealroom_2026-02-23/06_INVESTOR_ASSETS/founders_hub_application_TR.md (deflated 35%)
updating: MertFormer_Titan_Dealroom_2026-02-23/05_COMMERCIAL_PILOT/ (stored 0%)
updating: MertFormer_Titan_Dealroom_2026-02-23/05_COMMERCIAL_PILOT/sales_funnel_90d_TR.md (deflated 29%)
updating: MertFormer_Titan_Dealroom_2026-02-23/05_COMMERCIAL_PILOT/sales_funnel_90d_EN.md (deflated 28%)
updating: MertFormer_Titan_Dealroom_2026-02-23/05_COMMERCIAL_PILOT/pilot_readiness_kit_EN.md (deflated 31%)
updating: MertFormer_Titan_Dealroom_2026-02-23/05_COMMERCIAL_PILOT/pilot_acceptance_signoff_TR.md (deflated 28%)
updating: MertFormer_Titan_Dealroom_2026-02-23/05_COMMERCIAL_PILOT/pilot_packages_EN.md (deflated 46%)
updating: MertFormer_Titan_Dealroom_2026-02-23/05_COMMERCIAL_PILOT/pilot_packages_TR.md (deflated 41%)
updating: MertFormer_Titan_Dealroom_2026-02-23/05_COMMERCIAL_PILOT/pilot_acceptance_signoff_EN.md (deflated 29%)
updating: MertFormer_Titan_Dealroom_2026-02-23/05_COMMERCIAL_PILOT/pilot_readiness_kit_TR.md (deflated 31%)
updating: MertFormer_Titan_Dealroom_2026-02-23/05_COMMERCIAL_PILOT/poc_protocol_EN.md (deflated 34%)
updating: MertFormer_Titan_Dealroom_2026-02-23/05_COMMERCIAL_PILOT/poc_protocol_TR.md (deflated 34%)
updating: MertFormer_Titan_Dealroom_2026-02-23/05_COMMERCIAL_PILOT/company_readiness_pack_v2.md (deflated 44%)
updating: MertFormer_Titan_Dealroom_2026-02-23/01_START_HERE/ (stored 0%)
updating: MertFormer_Titan_Dealroom_2026-02-23/01_START_HERE/README_TR_EN.md (deflated 47%)
updating: MertFormer_Titan_Dealroom_2026-02-23/07_OUTREACH_READY/ (stored 0%)
updating: MertFormer_Titan_Dealroom_2026-02-23/07_OUTREACH_READY/email_templates_TR_EN.md (deflated 51%)
updating: MertFormer_Titan_Dealroom_2026-02-23/07_OUTREACH_READY/DM_templates_TR_EN.md (deflated 50%)
updating: MertFormer_Titan_Dealroom_2026-02-23/07_OUTREACH_READY/EXECUTE_NOW_48H_TR.md (deflated 42%)
updating: MertFormer_Titan_Dealroom_2026-02-23/07_OUTREACH_READY/linkedin_posts_TR_EN.md (deflated 49%)
updating: MertFormer_Titan_Dealroom_2026-02-23/07_OUTREACH_READY/v1_closure_sync_v2.md (deflated 21%)
updating: requirements.txt (deflated 47%)
updating: README_SUMMARY_TR.pdf (deflated 18%)
updating: WHITE_PAPER_LIQUIDROUTER.md (deflated 48%)
updating: CHANGELOG.md (deflated 51%)
updating: CHANGELOG_TR.md (deflated 48%)
updating: TECHNICAL_REPORT_TR.md (deflated 51%)
updating: config/ (stored 0%)
updating: config/config.py (deflated 67%)
updating: config/__init__.py (stored 0%)
updating: config/model/ (stored 0%)
updating: config/model/mertformer_moe.yaml (deflated 15%)
updating: config/model/mertformer_max_arch.yaml (deflated 46%)
updating: config/model/mertformer_small.yaml (deflated 37%)
updating: config/train/ (stored 0%)
updating: config/train/pretrain.yaml (deflated 13%)
updating: config/train/finetune.yaml (deflated 16%)
updating: config/export/ (stored 0%)
updating: config/export/onnx_mobile.yaml (deflated 5%)
updating: config/base.yaml (deflated 50%)
updating: USAGE_GUIDE.md (deflated 48%)
updating: MODEL_CARD.md (deflated 42%)
updating: Dockerfile (deflated 47%)
updating: WHITE_PAPER_LIQUIDROUTER_TR.md (deflated 48%)
updating: CITATION.cff (deflated 36%)
updating: experiments/ (stored 0%)
updating: experiments/exp_001_baseline/ (stored 0%)
updating: experiments/exp_001_baseline/metrics.json (deflated 15%)
updating: experiments/exp_001_baseline/notes_TR.md (deflated 10%)
updating: experiments/exp_001_baseline/config.yaml (deflated 35%)
updating: experiments/exp_001_baseline/notes.md (deflated 12%)
updating: tokenizer/ (stored 0%)
updating: tokenizer/stats.md (deflated 20%)
updating: tokenizer/tokenizer.json (deflated 27%)
updating: tokenizer/drift_report_TR.md (deflated 8%)
updating: tokenizer/drift_report.md (deflated 12%)
updating: tokenizer/stats_TR.md (deflated 16%)
updating: tokenizer/tr/ (stored 0%)
updating: tokenizer/tr/README.md (deflated 32%)
updating: tokenizer/tr/README_TR.md (deflated 30%)
updating: ablations/ (stored 0%)
updating: ablations/no_liquid/ (stored 0%)
updating: ablations/no_liquid/README.md (deflated 29%)
updating: ablations/no_liquid/README_TR.md (deflated 20%)
updating: ablations/bitlinear_off/ (stored 0%)
updating: ablations/bitlinear_off/README.md (deflated 26%)
updating: ablations/bitlinear_off/README_TR.md (deflated 18%)
updating: ablations/no_moe/ (stored 0%)
updating: ablations/no_moe/README.md (deflated 28%)
updating: ablations/no_moe/README_TR.md (deflated 19%)
updating: ablations/dense_only/ (stored 0%)
updating: ablations/dense_only/README.md (deflated 29%)
updating: ablations/dense_only/README_TR.md (deflated 21%)
updating: ablations/results_TR.md (deflated 47%)
updating: ablations/results.md (deflated 50%)
updating: pyproject.toml (deflated 48%)
updating: datasets/ (stored 0%)
updating: datasets/INTERNAL_POLICY_TR.md (deflated 46%)
updating: datasets/logits/ (stored 0%)
updating: datasets/SOURCES.md (deflated 75%)
updating: datasets/inventory.md (deflated 74%)
updating: datasets/SOURCES_TR.md (deflated 74%)
updating: datasets/golden_assertions.jsonl (deflated 75%)
updating: datasets/hashes.json (deflated 77%)
updating: datasets/LICENSES.md (deflated 62%)
updating: datasets/README.md (deflated 41%)
updating: datasets/inventory_TR.md (deflated 74%)
updating: datasets/validation.jsonl (deflated 62%)
updating: datasets/README_TR.md (deflated 38%)
updating: datasets/filters.yaml (deflated 21%)
updating: datasets/golden_samples.jsonl (deflated 65%)
updating: datasets/INTERNAL_POLICY.md (deflated 47%)
updating: datasets/inventory.json (deflated 87%)
updating: datasets/LICENSES_TR.md (deflated 61%)
updating: tests/ (stored 0%)
updating: tests/test_model.py (deflated 62%)
updating: tests/test_kaggle_compare_script.py (deflated 63%)
updating: tests/test_sdk_api.py (deflated 59%)
updating: tests/test_dispatcher_extended.py (deflated 58%)
updating: tests/test_agi_cognitive.py (deflated 75%)
updating: tests/test_world_model_head.py (deflated 64%)
updating: tests/test_kaggle_onefile_colab_math_fastproof.py (deflated 66%)
updating: tests/test_kernel_equivalence.py (deflated 54%)
updating: tests/test_eval_suites.py (deflated 62%)
updating: tests/test_sdk_pilot_cli.py (deflated 68%)
updating: tests/test_lifelong_safety.py (deflated 46%)
updating: tests/test_57_matrix_gate.py (deflated 58%)
updating: tests/test_drone_sitl_demo.py (deflated 59%)
updating: tests/test_kernel_dispatcher.py (deflated 54%)
updating: tests/test_onnx_metadata_hook.py (deflated 54%)
updating: tests/test_kpi_report_cli.py (deflated 57%)
updating: tests/test_kaggle_onefile_zero_shot_unseen.py (deflated 62%)
updating: tests/test_continual_adapter.py (deflated 49%)
updating: tests/test_onnx_custom_op_contract.py (deflated 59%)
updating: tests/test_kaggle_onefile_feature_coverage.py (deflated 56%)
updating: tests/test_mla_regressions.py (deflated 70%)
updating: tests/test_kaggle_onefile_compile_guard.py (deflated 61%)
updating: tests/test_cpp_kernel_loader.py (deflated 47%)
updating: tests/test_architecture_integrity.py (deflated 66%)
updating: tests/test_cognitive_extensions.py (deflated 65%)
updating: tests/test_orchestrator_swarm_runtime.py (deflated 60%)
updating: tests/test_export_metadata.py (deflated 52%)
updating: tests/test_comprehensive.py (deflated 71%)
updating: tests/test_kaggle_onefile_config.py (deflated 60%)
updating: tests/test_onnx_export_path.py (deflated 51%)
updating: tests/test_train_loop_sanity.py (deflated 72%)
updating: tests/test_triad_omega_api.py (deflated 66%)
updating: limits/ (stored 0%)
updating: limits/scaling_breakpoints.md (deflated 44%)
updating: limits/stress_curves.png (deflated 9%)
updating: limits/scaling_breakpoints_TR.md (deflated 40%)
updating: utils/ (stored 0%)
updating: utils/dataset_registry.py (deflated 60%)
updating: utils/__init__.py (stored 0%)
updating: utils/logger.py (deflated 74%)
updating: utils/safety.py (deflated 58%)
updating: PITCH.md (deflated 37%)
updating: TECHNICAL_REPORT.md (deflated 52%)
updating: TRAINING_PLAN.md (deflated 51%)
updating: docs/ (stored 0%)
updating: docs/PROJECT_STRUCTURE.md (deflated 81%)
updating: postmortems/ (stored 0%)
updating: postmortems/example_001_TR.md (deflated 29%)
updating: postmortems/_template.md (deflated 32%)
updating: postmortems/_template_TR.md (deflated 28%)
updating: postmortems/README.md (deflated 16%)
updating: postmortems/README_TR.md (deflated 16%)
updating: postmortems/example_001.md (deflated 31%)
updating: run.sh (deflated 68%)
updating: README.md (deflated 70%)
updating: DECISIONS_TR.md (deflated 25%)
updating: README_CHECKLIST_TR.md (deflated 44%)
updating: INTERNAL_AGI_GAP_TR.md (deflated 51%)
updating: README_SUMMARY_TR.md (deflated 47%)
updating: mertformer_sdk/ (stored 0%)
updating: mertformer_sdk/pilot.py (deflated 68%)
updating: mertformer_sdk/kernels/ (stored 0%)
updating: mertformer_sdk/kernels/dispatcher.py (deflated 64%)
updating: mertformer_sdk/kernels/metal/ (stored 0%)
updating: mertformer_sdk/kernels/metal/__init__.py (deflated 19%)
updating: mertformer_sdk/kernels/metal/engine.py (deflated 41%)
updating: mertformer_sdk/kernels/__init__.py (deflated 48%)
updating: mertformer_sdk/kernels/npu/ (stored 0%)
updating: mertformer_sdk/kernels/npu/__init__.py (deflated 15%)
updating: mertformer_sdk/kernels/npu/engine.py (deflated 40%)
updating: mertformer_sdk/kernels/onnx_custom_op.py (deflated 53%)
updating: mertformer_sdk/kernels/cpp/ (stored 0%)
updating: mertformer_sdk/kernels/cpp/__init__.py (stored 0%)
updating: mertformer_sdk/kernels/cpp/bitnet_cpu.cpp (deflated 38%)
updating: mertformer_sdk/kernels/cpp/loader.py (deflated 55%)
updating: mertformer_sdk/kernels/vulkan/ (stored 0%)
updating: mertformer_sdk/kernels/vulkan/__init__.py (deflated 20%)
updating: mertformer_sdk/kernels/vulkan/engine.py (deflated 40%)
updating: mertformer_sdk/kernels/triton_ternary.py (deflated 75%)
updating: mertformer_sdk/__init__.py (deflated 12%)
updating: mertformer_sdk/utils/ (stored 0%)
updating: mertformer_sdk/utils/__init__.py (deflated 20%)
updating: mertformer_sdk/utils/bitpack.py (deflated 58%)
updating: mertformer_sdk/utils/onnx_meta.py (deflated 47%)
updating: mertformer_sdk/export.py (deflated 46%)
updating: mertformer_sdk/api.py (deflated 66%)
updating: mertformer_sdk/cli.py (deflated 72%)
updating: mertformer_sdk/kpi.py (deflated 68%)
updating: logs/ (stored 0%)
updating: logs/verify/ (stored 0%)
updating: logs/verify/closure_57_matrix.verify_TR.md (deflated 71%)
updating: logs/verify/closure_57_matrix.verify.json (deflated 88%)
updating: logs/verify/closure_57_matrix.verify.md (deflated 71%)
updating: logs/operator_mode/ (stored 0%)
updating: logs/operator_mode/operator_2026-02-20_00-20-52.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-03-04_03-55-26.jsonl (deflated 60%)
updating: logs/operator_mode/operator_2026-03-04_03-20-41.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-03-04_03-48-22.jsonl (deflated 60%)
updating: logs/operator_mode/ALL_LOGS.jsonl (deflated 95%)
updating: logs/operator_mode/operator_2026-02-12_23-01-14.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-18_11-00-42.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-18_05-28-34.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-12_23-27-48.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-03-04_03-55-26.manifest.json (deflated 60%)
updating: logs/operator_mode/operator_2026-02-12_21-04-43.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-18_10-47-24.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-12_23-44-23.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-18_05-26-23.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-03-04_03-52-10.manifest.json (deflated 60%)
updating: logs/operator_mode/operator_2026-02-20_04-50-54.jsonl (deflated 58%)
updating: logs/operator_mode/operator_2026-02-18_18-04-52.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-19_22-18-22.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-03-04_03-49-07.jsonl (deflated 60%)
updating: logs/operator_mode/operator_2026-02-19_22-17-36.manifest.json (deflated 51%)
updating: logs/operator_mode/operator_2026-02-18_09-24-45.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-18_09-28-09.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-20_04-48-34.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-18_10-42-53.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-18_05-24-38.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-22_05-25-57.jsonl (deflated 58%)
updating: logs/operator_mode/operator_2026-02-18_05-19-10.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-03-04_03-49-07.manifest.json (deflated 60%)
updating: logs/operator_mode/operator_2026-02-12_20-47-58.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-18_05-34-58.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-03-04_03-50-21.manifest.json (deflated 60%)
updating: logs/operator_mode/operator_2026-03-04_03-50-53.jsonl (deflated 60%)
updating: logs/operator_mode/operator_2026-02-11_23-09-30.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-19_22-07-43.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-12_23-35-01.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-03-04_03-20-41.jsonl (deflated 58%)
updating: logs/operator_mode/operator_2026-02-22_05-23-21.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-20_00-17-32.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-03-04_03-50-21.jsonl (deflated 60%)
updating: logs/operator_mode/operator_2026-02-11_22-28-43.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-19_21-37-29.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-03-04_03-48-22.manifest.json (deflated 60%)
updating: logs/operator_mode/operator_2026-02-20_04-50-54.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-20_01-47-42.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-03-04_03-50-53.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-13_00-27-08.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-12_20-59-30.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-12_23-21-41.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-20_04-47-05.jsonl (deflated 58%)
updating: logs/operator_mode/operator_2026-03-04_03-51-24.jsonl (deflated 60%)
updating: logs/operator_mode/operator_2026-02-20_00-10-49.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-18_04-26-15.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-18_05-33-55.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-03-04_03-51-24.manifest.json (deflated 60%)
updating: logs/operator_mode/operator_2026-02-20_03-51-30.jsonl (deflated 58%)
updating: logs/operator_mode/operator_2026-02-20_04-45-54.jsonl (deflated 58%)
updating: logs/operator_mode/operator_2026-02-20_01-47-42.jsonl (deflated 58%)
updating: logs/operator_mode/operator_2026-03-04_03-52-10.jsonl (deflated 60%)
updating: logs/operator_mode/operator_2026-02-19_22-04-28.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-20_04-47-05.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-18_09-20-10.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-19_22-17-10.manifest.json (deflated 50%)
updating: logs/operator_mode/operator_2026-02-20_04-48-34.jsonl (deflated 58%)
updating: logs/operator_mode/operator_2026-02-20_03-51-30.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-22_05-23-21.jsonl (deflated 58%)
updating: logs/operator_mode/operator_2026-02-11_22-32-09.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-13_00-24-10.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-20_04-45-54.manifest.json (deflated 50%)
updating: logs/operator_mode/operator_2026-02-18_17-58-46.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-22_05-25-57.manifest.json (deflated 59%)
updating: logs/operator_mode/operator_2026-02-18_04-57-01.manifest.json (deflated 59%)
updating: logs/README.md (deflated 48%)
updating: logs/README_TR.md (deflated 47%)
updating: logs/preflight/ (stored 0%)
updating: logs/preflight/train_ready_status.json (deflated 62%)
updating: logs/preflight/titan_preflight.log (deflated 69%)
updating: SDK_GUIDE.md (deflated 55%)
updating: README_TR.md (deflated 69%)
updating: .gitignore (deflated 56%)
updating: snake_demo.py (deflated 67%)
updating: README_SUMMARY.md (deflated 48%)
updating: .env (deflated 12%)
updating: INTERNAL_AGI_GAP.md (deflated 53%)
updating: CONTRIBUTING.md (deflated 37%)
updating: repro/ (stored 0%)
updating: repro/accelerate_default.yaml (deflated 30%)
updating: repro/python_TR.md (deflated 44%)
updating: repro/python.md (deflated 46%)
updating: repro/cuda.lock (deflated 40%)
updating: repro/seed_policy_TR.md (deflated 33%)
updating: repro/seed_policy.md (deflated 35%)
updating: repro/pip_freeze.txt (deflated 39%)
updating: repro/env.lock (deflated 6%)
updating: README_SUMMARY.pdf (deflated 18%)
updating: scripts/ (stored 0%)
updating: scripts/record_dataset_hashes.py (deflated 65%)
updating: scripts/generate_energy_baselines.py (deflated 57%)
updating: scripts/check_translation_pointer_policy.py (deflated 58%)
updating: scripts/hash_manifest_to_json.py (deflated 57%)
updating: scripts/extract_dataset_refs.py (deflated 64%)
updating: scripts/titan_onnx_stress_test.py (deflated 56%)
updating: scripts/test_onnx_export.py (deflated 64%)
updating: scripts/mac_simulation.py (deflated 60%)
updating: scripts/update_system_hardware.py (deflated 71%)
updating: scripts/apply_github_policy.sh (deflated 49%)
updating: scripts/hardening_bundle.py (deflated 71%)
updating: scripts/linkcheck_gate.py (deflated 60%)
updating: scripts/benchmarks_internal.py (deflated 66%)
updating: scripts/drone_sitl_demo.py (deflated 67%)
updating: scripts/data_pipeline.py (deflated 68%)
updating: scripts/version_checker.py (deflated 54%)
updating: scripts/cleanroom_verify.sh (deflated 59%)
updating: scripts/docs_inventory.py (deflated 56%)
updating: scripts/golden_eval.py (deflated 58%)
updating: scripts/clean_runtime_artifacts.sh (deflated 68%)
updating: scripts/secret_scan.py (deflated 54%)
updating: scripts/train_smoke.py (deflated 63%)
updating: scripts/check_57_matrix.py (deflated 76%)
updating: scripts/kaggle_onefile_demo_build30.py (deflated 78%)
updating: scripts/mobile_export.py (deflated 68%)
updating: scripts/build_validation_set.py (deflated 71%)
updating: scripts/build_summary_pdf.py (deflated 68%)
updating: scripts/bitnet_kernel_benchmark_standalone.py (deflated 74%)
updating: scripts/sync_manifest.py (deflated 68%)
updating: scripts/__init__.py (stored 0%)
updating: scripts/train_tpu_turbo.py (deflated 61%)
updating: scripts/generate_bench_reports.py (deflated 59%)
updating: scripts/final_one_shot.sh (deflated 70%)
updating: scripts/nan_kill_test.py (deflated 36%)
updating: scripts/bootstrap_venv.sh (deflated 57%)
updating: scripts/write_cuda_lock.py (deflated 62%)
updating: scripts/verify_all.sh (deflated 60%)
updating: scripts/README.md (deflated 53%)
updating: scripts/md_integrity_check.py (deflated 57%)
updating: scripts/operator_mode_gate.py (deflated 70%)
updating: scripts/kaggle_train_compare_build30.py (deflated 73%)
updating: scripts/scaling_audit_math.py (deflated 64%)
updating: scripts/chat.py (deflated 64%)
updating: scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py (deflated 77%)
updating: scripts/README_TR.md (deflated 51%)
updating: scripts/check_doc_claim_consistency.py (deflated 62%)
updating: scripts/ram_guard.py (deflated 61%)
updating: scripts/smoke_train_benchmark.py (deflated 63%)
updating: scripts/logbook_build.py (deflated 72%)
updating: scripts/verify_datasets.py (deflated 63%)
updating: scripts/dealroom_sync.py (deflated 61%)
updating: scripts/release_closure_lock.sh (deflated 37%)
updating: scripts/mini_titan_poc.py (deflated 68%)
updating: scripts/smart_runner.py (deflated 60%)
updating: scripts/build_investor_deck.py (deflated 72%)
updating: scripts/xray.py (deflated 65%)
updating: scripts/overfit_gate.py (deflated 67%)
updating: scripts/eval.py (deflated 57%)
updating: scripts/failure_budget_drill.py (deflated 52%)
updating: scripts/titan_preflight.py (deflated 68%)
updating: scripts/repro_build_check.py (deflated 49%)
updating: scripts/unicode_path_guard.py (deflated 51%)
updating: scripts/md_quality_gate.py (deflated 61%)
updating: scripts/release_build30.sh (deflated 63%)
updating: scripts/generate_sbom.py (deflated 55%)
updating: scripts/runs/ (stored 0%)
updating: scripts/runs/preflight/ (stored 0%)
updating: scripts/runs/preflight/config_snapshot.json (deflated 58%)
updating: scripts/start_gate.py (deflated 52%)
updating: scripts/download_tr_tokenizer.py (deflated 44%)
updating: scripts/md_build30_sweep.py (deflated 58%)
updating: scripts/golden_score.py (deflated 69%)
updating: scripts/zip_denylist_audit.py (deflated 63%)
updating: scripts/verify_onnx_local.py (deflated 58%)
updating: scripts/generate_demo_bundle.py (deflated 67%)
updating: scripts/auto_demo_video.py (deflated 63%)
updating: scripts/check_tokenizer_sync.py (deflated 59%)
updating: scripts/reports/ (stored 0%)
updating: scripts/reports/pilots/ (stored 0%)
updating: scripts/reports/pilots/pilot_001/ (stored 0%)
updating: scripts/reports/pilots/pilot_001/sitl_20260211T214734Z/ (stored 0%)
updating: scripts/reports/pilots/pilot_001/sitl_20260211T214734Z/sitl_events.jsonl (deflated 95%)
updating: scripts/reports/pilots/pilot_001/sitl_20260211T214734Z/sitl_report.md (deflated 51%)
updating: scripts/reports/pilots/pilot_001/sitl_20260211T214734Z/sitl_summary.json (deflated 70%)
updating: scripts/reports/model_health_TR.md (deflated 24%)
updating: scripts/reports/model_health.md (deflated 24%)
updating: scripts/duplicate_zip_guard.py (deflated 53%)
updating: scripts/checkpoint_restore_drill.py (deflated 66%)
updating: packages/ (stored 0%)
updating: packages/MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip (stored 0%)
updating: .github/ (stored 0%)
updating: .github/CODEOWNERS (deflated 53%)
updating: .github/workflows/ (stored 0%)
updating: .github/workflows/ci.yml (deflated 65%)
updating: model/ (stored 0%)
updating: model/transformers.py (deflated 70%)
updating: model/__init__.py (stored 0%)
updating: economics/ (stored 0%)
updating: economics/cost_model.md (deflated 32%)
updating: economics/efficiency_report.md (deflated 27%)
updating: economics/cost_model_TR.md (deflated 28%)
updating: economics/flops_estimator.py (deflated 54%)
updating: economics/efficiency_report_TR.md (deflated 23%)
updating: train/ (stored 0%)
updating: train/continual_adapter.py (deflated 68%)
updating: train/__init__.py (stored 0%)
updating: train/train.py (deflated 75%)
updating: README_CHECKLIST.md (deflated 44%)
updating: TRAINING_PLAN_TR.md (deflated 48%)
updating: registry/ (stored 0%)
updating: registry/mertformer_v0.1.json (deflated 31%)
updating: prompts/ (stored 0%)
updating: prompts/changelog.md (deflated 3%)
updating: prompts/changelog_TR.md (deflated 2%)
updating: prompts/system_v1.txt (deflated 26%)
updating: mertformer_sdk.egg-info/ (stored 0%)
updating: mertformer_sdk.egg-info/PKG-INFO (deflated 68%)
updating: mertformer_sdk.egg-info/SOURCES.txt (deflated 64%)
updating: mertformer_sdk.egg-info/entry_points.txt (deflated 15%)
updating: mertformer_sdk.egg-info/requires.txt (deflated 28%)
updating: mertformer_sdk.egg-info/top_level.txt (stored 0%)
updating: mertformer_sdk.egg-info/dependency_links.txt (stored 0%)
updating: TASK.md (deflated 49%)
updating: V2_BACKLOG_SEED.md (deflated 41%)
updating: telemetry/ (stored 0%)
updating: telemetry/metrics_schema.json (deflated 58%)
updating: eval/ (stored 0%)
updating: eval/report_builder.py (deflated 69%)
updating: eval/generalization_suite.py (deflated 63%)
updating: eval/agentic_suite.py (deflated 69%)
updating: eval/gsm8k.py (deflated 66%)
updating: eval/golden.py (deflated 35%)
updating: eval/humaneval.py (deflated 36%)
updating: CONTRIBUTING_TR.md (deflated 30%)
updating: TASK_TR.md (deflated 46%)
updating: data/ (stored 0%)
updating: data/tokenizer/ (stored 0%)
updating: data/tokenizer/tr/ (stored 0%)
updating: data/tokenizer/tr/tokenizer_config.json (deflated 75%)
updating: data/tokenizer/tr/special_tokens_map.json (deflated 42%)
updating: data/tokenizer/tr/tokenizer.json (deflated 70%)
updating: data/tokenizer/tr/vocab.txt (deflated 56%)
updating: assets/ (stored 0%)
updating: assets/synaptic_map.png (deflated 0%)
updating: assets/snake_demo_proof.mp4 (deflated 5%)
updating: assets/header.png (deflated 0%)
updating: assets/sources/ (stored 0%)
updating: assets/sources/README.md (deflated 46%)
updating: assets/sources/README_TR.md (deflated 47%)
updating: assets/snake_demo_preview.gif (deflated 4%)
updating: LICENSE_TR (deflated 50%)
updating: IMPLEMENTATION_PLAN.md (deflated 49%)
updating: IMPLEMENTATION_PLAN_TR.md (deflated 44%)
updating: USE_POLICY_TR.md (deflated 41%)
updating: orchestrator/ (stored 0%)
updating: orchestrator/planner.py (deflated 65%)
updating: orchestrator/experience_store.py (deflated 71%)
updating: orchestrator/reasoning_engine.py (deflated 73%)
updating: orchestrator/compute_orchestrator.py (deflated 58%)
updating: orchestrator/agent_registry.py (deflated 72%)
updating: orchestrator/failure_budget.py (deflated 70%)
updating: orchestrator/distillation_manager.py (deflated 71%)
updating: orchestrator/paths.py (deflated 56%)
updating: orchestrator/memory.py (deflated 72%)
updating: orchestrator/tool_registry.py (deflated 58%)
updating: orchestrator/__init__.py (deflated 65%)
updating: orchestrator/telemetry.py (deflated 66%)
updating: orchestrator/core.py (deflated 72%)
updating: orchestrator/self_audit.py (deflated 69%)
updating: orchestrator/cognitive_loop.py (deflated 73%)
updating: orchestrator/alignment_contracts.py (deflated 61%)
updating: orchestrator/sense_engine.py (deflated 69%)
updating: orchestrator/self_improvement_guard.py (deflated 72%)
updating: orchestrator/web_sense.py (deflated 56%)
updating: orchestrator/hardware.py (deflated 54%)
updating: orchestrator/swarm_runtime.py (deflated 68%)
updating: orchestrator/tool_executor.py (deflated 74%)
updating: orchestrator/cognitive.py (deflated 67%)
updating: orchestrator/verifier.py (deflated 72%)
updating: orchestrator/audio_sense.py (deflated 62%)
updating: orchestrator/governance.py (deflated 62%)
updating: policy/ (stored 0%)
updating: policy/allow_deny_policy.yaml (deflated 59%)
updating: SECURITY.md (deflated 42%)
updating: USAGE_GUIDE_TR.md (deflated 46%)
updating: interfaces/ (stored 0%)
updating: interfaces/inference_contract.md (deflated 31%)
updating: interfaces/inference_contract_TR.md (deflated 28%)
updating: interfaces/pilot_report_v1.schema.json (deflated 83%)
updating: interfaces/kpi_report_v1.schema.json (deflated 67%)
updating: interfaces/tokenizer_spec.json (deflated 27%)
updating: interfaces/closure_57_matrix_v1.schema.json (deflated 77%)
updating: reports/ (stored 0%)
updating: reports/codex_deep_audit_EN.md (deflated 58%)
updating: reports/pilot_offer_packages.md (deflated 44%)
updating: reports/release_snapshot.md (deflated 53%)
updating: reports/verified_matrix.md (deflated 53%)
updating: reports/sales_funnel_90d_TR.md (deflated 36%)
updating: reports/technical_snapshot.md (deflated 47%)
updating: reports/pitch_kit/ (stored 0%)
updating: reports/pitch_kit/04_KAYNAK_README_TR.md (deflated 70%)
updating: reports/pitch_kit/01_VIZYON_Buluttan_Bagimsiz_Ozet.m4a (deflated 27%)
updating: reports/pitch_kit/04_KAYNAK_README.md (deflated 71%)
updating: reports/pitch_kit/02_STRATEJI_Sovereign_Edge_AI.pdf (deflated 0%)
updating: reports/pitch_kit/03_TEKNIK_Derin_Muhendislik.m4a (deflated 26%)
updating: reports/pitch_kit/00_BENI_OKU_Inceleme_Sirasi.txt (deflated 56%)
updating: reports/contamination_report_build30.md (deflated 45%)
updating: reports/sbom.cdx.json (deflated 85%)
updating: reports/drone_sitl_demo_TR.md (deflated 44%)
updating: reports/docs_packages_hash_manifest.json (deflated 31%)
updating: reports/release_snapshot_TR.md (deflated 50%)
updating: reports/sales_funnel_90d.md (deflated 38%)
updating: reports/pilot_readiness_kit.md (deflated 44%)
updating: reports/kpi_contract_build30.md (deflated 50%)
updating: reports/efficiency_convergence_analysis.md (deflated 51%)
updating: reports/dataset_health_TR.md (deflated 53%)
updating: reports/demo_video_script_TR.md (deflated 41%)
updating: reports/strategic_value.md (deflated 43%)
updating: reports/zip_denylist_audit_release_v2.md (deflated 47%)
updating: reports/startup_selfcheck_report.json (deflated 7%)
updating: reports/codex_deep_audit_TR.md (deflated 56%)
updating: reports/snapshots/ (stored 0%)
updating: reports/snapshots/2026-02-24/ (stored 0%)
updating: reports/snapshots/2026-02-24/mertformer_master_decision_report_TR_2026-02-24.md (deflated 53%)
updating: reports/snapshots/2026-02-24/web_validation_sources_2026-02-24.md (deflated 54%)
updating: reports/snapshots/2026-02-24/readiness_scorecard_v1_2026-02-24.json (deflated 60%)
updating: reports/snapshots/2026-02-24/claim_matrix_v2_2026-02-24.json (deflated 68%)
updating: reports/snapshots/2026-02-24/evidence_snapshot_2026-02-24.json (deflated 60%)
updating: reports/snapshots/2026-02-24/commercial_scenarios_v1_2026-02-24.json (deflated 55%)
updating: reports/snapshots/2026-02-24/report_interface_schema_v1.json (deflated 54%)
updating: reports/start_gate_report.json (deflated 63%)
updating: reports/fallback_policy_report.json (deflated 11%)
updating: reports/closure_57_matrix_TR.md (deflated 71%)
updating: reports/folder_structure_policy.md (deflated 37%)
updating: reports/cleanroom_verification_TR.md (deflated 49%)
updating: reports/zip_denylist_audit_release_v2.json (deflated 52%)
updating: reports/go_status_matrix.md (deflated 56%)
updating: reports/pilot_acceptance_signoff_TR.md (deflated 46%)
updating: reports/ownership_proof_bundle.json (deflated 56%)
updating: reports/differential_backend_report.json (deflated 6%)
updating: reports/policy_sync_report.json (deflated 21%)
updating: reports/go_nogo_signoff_onepager.md (deflated 42%)
updating: reports/strategic_value_TR.md (deflated 40%)
updating: reports/file_sync_matrix.json (deflated 32%)
updating: reports/go_nogo_signoff_onepager_TR.md (deflated 39%)
updating: reports/asset_stack.md (deflated 67%)
updating: reports/cli_smoke_log.md (deflated 27%)
updating: reports/final_sync_matrix.md (deflated 51%)
updating: reports/system_hardware.md (deflated 30%)
updating: reports/ip_licensing_split_TR.md (deflated 50%)
updating: reports/technical_snapshot_TR.md (deflated 40%)
updating: reports/static_analysis_report.json (deflated 23%)
updating: reports/pilots/ (stored 0%)
updating: reports/pilots/README.md (deflated 45%)
updating: reports/pilots/README_TR.md (deflated 44%)
updating: reports/poc_protocol.md (deflated 47%)
updating: reports/codex_deep_audit_DE.md (deflated 55%)
updating: reports/efficiency_convergence_analysis_TR.md (deflated 51%)
updating: reports/pilot_readiness_kit_TR.md (deflated 43%)
updating: reports/system_hardware_TR.md (deflated 29%)
updating: reports/teacher_output_license_assessment.md (deflated 46%)
updating: reports/determinism_report.json (deflated 6%)
updating: reports/security_compliance_TR.md (deflated 45%)
updating: reports/kpi_report_v1.json (deflated 68%)
updating: reports/demo_video_script.md (deflated 44%)
updating: reports/md_lint_report.json (deflated 86%)
updating: reports/model_health_TR.md (deflated 24%)
updating: reports/drone_sitl_demo.md (deflated 45%)
updating: reports/thermal_baseline.json (deflated 2%)
updating: reports/kpi_pack_v1.md (deflated 41%)
updating: reports/unicode_path_guard_report.json (deflated 20%)
updating: reports/release_manifest.json (deflated 74%)
updating: reports/project_structure_sync_report.json (deflated 47%)
updating: reports/verified_matrix_TR.md (deflated 52%)
updating: reports/hardening_bundle_summary.json (deflated 47%)
updating: reports/one_pager_TR.md (deflated 42%)
updating: reports/sanitizer_report.json (deflated 22%)
updating: reports/report_accuracy_audit.md (deflated 50%)
updating: reports/final_sync_matrix_TR.md (deflated 49%)
updating: reports/folder_drift_report.json (deflated 78%)
updating: reports/benchmarks/ (stored 0%)
updating: reports/benchmarks/summary.json (deflated 66%)
updating: reports/benchmarks/generalization_suite_build30.json (deflated 71%)
updating: reports/benchmarks/internal_smoke_summary.json (deflated 6%)
updating: reports/benchmarks/kaggle_compare_build30.json (deflated 59%)
updating: reports/benchmarks/README.md (deflated 50%)
updating: reports/benchmarks/smoke_train_metrics.json (deflated 47%)
updating: reports/benchmarks/README_TR.md (deflated 46%)
updating: reports/benchmarks/kaggle_compare_build30.csv (deflated 38%)
updating: reports/benchmarks/kaggle_compare_build30.md (deflated 31%)
updating: reports/benchmarks/agentic_suite_build30.json (deflated 73%)
updating: reports/bench_npu_report.json (deflated 9%)
updating: reports/investor_deck.pptx (deflated 29%)
updating: reports/cleanroom_verification.md (deflated 51%)
updating: reports/demo_script.md (deflated 31%)
updating: reports/go_status_matrix_TR.md (deflated 53%)
updating: reports/poc_protocol_TR.md (deflated 44%)
updating: reports/review_checklist_TR.md (deflated 47%)
updating: reports/bench_vulkan_report.json (deflated 9%)
updating: reports/system_stats.jsonl (deflated 67%)
updating: reports/runbook_validation_report.json (deflated 10%)
updating: reports/demo_validation_report.json (deflated 35%)
updating: reports/demo_checksum.sha256 (deflated 10%)
updating: reports/investor_deck_TR.pptx (deflated 29%)
updating: reports/report_accuracy_audit_TR.md (deflated 47%)
updating: reports/codex_deep_audit_EN_TR.md (deflated 40%)
updating: reports/dealroom_reference.json (deflated 41%)
updating: reports/one_pager.md (deflated 48%)
updating: reports/ram_guard_report.json (deflated 39%)
updating: reports/pilot_acceptance_signoff.md (deflated 50%)
updating: reports/review_checklist.md (deflated 50%)
updating: reports/bench_metal_report.json (deflated 8%)
updating: reports/demo_notes.md (deflated 21%)
updating: reports/repro_build_report.json (deflated 52%)
updating: reports/ip_licensing_split.md (deflated 51%)
updating: reports/codex_deep_audit_DE_TR.md (deflated 40%)
updating: reports/duplicate_zip_guard_report.json (deflated 29%)
updating: reports/kernel_fuzz_report.json (deflated 12%)
updating: reports/bench_zero_copy_report.json (deflated 39%)
updating: reports/bench_cpp_report.json (deflated 11%)
updating: reports/dataset_health.md (deflated 53%)
updating: reports/energy_baseline.json (deflated 22%)
updating: reports/linkcheck_report.json (deflated 26%)
updating: reports/closure_57_matrix.md (deflated 71%)
updating: reports/license_gate_report.json (deflated 15%)
updating: reports/pilot_offer_packages_TR.md (deflated 41%)
updating: reports/closure_57_matrix.json (deflated 88%)
updating: reports/docs_dedup_canonical_list.md (deflated 83%)
updating: reports/backup_restore_report.json (deflated 1%)
updating: reports/latency_baseline.json (deflated 3%)
updating: reports/cli_smoke_log_TR.md (deflated 24%)
updating: reports/model_health.md (deflated 24%)
updating: reports/kpi_pack_v1_TR.md (deflated 38%)
updating: reports/security_compliance.md (deflated 49%)
updating: reports/founders_hub_application.md (deflated 46%)
updating: reports/asset_stack_TR.md (deflated 64%)
updating: reports/founders_hub_application_TR.md (deflated 41%)
updating: reports/legal_cleanroom_signoff_internal.md (deflated 50%)
updating: DECISIONS.md (deflated 29%)
updating: MODEL_CARD_TR.md (deflated 40%)
  adding: MertFormer_Titan_Dealroom_2026-02-23/MANIFEST/main_release_reference_v1.0.0.json (deflated 38%)
  adding: logs/operator_mode/operator_2026-03-04_04-09-13.manifest.json (deflated 60%)
  adding: logs/operator_mode/operator_2026-03-04_03-58-26.jsonl (deflated 60%)
  adding: logs/operator_mode/operator_2026-03-04_03-58-26.manifest.json (deflated 60%)
  adding: logs/operator_mode/operator_2026-03-04_04-09-13.jsonl (deflated 60%)
  adding: logs/operator_mode/operator_2026-03-04_04-07-02.jsonl (deflated 60%)
  adding: logs/operator_mode/operator_2026-03-04_04-07-38.manifest.json (deflated 60%)
  adding: logs/operator_mode/operator_2026-03-04_04-07-38.jsonl (deflated 60%)
  adding: logs/operator_mode/operator_2026-03-04_04-07-02.manifest.json (deflated 60%)
  adding: reports/execution_trace.json (deflated 24%)
  adding: reports/github_policy_report.json (deflated 13%)
  adding: reports/snapshot_manifest_main.json (deflated 45%)
  adding: reports/final_repo_audit.md (deflated 52%)
  adding: reports/release_closure_note.md (deflated 15%)
  adding: reports/release_closure_lock_report.json (deflated 19%)
  adding: reports/snapshot_manifest_dealroom.json (deflated 48%)
github policy: skipped_or_failed
release closure lock prepared for v1.0.0
[final] COMPLETED.
- Main repo release branch: .
- Dealroom repo release branch: .

## Gates
- : pass
- : pass (, no in-scope pending)
- [verify] Python: Python 3.11.14
[verify] TITAN_OFFLINE=1
[verify] Secret scan ...
OK: no secret patterns detected in tracked files.
[verify] Pytest ...
................................s....................................... [ 63%]
s........................................                                [100%]
=============================== warnings summary ===============================
.titan-venv/lib/python3.11/site-packages/torch/jit/_script.py:1480
.titan-venv/lib/python3.11/site-packages/torch/jit/_script.py:1480
.titan-venv/lib/python3.11/site-packages/torch/jit/_script.py:1480
  /Users/mertyunlu/Desktop/mertformer-titan-core/.titan-venv/lib/python3.11/site-packages/torch/jit/_script.py:1480: DeprecationWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

scripts/test_onnx_export.py::test_export
  /opt/homebrew/Cellar/python@3.11/3.11.14_3/Frameworks/Python.framework/Versions/3.11/lib/python3.11/copyreg.py:105: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    return cls.__new__(cls, *args)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
111 passed, 3 skipped, 6 warnings in 13.17s
[verify] Preflight (offline) ...
🔄 Resuming from Chunk 0
[verify] Operator mode gate (safe, offline) ...
Checkpoint restore drill: PASS
Failure budget drill: PASS
Overfit step 0: loss=121.5519
Overfit step 12: loss=117.8494
Overfit step 24: loss=113.7169
Overfit step 36: loss=97.5477
Overfit step 48: loss=50.8628
Overfit gate: PASS
Golden samples: 50
- algorithms: 13
- cli: 1
- concurrency: 2
- config: 1
- data: 3
- data_structures: 4
- database: 1
- debugging: 2
- errors: 1
- files: 1
- logging: 1
- math: 1
- metrics: 1
- ml: 1
- networking: 2
- parsing: 1
- performance: 2
- refactor: 2
- regex: 1
- security: 2
- serialization: 1
- streaming: 1
- system: 1
- tests: 2
- text: 1
- time: 1
Golden sample eval: PASS
Benchmark runner configured. Use --run to execute.
{
  "status": "completed",
  "elapsed_sec": 7.746546030044556,
  "results": [
    {
      "step": "nan_kill_switch",
      "status": "pass"
    },
    {
      "step": "checkpoint_restore_drill",
      "status": "pass"
    },
    {
      "step": "failure_budget_drill",
      "status": "pass"
    },
    {
      "step": "overfit_gate",
      "status": "pass_fast"
    },
    {
      "step": "golden_samples",
      "status": "pass"
    },
    {
      "step": "telemetry_snapshot",
      "status": "pass",
      "snapshot": {
        "timestamp_utc": "2026-03-04T01:10:05Z",
        "cpu_percent": 26.1,
        "ram_used_gb": 5.3788299560546875,
        "ram_total_gb": 16.0
      }
    },
    {
      "step": "benchmarks",
      "status": "ready"
    }
  ]
}
[verify] Closure 57 matrix gate (strict in-scope pending) ...
{"total_items": 57, "green_items": 57, "all_green": true, "no_pending_in_scope": true, "in_scope_pending_ids": [], "out_of_scope_pending_ids": [8, 9, 11, 12, 51, 52, 54, 55, 56, 57], "evidence_pending_ids": [8, 9, 11, 12, 51, 52, 54, 55, 56, 57]}
[verify] Tokenizer sync gate ...
canonical=/Users/mertyunlu/Desktop/mertformer-titan-core/interfaces/tokenizer_spec.json
mirror=/Users/mertyunlu/Desktop/mertformer-titan-core/tokenizer/tokenizer.json
canonical_sha256=21d29a6ad849dd9ef567e5d750e15ac08d4d445bd0985bb736f40c66a18fa95e
mirror_sha256=21d29a6ad849dd9ef567e5d750e15ac08d4d445bd0985bb736f40c66a18fa95e
OK: tokenizer spec files are byte-identical
[verify] Translation pointer policy gate ...
OK: translation pointer policy satisfied
[verify] Documentation claim consistency gate ...
OK: documentation claim consistency checks passed
[verify] Unicode path guard gate ...
OK: unicode path guard; hits=0
[verify] Duplicate zip guard gate ...
OK: duplicate zip guard; groups=0
[verify] Manifest sync gate ...
{"manifest_entries": 653, "ok": true}
[verify] OK: pass
- Unicode path guard: pass
- Duplicate zip guard: pass
- Manifest sync guard: pass

## Security / Hardening Outputs
- SBOM: 
- Static analysis: 
- Sanitizer smoke: 
- Kernel fuzz smoke: 
- Determinism: 
- Differential backend: 
- License gate: 

## Release Artifacts
- 
- 
- 
- 
- 

## Provenance
- Main signed commit and signed tag: 
- Main PR merges: , 
- Dealroom signed commit and signed tag: 
- Dealroom PR merge: 
- Ownership bundle: 

## Notes
- GitHub branch-protection API update returned 403 on plan-limited account tier; report recorded in .
- Real 2.64B training not started in this closure pass.
