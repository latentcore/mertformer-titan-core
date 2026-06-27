# Code-Truth Delta Audit

Current repo-side code-truth audit for the closure pass. This report does not upgrade any claim by rhetoric; it labels maturity, evidence shape, and marker risk explicitly.

## Done Rule
A closure-critical item is only done when code path, canonical command, verification, and artifact/report evidence all exist together.

Required evidence columns:
- `code_path`
- `canonical_command`
- `verification`
- `artifact`

## Surface Groups
### no-touch
- `AGENTS.md`
- `reports/final_backlog_classification.md`
- `reports/final_freeze_manifest.md`
- `reports/source_of_truth_map.md`
- `reports/final_truth_matrix.md`

### high-risk
- `layers/bitlinear.py`
- `mertformer_sdk/kernels/dispatcher.py`
- `mertformer_sdk/kernels/triton_ternary.py`
- `mertformer_sdk/kernels/cpp/bitnet_cpu.cpp`
- `mertformer_sdk/kernels/metal/engine.py`
- `scripts/chess_5080_onefile.py`

### closure
- `zero_touch_start.sh`
- `scripts/verify_all.sh`
- `scripts/final_orchestrator.py`
- `scripts/post_train_autorun.py`
- `scripts/build_closure_governance_pack.py`

### research
- `scripts/train_tpu_turbo.py`
- `reports/phase2_carryover.md`
- `CHESS_5080_POC_INTERNAL_TR.md`
- `OFFLINE_4060_DEMO.md`

### desktop-hygiene
- `scripts/build_workspace_hygiene_manifest.py`
- `interfaces/workspace_hygiene_manifest_v1.schema.json`
- `reports/workspace_hygiene_manifest.md`
- `reports/workspace_hygiene_manifest.json`

## Technical Surface Maturity

| Path | Lane | Surface Class | Maturity | Markers | Evidence Complete | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `layers/bitlinear.py` | `bitnet-kernel` | `living` | `tested_fallback` | `fallback` | `true` | BitLinear is correctness-first and dispatch-aware, but production-depth performance claims still depend on backend-specific measurement. |
| `mertformer_sdk/kernels/dispatcher.py` | `bitnet-kernel` | `maintained` | `tested_fallback` | `fallback` | `true` | Dispatcher routing is deterministic and test-covered; it is a contract surface rather than a speed claim by itself. |
| `mertformer_sdk/kernels/triton_ternary.py` | `bitnet-performance` | `living` | `tested_fallback` | `fallback` | `true` | The Triton kernel is explicitly experimental; correctness is partially covered, but it is not yet a release-grade performance claim surface. |
| `mertformer_sdk/kernels/cpp/bitnet_cpu.cpp` | `cpu-reference` | `maintained` | `reference_safe` | `scaffold`, `fallback` | `true` | This file is a minimal CPU reference kernel and should remain a parity/debug surface, not a production-depth speed claim. |
| `mertformer_sdk/kernels/metal/engine.py` | `mps-metal` | `living` | `tested_fallback` | `fallback` | `true` | Metal currently routes through deterministic PyTorch fallback math and must not be narrated as a custom optimized kernel path. |
| `scripts/chess_5080_onefile.py` | `chess-proof` | `living` | `tested_fallback` | `fallback` | `true` | The chess onefile is a real code path with tests and delivery helpers, but it remains a proof/product baseline rather than a solved final product lane. |
| `scripts/export_chess_5080_share.py` | `chess-proof` | `maintained` | `tested_fallback` | none | `true` | Delivery/export logic is implemented and tested, but external product-grade distribution still depends on trained outputs and operator validation. |

## Doc-to-Code Crosswalk

| Doc | Claim | Code Path | Canonical Command | Verification | Artifact |
| --- | --- | --- | --- | --- | --- |
| `README.md` | Canonical verification and train-end entrypoints are real code paths. | `scripts/verify_all.sh`, `zero_touch_start.sh`, `scripts/final_orchestrator.py` | `bash scripts/verify_all.sh`, `bash zero_touch_start.sh --check-only` | `reports/train_readiness_decision.json`, `reports/start_gate_report.json` | `reports/final_truth_matrix.md`, `reports/canonical_entrypoint.md` |
| `MODEL_CARD.md` | Measured versus target boundaries remain explicit. | `scripts/check_doc_claim_consistency.py`, `scripts/build_closure_governance_pack.py` | `python3 scripts/check_doc_claim_consistency.py`, `python3 scripts/build_closure_governance_pack.py` | `reports/claim_registry.json`, `reports/final_truth_matrix.md` | `reports/code_truth_delta_audit.md` |
| `reports/final_truth_matrix.md` | Closure-critical claims map to evidence instead of living only as prose. | `scripts/build_closure_governance_pack.py`, `scripts/build_code_truth_audit.py` | `python3 scripts/build_closure_governance_pack.py`, `python3 scripts/build_code_truth_audit.py` | `reports/claim_registry.json`, `reports/code_truth_delta_audit.json` | `reports/final_truth_matrix.md`, `reports/code_truth_delta_audit.md` |

## Marker Scan

Marker hits are review prompts, not automatic bug declarations. In particular, dataset compliance tables may intentionally retain `TBD` placeholders until legal/compliance review finishes.

| Path | Markers |
| --- | --- |
| `AGENTS.md` | `fallback` |
| `datasets/INTERNAL_POLICY.md` | `TBD` |
| `datasets/INTERNAL_POLICY_TR.md` | `TBD` |
| `datasets/LICENSES.md` | `TBD` |
| `datasets/LICENSES_TR.md` | `TBD` |
| `datasets/SOURCES.md` | `fallback` |
| `datasets/SOURCES_TR.md` | `fallback` |
| `datasets/inventory.md` | `TBD` |
| `datasets/inventory_TR.md` | `TBD` |
| `layers/bitlinear.py` | `fallback` |
| `layers/ffn.py` | `fallback` |
| `layers/mertformer_block.py` | `fallback` |
| `layers/mla.py` | `fallback` |
| `layers/moe.py` | `fallback` |
| `layers/qinn.py` | `fallback` |
| `mertformer_sdk/api.py` | `fallback` |
| `mertformer_sdk/kernels/cpp/bitnet_cpu.cpp` | `scaffold`, `fallback` |
| `mertformer_sdk/kernels/cpp/loader.py` | `fallback` |
| `mertformer_sdk/kernels/dispatcher.py` | `fallback` |
| `mertformer_sdk/kernels/metal/__init__.py` | `fallback` |
| `mertformer_sdk/kernels/metal/engine.py` | `fallback` |
| `mertformer_sdk/kernels/npu/__init__.py` | `fallback` |
| `mertformer_sdk/kernels/npu/engine.py` | `fallback` |
| `mertformer_sdk/kernels/onnx_custom_op.py` | `fallback` |
| `mertformer_sdk/kernels/triton_fused_bitlinear.py` | `fallback` |
| `mertformer_sdk/kernels/triton_ternary.py` | `fallback` |
| `mertformer_sdk/kernels/vulkan/__init__.py` | `fallback` |
| `mertformer_sdk/kernels/vulkan/engine.py` | `fallback` |
| `mertformer_sdk/kpi.py` | `fallback` |
| `mertformer_sdk/pilot.py` | `fallback` |
| `scripts/README.md` | `fallback` |
| `scripts/README_TR.md` | `fallback` |
| `scripts/benchmarks_internal.py` | `fallback` |
| `scripts/bitnet_kernel_benchmark_standalone.py` | `fallback` |
| `scripts/build_chess_training_readiness_report.py` | `fallback` |
| `scripts/build_closure_governance_pack.py` | `TODO`, `scaffold`, `fallback` |
| `scripts/check_57_matrix.py` | `scaffold`, `fallback` |
| `scripts/check_doc_claim_consistency.py` | `fallback` |
| `scripts/chess_5080_onefile.py` | `fallback` |
| `scripts/data_pipeline.py` | `fallback` |
| `scripts/eval.py` | `fallback` |
| `scripts/generate_bench_reports.py` | `fallback` |
| `scripts/golden_eval.py` | `fallback` |
| `scripts/golden_score.py` | `fallback` |
| `scripts/hardening_bundle.py` | `fallback` |
| `scripts/kaggle_onecell_t4_build30.py` | `fallback` |
| `scripts/kaggle_onefile_closure_build30.py` | `fallback` |
| `scripts/kaggle_onefile_demo_build30.py` | `fallback` |
| `scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py` | `fallback` |
| `scripts/mertformer_5080_final_onefile.py` | `scaffold`, `fallback` |
| `scripts/mobile_export.py` | `fallback` |
| `scripts/precompute_logits_topk.py` | `fallback` |
| `scripts/record_dataset_hashes.py` | `fallback` |
| `scripts/run_liquid_ablation.py` | `fallback` |
| `scripts/smart_runner.py` | `fallback` |
| `scripts/sync_manifest.py` | `fallback` |
| `scripts/test_onnx_export.py` | `fallback` |
| `scripts/titan_preflight.py` | `fallback` |
| `scripts/update_investor_deck.py` | `fallback` |
| `scripts/update_system_hardware.py` | `fallback` |
| `scripts/verify_datasets.py` | `fallback` |
| `tests/test_architecture_integrity.py` | `fallback` |
| `tests/test_benchmark_tokenizer_id.py` | `fallback` |
| `tests/test_distillation_topk.py` | `fallback` |
| `tests/test_onnx_custom_op_contract.py` | `fallback` |
| `tests/test_sdk_api.py` | `fallback` |
| `tests/test_tokenizer_parity.py` | `fallback` |
