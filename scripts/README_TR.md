# Script Kataloğu

Tüm scriptler repo kökünden çalıştırılmak üzere tasarlanmıştır.

Konvansiyonlar:
- Tercihen `.titan-venv/bin/python scripts/<ad>.py` kullanın (bkz: `scripts/bootstrap_venv.sh`).
- Script doğrulama akışları offline-first kalır (`TITAN_OFFLINE=1`).
- `run.sh` eğitim sözleşmesi varsayılan online çalışır ve readiness-only modu içerir (`bash run.sh --train-ready`).
- Profil sözleşmesi: `TITAN_PROFILE=stable` (varsayılan) veya tüm ileri overlay için `TITAN_PROFILE=max_arch`.

Emin değilseniz önce tek komut doğrulama çalıştırın: `bash scripts/verify_all.sh`.

## Kanonik One-File Path
- Resmi/kanonik one-file script: `scripts/kaggle_onefile_demo_build30.py`
- Colab fastproof companion one-file script: `scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py`
- Text-understanding PoC one-file script: `scripts/kaggle_onefile_demo_build30_text_understanding.py`
- Windows RTX 5080 satranç PoC one-file script: `scripts/chess_5080_onefile.py`
- Satranç one-file desteklenen modlar: `train`, `verify`, `benchmark`, `package`, `resume`, `arena`
- Satranç one-file artık isimli feature bundle desteği taşır: `--feature-bundle` ve bunun üstüne `--enable-features` / `--disable-features` override’ları kullanılabilir.
- Önerilen ileri bundle adları: `routing_stack`, `liquid_stack`, `memory_attention_stack`, `cognitive_stack`, `objective_stack`, `postrun_analysis_stack`, `all_stable_extensions`, `all_on_experimental`.
- Kanonik 24 saatlik RTX 4060 profili: `strength_4060_24h` (`baseline_supported`, release-candidate uygun).
- Desteklenen taşınabilir baseline profil: `production_5080` (`supported_portable_baseline`).
- Research-only 24 saatlik RTX 4060 profilleri: `strength_4060_24h_all_on_experimental` (`experimental`) ve `strength_4060_24h_omni_max` (`experimental_high_risk`).
- `--mode arena` insan-vs-model terminal yüzeyi açar; anlamlı oyun için `--resume-from <checkpoint>` kullanın.
- Satranç one-file artık kanonik Build30 trunk ailelerini tek dosyada mirror eder: BitLinear, MLA, CfC Liquid, MoE/LiquidRouter, QINN, cognitive extension katmanları ve world-model hookları.
- Yardımcı satranç head’leri artık feature flag ile açılıp kapanabilir: `phase_head`, `wdl_head`, `legality_head`.
- Koşu-sonrası satranç analiz yüzeyleri artık feature flag ile açılıp kapanabilir: `selfplay_eval_enabled`, `tournament_eval_enabled`, `replay_buffer_enabled`.
- Mirror anti-drift kanıtı satranç koşularında `reports/mirror_parity_report.json` olarak yazılır.
- Feature-bundle kanıtı satranç koşularında `reports/feature_flag_report.json` ve `reports/feature_flag_report.md` olarak yazılır.
- Closure manifest yüzeyi artık `reports/run_status_manifest.json`, `reports/postrun_analysis_manifest.json` ve `reports/artifact_truth_matrix.json` dosyalarını üretir.
- Release/evidence registry yüzeyi artık `reports/run_contract.json`, `reports/release_snapshot.json`, `reports/evidence_pack_stub.json` ve `reports/final_truth_registry.json` dosyalarını üretir.
- Ek release-truth artefaktları artık `reports/claim_registry.json`, `reports/known_limits.json`, `reports/support_matrix.json` ve `reports/release_gate_summary.json` dosyalarını üretir.
- Handoff/release stub artefaktları artık `reports/rc_stub.json`, `reports/golden_stub.json`, `reports/handoff_pack_manifest.json` ve `reports/operator_handoff_summary.json` dosyalarını üretir.
- External closure stub artefaktları artık `reports/external_repro_stub.json`, `reports/pilot_stub.json`, `reports/security_stub.json` ve `reports/legal_stub.json` dosyalarını üretir.
- Operator/DR closure stub artefaktları artık `reports/operator_handbook_stub.json`, `reports/dr_evidence_stub.json`, `reports/backup_retention_stub.json` ve `reports/blind_handoff_stub.json` dosyalarını üretir.
- Release-governance artefaktları artık `reports/release_notes_stub.json`, `reports/freeze_manifest_stub.json`, `reports/changelog_snapshot.json` ve `reports/maintenance_policy_stub.json` dosyalarını üretir.
- Device/export/packaging closure artefaktları artık `reports/export_truth_stub.json`, `reports/device_validation_stub.json`, `reports/packaging_closure_stub.json` ve `reports/installer_validation_stub.json` dosyalarını üretir.
- Benchmark-closure artefaktları artık `reports/benchmark_raw_outputs_stub.json`, `reports/benchmark_compare_report_stub.json`, `reports/benchmark_summary_stub.json` ve `reports/benchmark_manifest_stub.json` dosyalarını üretir.
- Training/accounting closure artefaktları artık `reports/training_report_stub.json`, `reports/token_accounting_stub.json`, `reports/compute_accounting_stub.json` ve `reports/cost_report_stub.json` dosyalarını üretir.
- Trained-artifact-truth artefaktları artık `reports/final_weights_truth_stub.json`, `reports/best_checkpoint_truth_stub.json`, `reports/latest_checkpoint_truth_stub.json` ve `reports/trained_artifact_registry_stub.json` dosyalarını üretir.
- Management-closure artefaktları artık `reports/core_complete_decision_stub.json`, `reports/research_continues_stub.json`, `reports/product_maintenance_only_stub.json` ve `reports/closure_decision_record_stub.json` dosyalarını üretir.
- Repo-truth summary artefaktları artık `reports/master_closure_table.json`, `reports/remaining_core_blockers.json`, `reports/repo_side_completion_summary.json` ve `reports/readiness_snapshot.json` dosyalarını üretir.
- Aggregated truth artefaktları artık `reports/aggregated_master_table.json`, `reports/real_remaining_core_work.json`, `reports/repo_truth_inventory.json` ve `reports/closure_gap_summary.json` dosyalarını üretir.
- Project-truth ve docs-alignment artefaktları artık `reports/project_master_truth_reference.json`, `reports/project_remaining_real_blockers.json`, `reports/truth_docs_index.json` ve `reports/truth_docs_drift_report.json` dosyalarını üretir.
- Consistency/action artefaktları artık `reports/project_blocker_action_plan.json`, `reports/project_blocker_dependency_graph.json`, `reports/project_execution_sequence.json`, `reports/project_lane_status_board.json`, `reports/project_closure_phase_plan.json`, `reports/project_phase_readiness_scoreboard.json`, `reports/project_owner_accountability_matrix.json`, `reports/project_owner_work_queue.json`, `reports/project_critical_path_report.json`, `reports/project_owner_next_actions_summary.json`, `reports/project_ready_now_board.json`, `reports/project_unlock_impact_report.json`, `reports/project_parallel_workset_report.json`, `reports/project_phase_exit_criteria_report.json`, `reports/project_execution_wave_report.json`, `reports/project_evidence_backlog_report.json`, `reports/project_dependency_bottleneck_report.json`, `reports/project_owner_phase_frontier_report.json`, `reports/project_evidence_criticality_report.json`, `reports/project_phase_transition_matrix.json`, `reports/project_owner_load_report.json`, `reports/project_phase_dependency_pressure_report.json`, `reports/project_owner_bottleneck_alignment_report.json`, `reports/project_evidence_phase_heatmap_report.json`, `reports/project_blocker_risk_register_report.json`, `reports/project_release_prereq_matrix_report.json`, `reports/project_foundation_run_dependency_report.json`, `reports/project_release_path_report.json`, `reports/project_external_closure_cluster_report.json`, `reports/project_owner_evidence_gap_report.json`, `reports/project_release_gate_dependency_report.json`, `reports/project_external_signoff_queue_report.json`, `reports/project_release_evidence_bridge_report.json`, `reports/project_training_run_readiness_report.json`, `reports/project_benchmark_closure_dependency_report.json`, `reports/project_release_decision_queue_report.json`, `reports/project_external_validation_readiness_report.json`, `reports/project_artifact_lock_readiness_report.json`, `reports/project_final_release_cutover_report.json`, `reports/project_real_run_execution_queue_report.json`, `reports/project_benchmark_evidence_lock_report.json`, `reports/project_final_signoff_cutset_report.json`, `reports/generated_truth_consistency_report.json` ve `reports/generated_truth_crosscheck_matrix.json` dosyalarını üretir.
- Donmuş 24 saatlik 4060 yolu için kanonik runbook/checklist artık `runbooks/chess_4060_24h.md` ve `checklists/chess_4060_24h.md` altındadır.
- Deneysel 24 saatlik 4060 runbook/checklist dokümanları `runbooks/chess_4060_24h_all_on_experimental.md` ve `checklists/chess_4060_24h_all_on_experimental.md` altında kalır.
- Kanonik repo-side contract yüzeyleri artık `configs/`, `releases/`, `knowledge/` ve `evidence/` klasörlerini de içerir.
- Satranç runtime observability sözleşmesi artık nettir: `logs/run_log.jsonl`, `reports/logging_contract.json`, `reports/observability_report.json`.
- Fatal runtime hatalarının hem `logs/run_log.jsonl` içindeki `fatal_exception` event'inde hem de Desktop tarafındaki `*_FAILED_*.json` artefaktında görünmesi beklenir.
- Windows builder/export akışı artık `MERTFORMER_CHESS_ARCHIVE_PASSWORD` değerini derlenmiş launcher içine gömmez; şifreli çıktı gerekiyorsa final EXE çalıştırılmadan önce hedef makinede ortam değişkeni olarak verin.
- Windows RTX 5080 share/export builder: `scripts/export_chess_5080_share.py`
- Repo dışı one-file kopyaları desteklenmez ve drift kaynağı sayılır.

### Satranç Onefile Bundle Örnekleri
```bash
# Kanonik 24 saatlik RTX 4060 eğitim-başlat komutu
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h

# Taşınabilir baseline üstüne nispeten stabil ileri stack
python3 scripts/chess_5080_onefile.py --mode train --profile production_5080 --feature-bundle all_stable_extensions

# RTX 4060 için 24 saatlik, tüm büyük onefile extension’ları açık research-only profil
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h_all_on_experimental

# RTX 4060 için daha agresif omni-max research varyantı
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h_omni_max

# All-on profilden başlayıp belirli riskli yüzeyleri kapatma örneği
python3 scripts/chess_5080_onefile.py --mode train --profile strength_4060_24h_all_on_experimental --disable-features use_qinn,use_world_model_head

# Gövde profili koruyup sadece koşu-sonrası self-play/tournament/replay artefaktlarını açma örneği
python3 scripts/chess_5080_onefile.py --mode train --profile production_5080 --feature-bundle postrun_analysis_stack
```

## Çekirdek Akışlar
- `smart_runner.py` — Ana orkestratör: veri → distill → eğitim.
- `data_pipeline.py` — Veri hazırlama (5 aşamalı müfredat).
- `titan_preflight.py` — Uçtan uca preflight doğrulama.
- `operator_mode_gate.py` — Tek girişli ops gate (güvenlik + sanity checks).
- `overfit_gate.py` — 1MB overfit gate (safe/full mod).
- `train_smoke.py` — Küçük offline training sanity loop (CPU/MPS).

## Review-Ready Araçlar
- `bootstrap_venv.sh` — `.titan-venv` üretir (Python 3.11 baseline). Demo için `--demo` ile `pygame` kurar.
- `verify_all.sh` — Offline-first verify-all: secret scan → pytest → preflight → operator gate (safe).
- `secret_scan.py` — Track'li dosyalarda olası secret pattern taraması (CI gate).
- `check_tokenizer_sync.py` — Kanonik tokenizer spec senkronunu zorunlu kılar (`interfaces/tokenizer_spec.json` -> `tokenizer/tokenizer.json`).
- `check_translation_pointer_policy.py` — Derin denetim TR counterpart dosyalarında pointer politikasını zorunlu kılar.
- `check_doc_claim_consistency.py` — Ana dokümanlarda claim/evidence tutarliligini kontrol eder.
- `build_code_truth_audit.py` — Maturity etiketleri, dört sütunlu kanıt zorunluluğu ve marker taraması ile code-truth delta audit üretir.
- `build_workspace_hygiene_manifest.py` — Quarantine-first workspace hygiene manifest üretir; `--apply-quarantine` yalnız üretilen manifest insan gözüyle incelendikten sonra kullanılmalıdır.
- `clean_runtime_artifacts.sh` — Runtime artefaktlarını temizler (kök `kaggle_onefile_build30_*.jsonl` dahil).
- `run_and_clean_pycache.py` — Herhangi bir komutu çalıştırır ve koşu sonunda cache temizliğini garanti eder (`--full-clean` ile `.DS_Store`, `.cache`, `.ipynb_checkpoints`, `.tox`, `.nox`, `.hypothesis`, `.vs`; venv cache icin `--include-venv-caches`).
- `zip_denylist_audit.py` — Release zip'i denylist yol/secret pattern kontrolünden geçirir.
- `build_scoped_external_intake_matrix.py` — Scoped Desktop/Documents/Downloads/Applications proje artefaktlarını hash'leyip closure intake matrix'ine sınıflandırır.
- `cleanup_scoped_closure_junk.py` — Repo + scoped external dizinlerde closure artığı çöpü (`__pycache__`, `.pyc`, stale duplicate zip) temizler.

## SOP Çıktıları
- `reports/one_command_full_sop_summary.md` — Full tek-komut SOP koşusunun tek belgede konsolide özeti.
- `reports/one_command_full_sop.log` — Aynı koşunun ham tam logu.
- Her full SOP koşusunda iki artefakt da güncellenir/üzerine yazılır.

## Değerlendirme & Benchmark
- `golden_eval.py` — Golden sample evaluator (50 prompt).
- `golden_score.py` — Assertion tabanlı golden skorlayıcı (`reports/benchmarks/golden_summary.json`).
- `benchmarks_internal.py` — HumanEval / MBPP çıktı üretimi (checkpoint/dataset yoksa SKIP).
- `bitnet_kernel_benchmark_standalone.py` — Tek dosyalı standalone BitNet ternary kernel benchmark'ı (kernel + quantization + benchmark akışı tek dosyada).
- `eval.py` — GSM8K eval wrapper (legacy; bkz: `eval/gsm8k.py`).

## Export & ONNX
- `mobile_export.py` — Mobil/edge ONNX export.
- `test_onnx_export.py` — ONNX export test.
- `verify_onnx_local.py` — Yerel ONNX doğrulama.
- `titan_onnx_stress_test.py` — ONNX stres testi.

## Dataset Uyum / Provenans
- `extract_dataset_refs.py` — Kodun referansladığı dataset ID'lerini `datasets/inventory*` dosyalarına çıkarır (default offline).
- `verify_datasets.py` — Online dataset erişim sanity kontrolleri (`--login` ile HF login opt-in).

## Ops Drill’leri
- `nan_kill_test.py` — Yapay NaN kill‑switch drill.
- `failure_budget_drill.py` — Failure budget drill.
- `checkpoint_restore_drill.py` — Checkpoint restore drill.

## Artefact & Raporlar
- `mini_titan_poc.py` — Adli PoC logger (hash zincirli loglar).
- `scaling_audit_math.py` — Scaling audit matematiği.
- `update_system_hardware.py` — `reports/system_hardware*.md` günceller (`run.sh --test` modunda atlanır).
- `write_cuda_lock.py` — Mevcut sistemden `repro/cuda.lock` üretir.

## Yardımcılar
- `chat.py` — Etkileşimli chat arayüzü.
- `mathfp_interactive_chat.py` — Math-fastproof interaktif soru/cevap (çıkış için `q`).
- `xray.py` — Proje denetleyici (yapı dump).
- `mac_simulation.py` — Mac simülasyon koşusu (CPU/MPS).
- `train_tpu_turbo.py` — TPU eğitim başlatıcı (experimental).
- `download_tr_tokenizer.py` — Türkçe tokenizer indirme (opt‑in).
- `logbook_build.py` — Birleşik logbook üreticisi (çıktı `logs/` altında gitignored artifact).
- `version_checker.py` — Sürüm tutarlılık kontrolü (deprecated marker bulursa hata verir).
- `resume_compat_check.py` — Resume uyumunu doğrular ve `reports/resume_compat_report.json` yazar.
- `tools/claim_number_audit.py` — `*.md` içindeki parametre formatlarını denetler (`reports/claim_number_audit.json`).
- `tools/denylist_scan_zip.py` — Release zip denylist ön kontrolü (`reports/artifacts_zip_denylist_audit.json`).

## Varlıklar
- `build_investor_deck.py` — PPTX yatırımcı deck üretimi.
- `update_investor_deck.py` — Yatırımcı deck PPTX dosyasını Build 30 V2 ile günceller (otomatik V2 slaytı + metin dönüşümü).

## Klasörler
- `scripts/reports/` — Script ürettiği rapor çıktıları.
- `scripts/runs/` — Koşu çıktı dizinleri.

---

Not: `run.sh` ana otomasyon yolunu kapsar (install + strict preflight + eğitim). Review için `scripts/verify_all.sh` önerilir.

## Build30 Colab Math Fastproof V2 (V1 Kapanış)

`scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py` dosyası kapanış seviyesi PoC paketlemesi için korumalı full-spectrum hooklar ile güncellendi.

- Katı config schema v2 (`validate_run_config_schema`) ile unknown-key reddi/fail-fast.
- Runtime fingerprint ve ownership paketi (`runtime_fingerprint`, `ownership_proof`, `env_snapshot_redacted`, `reproduce_command`).
- Compile/CUDAGraph stall guard (`compile_policy=off` varsayılan, timeout fallback, guard telemetry).
- Zero-shot unseen matematik split (`eval_unseen_*`) ve compare payload v2 (`exact_match_unseen`).
- Interpretability çıktıları (`gradient_flow_heatmap.png`, `moe_expert_bar_proxy.png`) feature flag arkasında.
- Payload içinde completeness yüzdesi ile `feature_coverage_matrix` sözleşmesi.
