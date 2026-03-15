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
- Repo dışı one-file kopyaları desteklenmez ve drift kaynağı sayılır.

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
- `clean_runtime_artifacts.sh` — Runtime artefaktlarını temizler (kök `kaggle_onefile_build30_*.jsonl` dahil).
- `run_and_clean_pycache.py` — Herhangi bir komutu çalıştırır ve koşu sonunda cache temizliğini garanti eder (`--full-clean` ile `.DS_Store`, `.cache`, `.ipynb_checkpoints`, `.tox`, `.nox`, `.hypothesis`, `.vs`; venv cache icin `--include-venv-caches`).
- `zip_denylist_audit.py` — Release zip'i denylist yol/secret pattern kontrolünden geçirir.

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
