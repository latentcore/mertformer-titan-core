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
- Repo disi one-file kopyalari desteklenmez ve drift kaynagi sayilir.

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
- `check_tokenizer_sync.py` — Kanonik tokenizer spec senkronunu zorunlu kilar (`interfaces/tokenizer_spec.json` -> `tokenizer/tokenizer.json`).
- `check_translation_pointer_policy.py` — Derin denetim TR counterpart dosyalarinda pointer politikasini zorunlu kilar.
- `check_doc_claim_consistency.py` — Ana dokumanlarda claim/evidence tutarliligini kontrol eder.
- `clean_runtime_artifacts.sh` — Runtime artefaktlarini temizler (kök `kaggle_onefile_build30_*.jsonl` dahil).
- `zip_denylist_audit.py` — Release zip'i denylist yol/secret pattern kontrolünden geçirir.

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

## Varlıklar
- `build_investor_deck.py` — PPTX yatırımcı deck üretimi.
- `auto_demo_video.py` — Opsiyonel demo video otomasyonu (ffmpeg gerekli).

## Klasörler
- `scripts/reports/` — Script ürettiği rapor çıktıları.
- `scripts/runs/` — Koşu çıktı dizinleri.

---

Not: `run.sh` ana otomasyon yolunu kapsar (install + strict preflight + eğitim). Review için `scripts/verify_all.sh` önerilir.
