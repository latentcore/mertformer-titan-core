# Script Kataloğu

Tüm scriptler repo kökünden `python3 scripts/<ad>.py` şeklinde çalıştırılır.
Bazıları GPU, HF_TOKEN veya WANDB_API_KEY ister. Emin değilsen `run.sh` ile başla.

## Çekirdek Akışlar
- `smart_runner.py` — Ana orkestratör: veri → distill → eğitim.
- `data_pipeline.py` — Veri hazırlama (5 aşamalı müfredat).
- `titan_preflight.py` — Uçtan uca preflight doğrulama.
- `operator_mode_gate.py` — Tek girişli ops gate (güvenlik + sanity checks).
- `overfit_gate.py` — 1MB overfit gate (safe/full mod).

## Değerlendirme & Benchmark
- `golden_eval.py` — Golden sample evaluator (50 prompt).
- `benchmarks_internal.py` — HumanEval / MBPP çıktı üretimi.
- `eval.py` — GSM8K eval wrapper (legacy; bkz: `eval/gsm8k.py`).

## Export & ONNX
- `mobile_export.py` — Mobil/edge ONNX export.
- `test_onnx_export.py` — ONNX export test.
- `verify_onnx_local.py` — Yerel ONNX doğrulama.
- `titan_onnx_stress_test.py` — ONNX stres testi.

## Ops Drill’leri
- `nan_kill_test.py` — Yapay NaN kill‑switch drill.
- `failure_budget_drill.py` — Failure budget drill.
- `checkpoint_restore_drill.py` — Checkpoint restore drill.

## Artefact & Raporlar
- `mini_titan_poc.py` — Adli PoC logger (hash zincirli loglar).
- `scaling_audit_math.py` — Scaling audit matematiği.
- `update_system_hardware.py` — `reports/system_hardware*.md` günceller.
- `write_cuda_lock.py` — Mevcut sistemden `repro/cuda.lock` üretir.
- `verify_datasets.py` — Dataset sanity kontrolleri.

## Yardımcılar
- `chat.py` — Etkileşimli chat arayüzü.
- `xray.py` — Proje denetleyici (yapı dump).
- `mac_simulation.py` — Mac simulasyon koşusu (CPU/MPS).
- `train_tpu_turbo.py` — TPU eğitim başlatıcı (experimental).
- `download_tr_tokenizer.py` — Türkçe tokenizer indirme (opt‑in).
- `logbook_build.py` — Birleşik logbook üreticisi (`logs/ALL_LOGS.jsonl`).
- `version_checker.py` — Sürüm tutarlılık kontrolü (deprecated marker bulursa hata verir).

## Varlıklar
- `build_investor_deck.py` — PPTX yatırımcı deck üretimi.
- `auto_demo_video.py` — Opsiyonel demo video otomasyonu (ffmpeg gerekli).

## Klasörler
- `scripts/reports/` — Script ürettiği rapor çıktıları.
- `scripts/runs/` — Koşu çıktı dizinleri.

---

Not: `run.sh` ana otomasyon yolunu kapsar (env + preflight + eğitim).
