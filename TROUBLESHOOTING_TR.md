# Sorun Giderme — MertFormer Titan (Build 30 V2)

## 1) HF_TOKEN yok (online eğitim / teacher)
**Belirti:** Preflight veya eğitim gated teacher/dataset erişimi uyarısı verir.
**Çözüm:** `.env` veya ortam değişkenlerine `HF_TOKEN` ekleyip tekrar deneyin.

## 2) Stage JSONL eksik (offline mod)
**Belirti:** Preflight `Stage JSONL missing` uyarısı verir veya eğitim offline modda durur.
**Çözüm:** `python3 scripts/data_pipeline.py` çalıştırın veya online moda geçin (`TITAN_OFFLINE=0`).

## 3) accelerate config uyumsuzluğu
**Belirti:** Eğitim 1 GPU’da çalışır veya dağıtık ayarlar uygulanmaz.
**Çözüm:** `~/.cache/huggingface/accelerate/default_config.yaml` dosyasını silin/güncelleyin veya `TITAN_FORCE_ACCELERATE_RECONF=1` ile `run.sh` çalıştırın.

## 4) cuda.lock eksik
**Belirti:** Eğitim donanımında preflight `CUDA lock missing` uyarısı verir.
**Çözüm:** Hedef GPU makinesinde `python3 scripts/write_cuda_lock.py` çalıştırın.

## 5) Precomputed logits yolu uyumsuz
**Belirti:** Offline distillation logits shard bulamaz.
**Çözüm:** `TITAN_LOGITS_PATH` değerini `./datasets/logits/` (veya kendi yolunuz) olarak ayarlayın.

## 6) MPS üzerinde torch.compile
**Belirti:** macOS/MPS’te CfC fast path hata verir.
**Çözüm:** `liquid_fast_path=false` kullanın. (Varsayılan davranış zaten korur.)

## 7) OOM (bellek taşması)
**Belirti:** CUDA OOM veya süreç öldürülmesi.
**Çözüm:** Batch/seq düşürün, gradient checkpointing açın veya donanımı büyütün.

## 8) Token bütçesi beklenenden uzun sürüyor
**Belirti:** Eğitim planlanan adım sayısını aşıyor.
**Çözüm:** `TITAN_TOKEN_BUDGET_MODE=fixed_steps` veya `TITAN_MAX_STEPS` kullanın.
