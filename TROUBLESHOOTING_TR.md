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

## 9) NCCL hang / deadlock (çoklu GPU)
**Belirti:** Çoklu-GPU eğitim ilerlemeden takılır; bir rank collective'de kilitlenir; loglarda `NCCL timeout`.
**Çözüm:** Tüm rank'lerin her collective'e ulaştığından emin olun (NaN/skip yolları aynı kararı broadcast etmeli — train loop bunları zaten DDP-sync eder). Yavaş I/O için `NCCL_TIMEOUT` artırın; Accelerate `concatenate()`'e tensör-olmayan metadata geçmeyin (str/int bilinçli düşürülür — bkz. `train/trainer_data.py`). Sürerse son checkpoint'ten yeniden başlatın.

## 10) Checkpoint bozulması / kesik checkpoint
**Belirti:** Resume `best.pt`/`latest.pt`/`final.pt` üzerinde unpickling/`EOFError`/boyut-uyumsuzluğu ile patlar.
**Çözüm:** Checkpoint'ler atomik yazılır (`*.pt.tmp` → `os.replace`), bu yüzden provider öldürse bile önceki checkpoint sağlam kalır — `latest.pt` (veya `best.pt`)'ten resume edin. Yeniden kullanmadan önce SHA256 sidecar ile bütünlüğü doğrulayın. Asla `*.pt.tmp` dosyasından resume etmeyin.

## 11) Tokenizer uyumsuzluğu (train/eval)
**Belirti:** Eval veya resume saçma üretir; vocab boyutu farklı; kanonik Llama-3 tokenizer (vocab 128256) yerine `gpt2`-tipi tokenizer (vocab 50257) yüklenir.
**Çözüm:** Kanonik yol `utils/tokenizer_resolver.py` kullanır (**sessiz fallback yok**) ve her checkpoint/shard'a `tokenizer_identity` hash'i basar. Uyumsuzluk bildirilirse `cfg.teacher_model_id` / çözülen tokenizer'ı doğrulayıp tekrar koşun; genel bir tokenizer ile override etmeyin.
