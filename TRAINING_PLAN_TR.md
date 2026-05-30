# MertFormer Titan: Eğitim Runbook'u (Review-Ready)

Bu belge, eğitimi başlatmak için **runbook + kapılı kontrol listesi** formatında hazırlanmıştır. Dış mühendislik ekibi tarafından incelenebilir olacak şekilde yazılmıştır.

Prensipler:
- Varsayılan çalışma şekli **offline-first** (`TITAN_OFFLINE=1`). Online adımlar açıkça etkinleştirilir.
- **Verified (Run)** kontrolleri ile **Target/Claim** iddiaları ayrı tutulur.
- Dataset provenans ve lisans doğrulaması, herhangi bir üretim eğitim koşusu öncesinde **P0 gate** kabul edilir.

## Baseline

- Python **3.11** (bkz. `repro/python_TR.md`)
- Yerel venv: `.titan-venv` (`scripts/bootstrap_venv.sh` ile oluşturulur)

## Kapanış Kararları

- Önerilen harici başlangıç hattı `remote_bootstrap`.
- Katı yerel hat `offline_clean` olarak kalır.
- `offline_clean` için kanonik iç davranış artık **katı precomputed KD** yoludur.
- Sabit öğretmen yüzeyi: `meta-llama/Llama-3.3-70B-Instruct`.
- `teacherless`, 45K `offline_clean` koşusu için artık kanonik fallback değildir.
- Logits shard'ları tamam değilse claim-safe toparlanma yolu, geçerli `HF_TOKEN` ile Phase-0 precompute çalıştırmaktır; aksi halde hat bloklu kalır.
- Eğitim sonrası çıktı toplama iki paket stratejisi kullanır:
  - release/repo zipleri kapanış artefaktı olarak kalır
  - gerçek koşu çıktıları ayrıca `artifacts/mertformer_training_outputs_bundle.zip` içinde toplanır

## Aşama 0: Offline Doğrulama (Geçmek Zorunda)

Offline verify-all hattını çalıştırın:

```bash
bash scripts/verify_all.sh
```

Kapsadığı kontroller:
- Track edilen dosyalarda secret taraması (`scripts/secret_scan.py`)
- Unit/integration testleri (`pytest`)
- Preflight (`scripts/titan_preflight.py`, offline-safe)
- Operator gate (safe mod, offline-safe)

Opsiyonel: küçük eğitim smoke testi (CPU/MPS):

```bash
.titan-venv/bin/python scripts/train_smoke.py --cleanup
```

## Aşama 0.5: Veri Uyum Kapısı (P0)

Eğitimden önce:
- `datasets/inventory.md` ve `datasets/inventory_TR.md`, kod referanslarıyla uyumlu olmalı (`scripts/extract_dataset_refs.py` ile üretilir).
- `datasets/SOURCES*.md`, pipeline ve eval'in referans ettiği tüm datasetleri listelemeli.
- `datasets/LICENSES*.md`, **çekirdek eğitim datasetleri** için **unknown/TBD lisans** bırakmamalı; opsiyonel/demo datasetler doğrulanana kadar kapalı kalmalı.
- `datasets/hashes.json`, kullanılacak **tam dataset snapshot'ları** için snapshot metadata ve SHA256 hash bilgileriyle doldurulmalı (air-gapped uyumlu).

Herhangi bir **çekirdek eğitim** lisansı `TBD` ise veya snapshot hash'i eksikse eğitim **review-ready değildir**. Opsiyonel/demo datasetler doğrulanana kadar devre dışı kalmalıdır.

Validation veri politikası:
- Smoke testleri için küçük validation dosyası kabul edilebilir.
- Claim-grade koşular için temsil gücü olan validation setini yeniden üretin:

```bash
python3 scripts/build_validation_set.py --target-size 1500
python3 scripts/record_dataset_hashes.py
```

- `TITAN_CLAIM_MODE=1`, minimum validation boyutunu (`cfg.validation_min_samples_claim`) zorlar.

## Aşama 1: Damıtma / Foundation Koşusu (Online, Eğitim Donanımı)

Gerekli:
- Eğitim donanımı (örn. multi-GPU Linux host) ve stabil CUDA toolchain
- Önerilen `remote_bootstrap` hattı için hedef makinede enjekte edilmiş `HF_TOKEN`, veya katı `offline_clean` hattı hâlâ Phase-0 logits precompute gerektiriyorsa yerelde mevcut `HF_TOKEN`
- Opsiyonel: experiment tracking açıksa `WANDB_API_KEY`
- Mevcut repo tarafı hat zaten yeşil:
  - `TRAIN_ALLOWED`
  - `READY_REMOTE_BOOTSTRAP`
- Kalan kazanan olmayan blocker'lar:
  - `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`
  - `online_teacher:MISSING_HF_TOKEN`

Kanonik readiness kapısı:

```bash
bash zero_touch_start.sh --check-only
```

Bu 45K koşusu, nihai yetenek tavanı değil; ilk ciddi mimari doğrulama koşusudur.

Önerilen çalıştırma (hedef donanımda remote-bootstrap hattı):

```bash
HF_TOKEN=... TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh
```

Opsiyonel hız-smoke kontrolleri (yalnız hedef makine; iddia değildir):

```bash
python3 -m pytest -q tests/test_packed_projection_equivalence.py tests/test_liquid_safeguard.py

HF_TOKEN=... \
TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable \
TITAN_BATCH_SIZE=1024 TITAN_BATCH_SIZE_FALLBACKS=1024,512,256 \
TITAN_DATALOADER_PIN=1 TITAN_DATALOADER_NONBLOCKING=1 \
TITAN_FFN_PACK=1 TITAN_MOE_PACK=1 TITAN_MLA_KV_PACK=1 \
TITAN_LIQUID_FAST_PATH=0 TITAN_LIQUID_TRAIN_IMPL=packed_pair \
MERTFORMER_LOWBIT_KERNEL=0 MERTFORMER_FUSED_BACKWARD=0 \
bash zero_touch_start.sh --dry-run
```

Hız kontrol sınırları:
- Packed projection flag'leri varsayılan kapalıdır ve kullanılmadan önce equivalence testlerinden geçmelidir.
- `TITAN_BATCH_SIZE` artık `config/config.py` içine bağlıdır; Ocean 2x H200 profili `1024` ile başlar ve yalnızca net OOM'da `1024 -> 512 -> 256` sırasıyla düşer.
- İlk Ocean uzun koşusunda `TITAN_LIQUID_FAST_PATH=0` tutulur ve `packed_pair_compile` kullanılmaz.
- `repro/accelerate_8xgpu.yaml`, yeniden üretilebilirlik/koşu config'idir ve yalnızca `ACCELERATE_CONFIG_FILE` ile seçilirse çalıştırmayı etkiler.
- 45K iddia sınırı değişmez: gerçek koşu ve ölçüm artefaktları oluşmadan `trained`, `benchmark-verified`, `mobile-ready`, `production-ready`, hız, enerji veya deployment iddiası geçerli değildir.

Kanonik offline-clean doğruluk sınırı:
- precomputed logits zaten tamamsa onları kullan
- değilse `HF_TOKEN` mevcutken Phase-0 ile üretmelerine izin ver
- aksi halde uzun koşuyu başlatma

Opsiyonel online teacher çalıştırma:

```bash
HF_TOKEN=... TITAN_OFFLINE=0 TITAN_WANDB=1 TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh
```

Notlar:
- `TITAN_INSTALL=1` opt-in'dir. Deterministik kurulum için öncesinde `scripts/bootstrap_venv.sh` tercih edilir.
- Loglar `logs/` altında yazılır ve **gitignored artefakt** kabul edilir. Commit değil, kanıt eki olarak ele alınmalıdır.
- `zero_touch_start.sh`, kanonik 45K başlatıcıdır. `run.sh`; `--test`, `--sitl-demo` ve `--cleanroom-verify` gibi legacy yardımcı akışlar için kalır.
- Handoff öncesi gerçek orchestrator sözleşmesini görmek için `bash zero_touch_start.sh --plan-only` veya `--dry-run` kullanılabilir.
- Başarılı post-train kapanış, hem release-side paketleri hem de hedef makineden indirilecek `artifacts/mertformer_training_outputs_bundle.zip` dosyasını yeniler.

## Aşama 2: Ajan Entegrasyonu (Opsiyonel / Foundation Sonrası)

Hedef işler:
- Her rol için rol-özel adapter'lar (LoRA/PEFT): QA, Security, Architect
- Dahili retrieval/memory datasetleri aynı provenans/lisans kurallarına uymalıdır

## Aşama 3: Optimizasyon (Opsiyonel)

Örnekler:
- Stabilite sertleştirme (OOM, NaN drill, checkpoint restore drill)
- Hedef edge runtime üzerinde profiling + export doğrulaması

## Aşama 4: Değerlendirme & Benchmarklar (Eğitim Sonrası)

Dahili doğruluk benchmarkları:
- HumanEval / MBPP runner: `scripts/benchmarks_internal.py --run`
- Çıktılar: `reports/benchmarks/*.jsonl`

Davranış:
- Checkpoint yoksa benchmark runner **SKIP** yapar ve exit 0 ile çıkar; pipeline'ı kırmaz.

## “Eğitimi Başlat” Kontrol Listesi (Yazdırılabilir)

- [ ] `bash scripts/verify_all.sh` geçer (offline).
- [ ] `datasets/LICENSES*.md`, **çekirdek eğitim datasetleri** için **TBD** girdisi içermez; opsiyonel/demo datasetler devre dışı kalır.
- [ ] `datasets/hashes.json`, tam eğitim verisi için snapshot hash'leriyle doldurulmuştur.
- [ ] Claim-grade modda `datasets/validation.jsonl` temsil gücüne sahiptir ve minimum örnek kapısını geçer.
- [ ] Eğitim config'i sabitlenmiş ve incelenmiştir (seed, dtype, model boyutu, batch ayarları).
- [ ] Donanım + sürücüler doğrulanmış ve notlanmıştır.
- [ ] Online tokenlar env içinde mevcut ve değerleri loglanmadan doğrulanmıştır.
- [ ] Opsiyonel packed/Liquid hız flag'leri açılacaksa equivalence testleri geçmiştir.
