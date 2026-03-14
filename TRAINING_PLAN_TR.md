# MertFormer Titan: Eğitim Runbook (Review-Ready)

Bu belge, eğitimi başlatmak için **runbook + gate/checklist** formatında hazırlanmıştır. Dış mühendislik incelemesine (engineering review) uygun olacak şekilde yazılmıştır.

Prensipler:
- Varsayılan çalışma şekli **offline-first** (`TITAN_OFFLINE=1`). Online adımlar açıkça etkinleştirilir.
- **Verified (Run)** doğrulamalar ile **Target/Claim** iddialar ayrılır.
- Dataset provenans/lisans konuları, üretim eğitimi için **P0 gate** kabul edilir.

## Baseline
- Python **3.11** (bkz: `repro/python_TR.md`)
- Yerel venv: `.titan-venv` ( `scripts/bootstrap_venv.sh` ile)

## Aşama 0: Offline Doğrulama (Geçmek Zorunda)

Offline verify-all:

```bash
bash scripts/verify_all.sh
```

Kapsadığı kontroller:
- Track'li dosyalarda secret taraması (`scripts/secret_scan.py`)
- Unit/integration testleri (`pytest`)
- Preflight (`scripts/titan_preflight.py`, offline-safe)
- Operator gate (safe mod, offline-safe)

Opsiyonel: küçük eğitim smoke testi (CPU/MPS):

```bash
.titan-venv/bin/python scripts/train_smoke.py --cleanup
```

## Aşama 0.5: Veri Uyum Gate'i (P0)

Eğitimden önce:
- `datasets/inventory.md` ve `datasets/inventory_TR.md` kod referanslarıyla uyumlu ( `scripts/extract_dataset_refs.py` ile üretilir )
- `datasets/SOURCES*.md` pipeline ve eval'in referans ettiği tüm datasetleri listeler
- `datasets/LICENSES*.md` çekirdek eğitim datasetleri için **TBD/Unknown lisans bırakmaz**; opsiyonel/demo datasetler doğrulanana kadar devre dışı tutulmalıdır
- `datasets/hashes.json` gerçek eğitimde kullanılacak snapshotlar için revision + tarih + SHA256 hash ile doldurulmuş olmalıdır (air-gapped uyumlu)

Eğer **çekirdek eğitim** lisanslarında `TBD` varsa veya snapshot hash'leri eksikse, eğitim **review-ready değildir**. Opsiyonel/demo datasetler doğrulanana kadar devre dışı tutulmalıdır.

Validation veri politikası:
- Smoke test için küçük validation dosyası kabul edilebilir.
- İddia seviyesi koşular için temsil gücü olan validation setini yeniden üretin:

```bash
python3 scripts/build_validation_set.py --target-size 1500
python3 scripts/record_dataset_hashes.py
```

- `TITAN_CLAIM_MODE=1` aktifken minimum validation boyutu (`cfg.validation_min_samples_claim`) zorlanır.

## Aşama 1: Damıtma / Foundation (Online, Eğitim Donanımı)

Gerekli:
- Eğitim donanımı (örn: multi-GPU Linux host) ve stabil CUDA toolchain
- Online mod için `HF_TOKEN`
- Opsiyonel: `WANDB_API_KEY` (tracking istenirse)

Önerilen çalışma (explicit online):

```bash
TITAN_OFFLINE=0 TITAN_WANDB=1 TITAN_INSTALL=1 bash run.sh
```

Notlar:
- `TITAN_INSTALL=1` opt-in. Deterministik kurulum için önce `scripts/bootstrap_venv.sh` önerilir.
- Loglar `logs/` altında üretilir ve **gitignored artifact** olarak kalır. Commit edilmez; “kanıt eki” olarak paketlenir.

## Aşama 2: Ajan Entegrasyonu (Opsiyonel / Foundation Sonrası)

Hedef işler:
- Her rol için adapter (LoRA/PEFT) (QA, Güvenlik, Mimar)
- Retrieval/memory datasetleri de aynı provenans/lisans kurallarına tabidir

## Aşama 3: Optimizasyon (Opsiyonel)

Örnekler:
- Stabilite hardening (OOM, NaN drill, checkpoint restore drill)
- Hedef edge runtime üzerinde profiling + export doğrulaması

## Aşama 4: Değerlendirme & Benchmarklar (Eğitim Sonrası)

Dahili benchmark:
- HumanEval / MBPP runner: `scripts/benchmarks_internal.py --run`
- Çıktılar: `reports/benchmarks/*.jsonl`

Davranış:
- Checkpoint yoksa runner **SKIP** yapar ve exit 0 döner (pipeline'ı kırmaz).

## “Eğitimi Başlat” Checklist (Yazdırılabilir)

- [ ] `bash scripts/verify_all.sh` PASS (offline).
- [ ] `datasets/LICENSES*.md` cekirdek egitim datasetleri icin **TBD icermiyor** (opsiyonel/demo datasetler devre disi tutulmalidir).
- [ ] `datasets/hashes.json` eğitimde kullanılacak snapshot hash’leriyle dolduruldu.
- [ ] İddia seviyesi modda `datasets/validation.jsonl` temsil gücüne sahip ve minimum örnek kapısını geçiyor.
- [ ] Training config review edildi (seed, dtype, model boyutu, batch ayarları).
- [ ] Donanım + driver doğrulandı ve notlandı (yerel ek olarak).
- [ ] Online tokenlar env’de mevcut ve doğrulandı (değerleri loglanmadan).
