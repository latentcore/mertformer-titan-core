# İnceleme Checklisti (Dış Mühendislik)

Bu checklist, dış mühendislik incelemesi (güvenlik, uyum, yeniden üretilebilirlik, operasyon) için hazırlanmıştır.

## 1) Klonla ve Doğrula (Offline)

Deterministik bootstrap:

```bash
bash scripts/bootstrap_venv.sh
```

Offline verify-all:

```bash
bash scripts/verify_all.sh
```

Beklenen:
- Secret scan PASS
- `pytest` PASS (varsa dokümante edilmiş SKIP'ler)
- Offline modda preflight PASS
- Operator gate PASS (safe mod, offline)

## 2) Offline-First Sözleşmesi

Varsayılanları doğrula:
- Default `TITAN_OFFLINE=1` (HF/WandB login veya dataset download yok)
- Online davranış ancak açıkça env/flag ile açılır

Kanıt:
- `run.sh` (offline/online gating)
- `scripts/titan_preflight.py` (offline secrets davranışı + mock tokenizer)

## 3) Secrets Hijyeni

Doğrula:
- `.env` gitignore
- `logs/` gitignore (sadece artifact; tek track'li doküman `logs/README.md`)
- CI secret scan mevcut ve tracked dosyalarda çalışıyor

Kanıt:
- `.gitignore`
- `scripts/secret_scan.py`
- `.github/workflows/ci.yml`

## 4) Dataset Provenans & Lisanslar (P0 Gate)

Envanterin tamam olduğunu doğrula:
- Otomatik envanter: `datasets/inventory.md` / `datasets/inventory_TR.md`
- Kaynak listesi: `datasets/SOURCES*.md`
- Lisans checklisti: `datasets/LICENSES*.md`
- Snapshot hash kaydı: `datasets/hashes.json`

Üretim eğitimi için sert kural:
- Çekirdek eğitim datasetlerinde `TBD` lisans kalamaz (opsiyonel/demo datasetler doğrulanana kadar devre dışı kalmalı)
- Gerçek eğitimde kullanılacak snapshotlar için hash kayıtları doldurulmuş olmalı

## 5) Reprodüsibilite

Doğrula:
- Python baseline pin: `pyproject.toml` (`>=3.11,<3.12`)
- Bootstrap script mevcut ve dokümante
- Accelerate config örneği mevcut: `repro/accelerate_default.yaml`

Kanıt:
- `repro/python_TR.md`
- `scripts/bootstrap_venv.sh`

## 6) CI Kapsamı

CI gate'lerini doğrula:
- Secret scan
- Ruff (lightweight)
- Preflight (offline)
- Pytest
- Operator gate (safe, offline)

Kanıt:
- `.github/workflows/ci.yml`

## 7) Demo (ML Dışı Görsel)

Demo bağımlılıklarını kur:

```bash
bash scripts/bootstrap_venv.sh --demo
```

Çalıştır:

```bash
.titan-venv/bin/python snake_demo.py
```

Beklenen:
- Header: `MERTFORMER TITAN v1.0 [LIVE DEMO]`
- Telemetry: `Reasoning Speed: 30ms`, `Tokens: 1.58b`, `Score: X`
- Otomatik oynar, ölürse anında restart eder

## Kanıt Paketleme (Ne Eklenecek?)

- `scripts/verify_all.sh` console çıktısı (token yok)
- `logs/` altında üretilen loglar (opsiyonel, sanitize, commit edilmez)
- Commit SHA ve ortam notları

