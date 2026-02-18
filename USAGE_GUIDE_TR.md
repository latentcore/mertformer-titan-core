# Kullanım Kılavuzu (Build 30)

Bu kılavuz, inceleme ve operasyon için hızlı uygulama adımlarını verir.

## 1) Ortam Kurulumu

```bash
bash scripts/bootstrap_venv.sh
```

Temel:
- Python: 3.11.x
- Varsayılan mod: `TITAN_OFFLINE=1`
- Sanal ortam: `.titan-venv`

## 2) Hızlı Doğrulama (Önerilen)

```bash
bash scripts/verify_all.sh
```

Bu komut şunları çalıştırır:
- Secret scan
- Pytest
- Preflight (offline)
- Operator-mode gate (safe offline)

## 3) Offline Test Girişi

```bash
TITAN_OFFLINE=1 bash run.sh --test
```

Beklenen:
- Dış login/download yok
- Preflight sonrası çıkış

## 4) Operator Gate (Manuel)

```bash
.titan-venv/bin/python scripts/operator_mode_gate.py \
  --no-pytest \
  --overfit-dataset datasets/validation.jsonl
```

## 5) Demo Komutları

Snake canlı demo:
```bash
.titan-venv/bin/python snake_demo.py
```

30 sn kanıt videosu (headless):
```bash
.titan-venv/bin/python snake_demo.py \
  --headless \
  --record assets/snake_demo_proof.mp4 \
  --record-seconds 30
```

## 6) Yaygın Sorunlar

- Yeni Torch/Python sürümünde ONNX export hatası:
  - `scripts/test_onnx_export.py` dosyasında dynamo/legacy uyum koruması bulunuyor.
- `wandb/hf token` yok:
  - Offline modda beklenen ve bloklamayan durumdur.
- Venv path uyuşmazlığı:
  - Komutları modül stili çalıştırın: `.titan-venv/bin/python -m ...`

## 7) Güvenlik Politikası (Zorunlu)

- Secret değerleri asla yazdırmayın/commit etmeyin.
- `.env` paketlere dahil edilmez.
- Üretilen loglar artefakttır; repoda yalnızca sanitize doküman kalır.

## 8) Teslim Öncesi Kontrol Listesi

- [ ] `bash scripts/verify_all.sh` PASS
- [ ] `TITAN_OFFLINE=1 bash run.sh --test` PASS
- [ ] README ve README_TR linkleri geçerli
- [ ] Eksik EN/TR markdown çifti yok
- [ ] Paket denylist temiz (`.env`, venv, cache, log yok)
