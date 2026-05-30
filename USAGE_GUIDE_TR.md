# Kullanım Kılavuzu (Build 30)

Bu kılavuz, inceleme yapanlar ve operatörler için hızlı operasyon başlangıcıdır.

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

## 3) Kanonik 45K Başlatma Kapısı

```bash
bash zero_touch_start.sh --check-only
```

Her gerçek 45K eğitim denemesinden önce bu komutu kullanın. Komut, kesin readiness kararını, blocker reason code'larını ve kanonik runtime sözleşmelerini yeniler.
45K koşusunu nihai yetenek tavanı olarak değil, ilk ciddi mimari doğrulama koşusu olarak ele alın.

Mevcut repo tarafı durum:
- `TRAIN_ALLOWED` via `remote_bootstrap`
- katı yerel `offline_clean`, yerel logits veya uygulanabilir yerel Phase-0 olmadan bloklu kalır
- kalan kazanan olmayan blocker: `online_teacher:MISSING_HF_TOKEN`

## 4) Kanonik 45K Başlatıcı

```bash
HF_TOKEN=... TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh
```

Katı yerel offline-clean yolu (yalnızca logits veya yerel Phase-0 koşulu sağlandığında):

```bash
TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh
```

Yaygın kontrol bayrakları:
- `--plan-only`
- `--dry-run`
- `--resume auto`
- `--resume off`
- `--resume /abs/path/to/checkpoint.pt`
- `--no-post`
- `--post-only`
- `--bench-only`
- `--demo-only`
- `--export-only`
- `--readme-update-only`

## 4.5) Opsiyonel Hedef Makine Hız Kontrolleri

Bu kontroller smoke/full-run ayarı için kablolanmıştır; **benchmark iddiası değildir**. Muhafazakâr baseline hattında, equivalence testleri ve kısa hedef makine smoke koşusu geçmeden kapalı tutulmalıdır.

Opsiyonel packed projection ve Liquid eğitim yolları için equivalence kontrolü:

```bash
python3 -m pytest -q tests/test_packed_projection_equivalence.py tests/test_liquid_safeguard.py
```

Kanonik kapı incelendikten sonra örnek Ocean 2x H200 smoke komutu:

```bash
HF_TOKEN=... \
TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable \
TITAN_BATCH_SIZE=1024 TITAN_BATCH_SIZE_FALLBACKS=1024,512,256 \
TITAN_DATALOADER_PIN=1 TITAN_DATALOADER_NONBLOCKING=1 \
TITAN_FFN_PACK=1 TITAN_MOE_PACK=1 TITAN_MLA_KV_PACK=1 \
TITAN_LIQUID_FAST_PATH=0 TITAN_LIQUID_TRAIN_IMPL=packed_pair \
MERTFORMER_LOWBIT_KERNEL=0 MERTFORMER_FUSED_BACKWARD=0 \
bash zero_touch_start.sh --dry-run
```

Operasyon sınırları:
- `repro/accelerate_8xgpu.yaml`, yeniden üretilebilirlik/koşu yapılandırması olduğu için `repro/` altında tutulur; `configs/` altındaki stabil model config sözleşmelerine ait değildir.
- `TITAN_BATCH_SIZE_FALLBACKS=1024,512,256` yalnızca net OOM sinyalinde kullanılır; OOM olmayan hata batch değiştirmeden durur.
- `TITAN_FFN_PACK`, `TITAN_MOE_PACK` ve `TITAN_MLA_KV_PACK` varsayılan kapalıdır ve equivalence testleriyle kapsanır.
- İlk Ocean uzun koşusunda `TITAN_LIQUID_FAST_PATH=0` tutulur ve `packed_pair_compile` kullanılmaz.
- `MERTFORMER_LOWBIT_KERNEL=1` açıksa packed projection yolları, deneysel low-bit inference kernel sınırını baypas etmemek için baseline yola düşer.
- Gerçek throughput yalnızca hedef makine loglarından çıkarılmalıdır; öngörülen hızlanmalar ölçülmüş iddia gibi sunulmamalıdır.

## 5) Offline Test Girişi

```bash
TITAN_OFFLINE=1 bash run.sh --test
```

Beklenen:
- Harici login/download yok
- Preflight kontrollerinden sonra çıkış

`run.sh`, legacy offline/test/demo akışları için yardımcı giriş olarak kalır. Kanonik 45K train-end başlatıcısı artık bu dosya değildir.

## 6) Operator Gate (Manuel)

```bash
.titan-venv/bin/python scripts/operator_mode_gate.py \
  --no-pytest \
  --overfit-dataset datasets/validation.jsonl
```

## 7) Demo Komutları

Snake canlı demo:

```bash
.titan-venv/bin/python snake_demo.py
```

30 saniyelik kanıt videosu (headless):

```bash
.titan-venv/bin/python snake_demo.py \
  --headless \
  --record assets/snake_demo_proof.mp4 \
  --record-seconds 30
```

## 8) Yaygın Sorunlar

- Yeni Torch/Python sürümünde `onnx export fail`:
  - Güncellenmiş exporter yolu `scripts/test_onnx_export.py` içinde dynamo/legacy uyumluluğu için korunur.
- `wandb/hf token missing`:
  - Yerel review makinesinde beklenebilir.
  - Önerilen `remote_bootstrap` hedef makine hattında, launch öncesi `HF_TOKEN` enjekte edilmelidir.
  - Katı yerel `offline_clean` hattı, yalnızca logits zaten lokalde varsa veya yerel Phase-0 uygulanabilirse yeşil kalabilir.
- `venv command path mismatch`:
  - Komutları modül stili çalıştırın: `.titan-venv/bin/python -m ...`
- Opsiyonel hız flag'leri uyumsuzluk veya kararsızlık üretirse:
  - Önce ilgili flag'i kapatın (`TITAN_FFN_PACK=0`, `TITAN_MOE_PACK=0`, `TITAN_MLA_KV_PACK=0` veya `TITAN_LIQUID_TRAIN_IMPL=baseline`) ve uzun koşudan önce equivalence testlerini yeniden çalıştırın.

## 8.5) Otomatik Cache Temizleme Sarmalayıcısı

Komut çalıştırma + garantili cache temizliği için:

```bash
python3 scripts/run_and_clean_pycache.py --full-clean --include-tool-caches --include-venv-caches -- \
  python3 -m pytest -q
```

## 9) Güvenlik Politikası (Zorunlu)

- Secret değerleri asla yazdırmayın veya commit etmeyin.
- `.env` paketlenmemelidir.
- Üretilen loglar artefakttır; yalnızca sanitize edilmiş dokümanlar track edilir.

## 10) Teslim Öncesi Kontrol Listesi

- [ ] `bash scripts/verify_all.sh` PASS
- [ ] `bash zero_touch_start.sh --check-only` incelendi
- [ ] Opsiyonel hız flag'leri açıldıysa packed/Liquid equivalence testleri PASS
- [ ] `TITAN_OFFLINE=1 bash run.sh --test` PASS
- [ ] README ve README_TR linkleri geçerli
- [ ] Eksik EN/TR markdown çifti yok
- [ ] Paket denylist temiz (`.env`, venv, cache, log yok)
