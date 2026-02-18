# Veri Setleri Genel Bakış

Bu dizin, eğitim hattında kullanılan aşamalı müfredat veri setlerini ve değerlendirme setlerini içerir.

## Yapı
- `stage1/` ... `stage5/` müfredat aşamaları
- `stage4_soul/`, `stage5_tools/` özel alt kümeler
- `golden_samples.jsonl` iç golden prompt seti
- `validation.jsonl` doğrulama seti

## İddia Seviyesi Validation Seti
Smoke test için küçük validation yeterli olabilir. Benchmark/iddia koşuları için temsil gücü olan set üretin:

```bash
python3 scripts/build_validation_set.py --target-size 1500
python3 scripts/record_dataset_hashes.py
```

Eğitim kapısı:
- `TITAN_CLAIM_MODE=1` aktifse minimum validation boyutu (`cfg.validation_min_samples_claim`) zorlanır.

## Kaynaklar
Kaynak ve lisans notları için `datasets/SOURCES.md` ve `datasets/LICENSES.md` dosyalarına bakın.
