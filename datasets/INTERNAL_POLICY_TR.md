# Dahili Dataset Politikası (Stage Setleri, Golden Samples)

Bu doküman, bu repo tarafından üretilen veya referanslanan **dahili** datasetler için politikayı tanımlar:
- **Stage setleri**: `scripts/data_pipeline.py` ile üretilen yerel curriculum snapshot'lari (örneğin `datasets/stage*/stage*_data.jsonl`).
- **Golden samples**: kısa regression/sanity prompt setleri (örneğin `datasets/golden_samples.jsonl`).

## Sınıflandırma
- **Lisans/sahiplik**: Internal (proprietary).
- **Dağıtım**: Açık onay olmadan organizasyon dışına paylaşılmaz.

## İçerik Kuralları (Zorunlu)
- Secret/token/credential/private key/internal URL bulunamaz.
- Kişi verisi (PII) veya hassas veri (sağlık, biyometrik vb.) bulunamaz.
- Telifli / kısıtlı 3. parti içerik ancak şu sartlarla kullanılabilir:
  - kaynak lisansı eğitim/yeniden dağıtım için açık izin veriyorsa ve
  - `datasets/SOURCES*.md` + `datasets/LICENSES*.md` içinde dokümante edilmişse.

## Provenans & Hash
- Her eğitim koşusu **tam olarak kullanılan** snapshot'i pinlemelidir:
  - snapshot metadata + SHA256 kaydı `datasets/hashes.json` içine girilir
  - ham snapshotlar git'e konmaz (`datasets/` varsayılan olarak gitignore'dur)

## Review Kapısı (Gate)
Production eğitimden önce:
- `datasets/LICENSES*.md` içinde eğitim datasetleri için **TBD kalamaz**.
- `datasets/hashes.json` eğitimde kullanılan snapshot için doldurulmuş olmalı.

