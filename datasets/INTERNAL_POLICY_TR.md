# Dahili Dataset Politikasi (Stage Setleri, Golden Samples)

Bu dokuman, bu repo tarafindan uretilen veya referanslanan **dahili** datasetler için politikayi tanimlar:
- **Stage setleri**: `scripts/data_pipeline.py` ile uretilen yerel curriculum snapshot'lari (ornegin `datasets/stage*/stage*_data.jsonl`).
- **Golden samples**: kisa regression/sanity prompt setleri (ornegin `datasets/golden_samples.jsonl`).

## Siniflandirma
- **Lisans/sahiplik**: Internal (proprietary).
- **Dagitim**: Acik onay olmadan organizasyon disina paylasilmaz.

## Icerik Kurallari (Zorunlu)
- Secret/token/credential/private key/internal URL bulunamaz.
- Kisi verisi (PII) veya hassas veri (saglik, biyometrik vb.) bulunamaz.
- Telifli / kisitli 3. parti icerik ancak su sartlarla kullanilabilir:
  - kaynak lisansi eğitim/yeniden dagitim için acik izin veriyorsa ve
  - `datasets/SOURCES*.md` + `datasets/LICENSES*.md` icinde dokumante edilmisse.

## Provenans & Hash
- Her eğitim kosusu **tam olarak kullanilan** snapshot'i pinlemelidir:
  - snapshot metadata + SHA256 kaydi `datasets/hashes.json` icine girilir
  - ham snapshotlar git'e konmaz (`datasets/` varsayilan olarak gitignore'dur)

## Review Kapisi (Gate)
Production eğitimden once:
- `datasets/LICENSES*.md` icinde eğitim datasetleri için **TBD kalamaz**.
- `datasets/hashes.json` eğitimde kullanilan snapshot için doldurulmus olmali.

