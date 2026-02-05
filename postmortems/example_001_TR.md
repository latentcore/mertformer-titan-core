# Postmortem — Ornek 001 (Dry-Run)

- **Olay ID**: PM-001
- **Tarih/Saat**: 2026-02-05 07:40
- **Etkisi**: Egitim kosusu 120. adimda NaN saptandigi icin durdu.
- **Kok Sebep**: LR warmup sonrasi gradient patlamasi; grad clip yuksek kaldi.
- **Tespit**: `nan_kill_test.py` guvenlik kapisi tetiklendi; log NaN uyarisi verdi.
- **Cozum**: `grad_clip` dusuruldu, ek loss kontrolleri eklendi.
- **Onleme**: Grad norm trendi icin erken uyarı ve LR ust siniri.

Durum: **Surec hazirligi icin dry-run ornek.**
