# Postmortem — Örnek 001 (Dry-Run)

- **Olay ID**: PM-001
- **Tarih/Saat**: 2026-02-05 07:40
- **Etkisi**: Eğitim kosusu 120. adimda NaN saptandigi için durdu.
- **Kok Sebep**: LR warmup sonrasi gradient patlamasi; grad clip yuksek kaldi.
- **Tespit**: `nan_kill_test.py` guvenlik kapisi tetiklendi; log NaN uyarisi verdi.
- **Çözüm**: `grad_clip` dusuruldu, ek loss kontrolleri eklendi.
- **Onleme**: Grad norm trendi için erken uyarı ve LR ust siniri.

Durum: **Surec hazirligi için dry-run örnek.**
