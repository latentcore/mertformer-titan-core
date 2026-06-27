# Postmortem — Örnek 001 (Dry-Run)

- **Olay ID**: PM-001
- **Tarih/Saat**: 2026-02-05 07:40
- **Etkisi**: Eğitim koşusu 120. adımda NaN saptandığı için durdu.
- **Kök Sebep**: LR warmup sonrası gradient patlaması; grad clip yüksek kaldı.
- **Tespit**: `nan_kill_test.py` güvenlik kapısı tetiklendi; log NaN uyarısı verdi.
- **Çözüm**: `grad_clip` düşürüldü, ek loss kontrolleri eklendi.
- **Önleme**: Grad norm trendi için erken uyarı ve LR üst sınırı.

Durum: **Süreç hazırlığı için dry-run örnek.**
