# Guvenlik ve Emniyet Politikasi

## Kapsam
Bu dokuman, MertFormer Titan ve eğitim/değerlendirme hatti için guvenlik sinirlarini tanimlar.

## Kullanım Sinirlari
- Modeli zararli, kotuye kullanım veya yasa disi amaclarla kullanmayin.
- Eğitim sirasinda hassas veya regule verileri dis servislerle paylasmayin.

## Kill-Switch ve Failure Budget
- Sayisal kararsizlik, kill-switch drill'leri ve failure budget korumalari ile ele alinir.
- Bkz: `scripts/nan_kill_test.py`, `orchestrator/failure_budget.py`.

## Veri Yonetimi
- Veri seti lisanslarina ve kaynaklarina saygi gosterin.
- Lisans izin vermedikce veri setlerini yeniden dagitmayin.

## Raporlama
- Olayları `postmortems/` altinda sablonla kaydedin.
- Çözüm sonrası önleyici adımları güncelleyin.

## Durum
Bu politika **temel şablondur** ve üretim koşuları sonrası güncellenecektir.
