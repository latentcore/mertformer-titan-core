# Guvenlik ve Emniyet Politikasi

## Kapsam
Bu dokuman, MertFormer Titan ve egitim/degerlendirme hatti icin guvenlik sinirlarini tanimlar.

## Kullanim Sinirlari
- Modeli zararli, kotuye kullanim veya yasa disi amaclarla kullanmayin.
- Egitim sirasinda hassas veya regule verileri dis servislerle paylasmayin.

## Kill-Switch ve Failure Budget
- Sayisal kararsizlik, kill-switch drill'leri ve failure budget korumalari ile ele alinir.
- Bkz: `scripts/nan_kill_test.py`, `orchestrator/failure_budget.py`.

## Veri Yonetimi
- Veri seti lisanslarina ve kaynaklarina saygi gosterin.
- Lisans izin vermedikce veri setlerini yeniden dagitmayin.

## Raporlama
- Olaylari `postmortems/` altinda sablonla kaydedin.
- Cozum sonrasi onleyici adimlari guncelleyin.

## Durum
Bu politika **temel sablondur** ve uretim kosulari sonrasi guncellenecektir.
