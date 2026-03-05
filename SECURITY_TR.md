# Güvenlik ve Emniyet Politikası

## Kapsam
Bu doküman, MertFormer Titan ve eğitim/değerlendirme hattı için güvenlik sınırlarını tanımlar.

## Kullanım Sınırları
- Modeli zararlı, kötüye kullanım veya yasa dışı amaçlarla kullanmayın.
- Eğitim sırasında hassas veya regüle verileri dış servislerle paylaşmayın.

## Kill-Switch ve Failure Budget
- Sayısal kararsızlık, kill-switch drill'leri ve failure budget korumaları ile ele alınır.
- Bkz: `scripts/nan_kill_test.py`, `orchestrator/failure_budget.py`.

## Veri Yönetimi
- Veri seti lisanslarına ve kaynaklarına saygı gösterin.
- Lisans izin vermedikce veri setlerini yeniden dagitmayin.

## Raporlama
- Olayları `postmortems/` altında sablonla kaydedin.
- Çözüm sonrası önleyici adımları güncelleyin.

## Durum
Bu politika **temel şablondur** ve üretim koşuları sonrası güncellenecektir.
