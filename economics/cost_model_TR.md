# Maliyet Modeli (Şablon)

Bu doküman, eğitim ve çıkarım için hafif bir maliyet modelini içerir.

## Değişkenler
- `P`: parametre sayısı
- `T`: toplam token
- `F`: token başına FLOPs (yaklaşık)

## Yaklaşım
- **Eğitim FLOPs** ≈ `6 * P * T` (kabaca)
- **Çıkarım FLOPs** ≈ `2 * P * T`

Üretim koşularından sonra gerçek ölçümlerle güncellenmelidir.
