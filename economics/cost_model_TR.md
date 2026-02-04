# Maliyet Modeli (Sablon)

Bu dokuman, egitim ve cikartim icin hafif bir maliyet modelini icerir.

## Degiskenler
- `P`: parametre sayisi
- `T`: toplam token
- `F`: token basina FLOPs (yaklasik)

## Yaklasim
- **Egitim FLOPs** ≈ `6 * P * T` (kabaca)
- **Cikartim FLOPs** ≈ `2 * P * T`

Uretim kosularindan sonra gercek olcumlerle guncellenmelidir.
