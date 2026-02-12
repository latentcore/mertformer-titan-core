# Varlık Kaynak Arşivi

Bu klasör, depoda kullanılan görsel varlıkların **düzenlenebilir kaynak dosyalarını** tutar.

## Amaç
- Mimari görselleri sürümler boyunca sürdürülebilir tutmak.
- Denetimler için kaynak-dan-çıktıya izlenebilirlik sağlamak.
- Son PNG/GIF çıktılarında geri döndürülemez düzenlemelerden kaçınmak.

## Beklenen Kaynak Formatları
- `.drawio` / `.xml` (diyagramlar)
- `.fig` (Figma çıktıları)
- `.psd` / `.ai` (tasarım ana dosyaları)
- Opsiyonel: düzenlenebilir ara çıktı olarak `.svg`

## İsimlendirme Kuralı
- `header_v<build>.{ext}`
- `synaptic_map_v<build>.{ext}`
- `*_source_v<build>.{ext}`

## Çıktı Eşleme Kuralı
Her kaynak dosya, `assets/` altındaki commit edilen bir çalışma zamanı varlığına eşlenmelidir:
- `header_v*.{ext}` -> `assets/header.png`
- `synaptic_map_v*.{ext}` -> `assets/synaptic_map.png`

Görsel güncellenirken:
1. Bu klasördeki kaynak dosyayı güncelleyin.
2. Çalışma zamanı görselini `assets/` altına export edin.
3. Güncellemeyi release notu veya snapshot raporunda belirtin.
