# GO Durum Matrisi (Pilot vs Ürün Claim)

## Kapsam
Bu matris, repoda tamamen biten işleri ve dış/operasyonel olarak kalan işleri ayırır.

## A) Pilot Teslim Hazırlığı (A1-A20)

| ID | Durum | Repo içinde tamamlanamadıysa nedeni |
| --- | --- | --- |
| A1-A18 | ✅ | Teknik ve dokümantasyon kapıları repo içinde tamamlandı. |
| A19 | ❌ | Hukuki onay dış danışman/imza gerektirir; kod ile otomatik tamamlanamaz. |
| A20 | ❌ | 2 ücretli pilot sözleşmesi veya 2 imzalı LOI gerekir; bu repo dışı ticari icradır. |

## B) Ürün/Benchmark Claim Hazırlığı (B1-B10)

| ID | Durum | Repo içinde tamamlanamadıysa nedeni |
| --- | --- | --- |
| B1 | ❌ | Gerçek stage snapshot ve final hash pinleme üretim veri akışından gelmelidir. |
| B2 | ❌ | Hedef eğitim donanımında tam pretrain/finetune koşusu gerekir. |
| B3 | ❌ | Eğitilmiş üretim checkpoint artefaktı gerekir. |
| B4 | ❌ | Benchmark çıktıları eğitilmiş checkpoint üzerinden üretilmelidir. |
| B5 | ❌ | Gerçek cihazda latency/power ölçümü gerekir. |
| B6 | ✅ | Claim dili ayrıştırıldı (ölçülmüş vs hedef/tahmin). |
| B7 | ❌ | Üçüncü taraf tekrar doğrulama dış doğrulayıcı ekip gerektirir. |
| B8 | ❌ | Nihai ticari lisans uyumu hukuk onayı gerektirir. |
| B9 | ❌ | Güvenlik/pentest raporu için ayrı güvenlik test kapsamı gerekir. |
| B10 | ❌ | SLA/incident/rollback sözleşme paketi iş/hukuk icrası gerektirir. |

## C) Pilot Teslim Paketi

| Kalem | Durum |
| --- | --- |
| verify_all özet/log | ✅ |
| operator gate JSON özeti | ✅ |
| pilot_report_v1 JSON | ✅ |
| offline tekrar çalıştırma adımları | ✅ |
| risk/limit notu | ✅ |
| kabul imza sayfası şablonu | ✅ |

## Güncel Kapı Kararı
- **Pilot teknik hazırlık:** GO
- **Ticari pilot kapanış hazırlığı:** A19 + A20 beklemede
- **Ürün/benchmark claim hazırlığı:** B1-B5, B7-B10 beklemede
