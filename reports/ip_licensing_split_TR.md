# Sektörel Fikri Hak Ayrımı Çerçevesi (Savunma ve Sivil)

## Amaç
Bu doküman, aşağıdaki iki alanı müzakere edilebilir ve profesyonel bir çerçevede ayırmak için hazırlanmıştır:

- Savunma/güvenlik alanı kullanım hakları (kuruma münhasır)
- Sivil/ticari alan kullanım hakları (geliştiricide saklı)

Bu metin teknik devir görüşmeleri için taslaktır; imza öncesi hukuk incelemesi zorunludur.

## Kapsam
MertFormer Titan mimarisi, ilgili kaynak kodlar ve türev dağıtımlar için geçerlidir.

Kapsam dışı:

- Gizli veri işleme prosedürleri
- Ülkeye özgü ihale/mevzuat detayları
- Bağlayıcı sözleşmenin nihai hukuk dili

## Önerilen Hak Ayrımı

### 1. Savunma Alanı (Kuruma Münhasır Lisans veya Devir)

- Savunma/güvenlik kullanım senaryolarında kuruma münhasır hak tanımlanır.
- Savunma kapsamı için kaynak erişimi, dağıtım, uyarlama ve iç türev hakları kapsanır.
- Sözleşme dışında üçüncü taraflara aktarım/satış, ayrıca onaya tabi tutulur.

### 2. Sivil Alan (Geliştiricide Saklı Haklar)

- Geliştirici, sivil ürün haklarını saklı tutar:
  - Tüketici uygulamaları
  - Eğitim ürünleri
  - Savunma dışı ticari yazılımlar
- Sivil dağıtım ve ürünleştirme yetkisi geliştiricide kalır.

### 3. Geçiş ve Sorumluluk Devri

- Sabit bir geçiş destek süresi tanımlanır (ör. 3 ay).
- Geçiş kabulünden sonra operasyonel sorumluluk kuruma devrolur.
- Geçiş sonrası destek, ayrı SLA/sözleşme kapsamında yürütülür.

## Ticari Yapı (Şablon)

- Teknoloji transfer bedeli (tek seferlik)
- İsteğe bağlı bakım paketi (süreli)
- İsteğe bağlı entegrasyon paketi (milestone bazlı)

## Asgari Sözleşme Maddeleri Kontrol Listesi

- Alan tanımları (savunma ve sivil)
- Münhasırlık sınırları
- Türev çalışmaların hak sınırları
- Gizlilik ve veri işleme hükümleri
- Garanti/sorumluluk sınırları
- Geçiş kabul kriterleri
- Geçiş sonrası destek koşulları
- Fesih ve uyuşmazlık çözümü

## Pilot/Devir Operasyon Teslimleri

- `verify_all` sonuç özeti
- Operator mode gate özeti
- `pilot_report_v1` JSON
- Offline çalıştırma runbook'u
- Risk ve claim-eligibility beyanı
- İmzalı kabul sayfası

## Durum

Bu doküman profesyonel hazırlık amaçlı çerçeve taslağıdır.
Hukuki bağlayıcılık, hukuk incelemesi ve imzalı nihai sözleşme metnine bağlıdır.
