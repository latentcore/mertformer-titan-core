# Güvenlik ve Emniyet Politikası

## Kapsam
Bu doküman, MertFormer Titan için güvenlik, yönetişim ve deployment sınırlarını tanımlar.

## Çekirdek Güvenlik Sınırları
- Sistem zararlı, yasa dışı veya kötü amaçlı kullanım için kullanılmaz.
- Sistem otonom saldırı veya gizli gözetim aracı gibi çerçevelenmez.
- High-risk kararlar insan onayı ister ve denetlenebilir kalmalıdır.

## Doğruluk ve Kanıt Disiplini
- Kanıt yoksa claim yoktur.
- `measured`, `target` ve `vision` ifadeleri ayrı kalmalıdır.
- `verified`, `hypothesis` ve `creative_or_folklore` modları birbirine karıştırılamaz.

## Değişiklik Kontrolü
- Eğitim davranışını etkileyebilecek değişiklikler açık inceleme ister ve denetlenebilir kalmalıdır.
- Runtime-invasive değişiklikler, kanonik eğitim yoluna açıkça zarar vermediği gösterilmedikçe ertelenir.

## Veri ve Secret Yönetimi
- Veri lisanslarına, provenance zincirine ve saklama sınırlarına uyun.
- Secret’ları versiyon kontrolü ve release artefaktlarından uzak tutun.
- Veri seti, rapor ve release bundle’ları için audit-ready manifest üretin.

## Güvenlik Açığı Bildirimi
- Güvenlik açıklarını gizli olarak GitHub Security Advisories üzerinden bildirin (depo → **Security** sekmesi → **Report a vulnerability**). Public issue açmayın.
- Operasyonel olaylar `postmortems/` altında şablonla kaydedilir; çözümden sonra mitigation adımları güncellenir.

## Durum
Bu politika mevcut pre-training (Build 30 V2) aşaması için aktiftir.
