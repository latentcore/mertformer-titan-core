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

## Readiness Guardrail
- Bu geçişte ana ship gate 45K readiness’tir.
- Runtime-invasive değişiklikler ancak 45K yoluna açıkça zarar vermediği gösterilirse kabul edilir; aksi halde phase-2’ye taşınır.
- Bu geçişin resmi risk tavanı Medium Refine’dır.

## Veri ve Secret Yönetimi
- Veri lisanslarına, provenance zincirine ve saklama sınırlarına uyun.
- Secret’ları versiyon kontrolü ve release artefaktlarından uzak tutun.
- Veri seti, rapor ve release bundle’ları için audit-ready manifest üretin.

## Raporlama
- Olayları `postmortems/` altında şablonla kaydedin.
- Çözümden sonra mitigation adımlarını güncelleyin.

## Durum
Bu politika Build 30 Max Closure geçişi için aktiftir.
