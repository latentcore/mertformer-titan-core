# Ownership ve Rol Tanımı

## Rol Modeli
- Seller/Builder (Mert): mimari niyet, teknik bağlam, geçişte karar desteği.
- Buyer/Operator: üretim işletimi, bakım, SLA, güvenlik yaması, uyumluluk yürütümü.

## Karar Yetkileri
1. Mimari roadmap ve yeni modül açma: Buyer approval required.
2. Release ve rollback kararı: Buyer Ops + Security.
3. Claim/public statement onayı: Buyer Legal + Product.
4. Incident seviyesi yükseltme: Buyer On-call lead.

## Sorumluluk Bölümü
- Seller (devir sonrası sınırlı):
1. Bilgi transferi ve handover oturumları.
2. Mevcut release kapsamının teknik açıklamaları.

- Buyer (devir sonrası ana sorumlu):
1. Üretim operasyonu ve on-call.
2. Güvenlik patch yönetimi.
3. Lisans/hash audit döngüsü.
4. Re-train/fine-tune takvimi ve bütçesi.

## Operasyon Sınırları
- Devir sonrası değişiklikler yeni release numarasıyla yürütülür.
- Kapanmış release tag'ine doğrudan müdahale yapılmaz.
