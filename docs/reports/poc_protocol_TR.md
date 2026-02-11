# Pilot / PoC Protokolü

## Amaç
MertFormer Titan’ın edge‑native kodlama yeteneğini ve operasyonel güvenliğini kontrollü bir pilotta doğrulamak.

## Kapsam
- Offline veya air‑gapped inference iş akışı
- Operator‑mode safety gate’ler
- Mutabık test setinde benchmark çıktıları

## Süre
- Önerilen: 2–4 hafta (kuruma göre ayarlanabilir)

## Başarı Kriterleri (Örnek)
- Operator‑mode gate’lerin geçmesi (kill‑switch, failure budget, restore drill)
- Non‑finite olay olmadan stabil inference
- Pilot prompt setinde hedef başarı (kurum tanımlar)
- Benchmark çıktılarının üretilebilir ve tekrarlanabilir olması

## Veri & Güvenlik
- Kurum, sanitize edilmiş prompt seti sağlar
- Inference için dış ağ bağımlılığı gerekmez
- Loglar, mutabık politikaya göre paylaşılır

## Teslimatlar
- Benchmark çıktı dosyaları (JSONL)
- Gate logları ve manifest
- Kısa pilot raporu (sonuç + öneriler)

## Sorumluluklar
- MertFormer: kurulum, runbook, troubleshooting
- Partner: ortam erişimi, veri onayı, güvenlik onayı

## Riskler & Önlemler
- Risk: instabilite → Önlem: kill‑switch ve failure budget
- Risk: veri kısıtları → Önlem: offline iş akışları
- Risk: performans sapması → Önlem: mutabık kabul eşikleri

## Kabul
- Başarı kriterlerine göre ortak imza
