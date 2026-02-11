# Güvenlik & Uyum Özeti

## Amaç
Bu doküman, MertFormer Titan dağıtımlarının güvenlik duruşunu ve uyum hedeflerini özetler. **Resmî sertifikasyon** veya üçüncü taraf denetimi iddiası değildir.

## Kapsam
- Eğitim pipeline’ı ve operator‑mode gate’ler
- Inference dağıtımları (cihaz‑içi / edge)
- Loglama, denetlenebilirlik ve veri işleme

## Veri İşleme
- Varsayılan tasarım **offline‑first** ve **cihaz‑içi**.
- Eğitim verisi kaynakları kontrol edilir ve loglanır (reproducibility stamp).
- Inference için dışarı veri aktarımı gerekmez.

## Erişim Kontrolü
- Eğitim ve dağıtım ortamlarında en az ayrıcalık prensibi.
- Gizli bilgiler env üzerinden yüklenir, loglara yazılmaz.

## Loglama & Denetlenebilirlik
- Run manifest: git hash, config, seed, dataset hash.
- JSONL logları zincir hash ile forensik bütünlüğe sahip.
- Operator‑mode gate’ler yapılandırılmış kanıt logları üretir.

## Model Güvenliği
- Non‑finite kill‑switch ile kararsız koşu durdurulur.
- Failure budget, öğrenme durursa pivot/debug tetikler.
- Checkpoint restore drill ile durum bütünlüğü doğrulanır.

## Ortam İzolasyonu
- Air‑gapped ve offline dağıtım senaryoları desteklenir.
- Reproducible ortam için opsiyonel Docker.

## Olay Müdahalesi (Operasyonel)
- NaN/instability durumunda fail‑fast.
- Post‑mortem için net audit izleri.
- Manual kill/resume runbook mevcut.

## Uyum Hedefleri (Niyet)
(Sertifikasyon iddiası değildir.)
- ISO 27001: loglama, erişim kontrolü, değişiklik yönetimi.
- GDPR ilkeleri: veri minimizasyonu, amaç sınırlaması, erişim kontrolü.
- Savunma sınıfı duruş: offline inference, kısıtlı veri erişimi.

## Kapsam Dışı (Şu an)
- Resmî sertifikasyon denetimleri (ISO/SOC/FedRAMP).
- Hukuki onaylar veya ihracat uyumluluğu.

## Sonraki Adımlar
- Kuruma özel güvenlik gereksinimlerini entegre etmek.
- Gerekirse resmî denetim süreçlerini başlatmak.
