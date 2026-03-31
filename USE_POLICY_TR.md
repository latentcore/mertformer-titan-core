# Kullanım Politikası — MertFormer Titan

## Amaç
Bu politika, MertFormer Titan kod tabanı, model ve artefaktlarının kabul edilebilir ve kısıtlı kullanımını tanımlar.

## İzinli Kullanım
- Kontrollü ortamlarda araştırma ve değerlendirme
- Offline-first ve edge-native deneyler
- Uyum incelemesi ile kurum içi prototipleme
- İnsan onay sınırları net olan karar destek sistemleri

## Kısıtlı Kullanım
- İnsan denetimi olmadan high-stakes kullanım
- Zararlı otonomi, gizli gözetim veya yetkisiz takip
- Veri lisanslarını, gizliliği veya politika sınırlarını ihlal eden kullanım
- Doğrulanmamış kabiliyeti gerçek ölçüm veya kesin zeka gibi sunmak

## Çıktı Doğruluk Modları
- `verified`: kaynak, ölçüm veya artefakt ile destekli ifade
- `hypothesis`: ileriye dönük ama henüz kanıtlanmamış ifade
- `creative_or_folklore`: açık etiketli anlatı veya stilistik çıktı
- Varsayılan mod `verified`

## Veri Kullanımı
- Açık onay olmadan hassas veya regüle veri yüklemeyin
- Veri minimizasyonu ve saklama politikalarına uyun
- Yeni veri eklerini `datasets/LICENSES.md`, `datasets/SOURCES.md` ve `datasets/hashes.json` ile doğrulayın

## 45K Guardrail
- Bu geçişte ana ship gate 45K readiness’tir
- Bir iş 45K readiness, reproducibility veya closure confidence riskini artırıyorsa phase-2’ye düşer
- 45K öncesi açık uçlu mimari refactor yoktur

## Güvenlik ve Yönetişim
- Gizli verileri versiyon kontrolüne koymayın
- Eğitimden önce preflight ve readiness kontrollerini çalıştırın
- High-risk aksiyonlar insan onayı ve denetlenebilir log ister

## Yürütme
İhlaller erişim iptali ve ilgili sahiplerine raporlama ile sonuçlanabilir.
