# Model Card — MertFormer Titan (Build 30 V2)

## Genel Bakış
MertFormer Titan; BitNet tarzı düşük bit katmanlar, Liquid yönlendirme dinamikleri ve sparse MoE yürütmesi etrafında kurulan, pre-training aşamasındaki offline-first ve edge-native bir dil modeli yığınıdır.

## Resmi Konumlandırma
Türkiye’ye fayda sağlayacak, offline-first, edge-native, yerli ve entegre edilebilir zeka altyapısı.

## Doğruluk Etiketleri
- `measured`: somut artefakt, benchmark, manifest veya log ile destekli
- `target`: henüz doğrulanmamış planlanan davranış
- `vision`: mevcut claim kapsamı dışındaki uzun vadeli yön
- `verified`, `hypothesis` ve `creative_or_folklore` ayrı çıktı modlarıdır

## Mevcut Durum
- Repo durumu: pre-training / claim-unverified
- Runtime toplamı: repo artefaktlarında ölçülmüş durumda
- Benchmark uygunluğu: eğitimli checkpoint olmadan `NOT ELIGIBLE FOR CLAIM`
- 45K koşusu: ilk ciddi mimari doğrulama koşusu, nihai kapasite tavanı değil

## Amaçlanan Kullanım
- Offline-first ve edge-native deneyler
- Denetlenebilir yerel deployment araştırması
- İnsan denetimli karar destek sistemleri

## Kapsam Dışı
- Eğitim kanıtı olmadan production veya certified safety claim’i
- İnsan incelemesi olmadan high-risk kullanım
- Gizli gözetim veya zararlı otonomi çerçevesi

## Eğitim ve Veri
Eğitim verisi, lisanslar, hash’ler ve stage bileşimi `datasets/` kaynak dosyaları ile closure sırasında üretilen provenance raporları tarafından yönetilir.

## Sorumlu Kullanım
Kullanım `USE_POLICY.md` ve `SECURITY.md` tarafından yönetilir.
