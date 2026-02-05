# Model Kartı — MertFormer Titan (v27.0)

## Genel Bakış
MertFormer Titan, BitNet 1.58-bit katmanları, LiquidRouter MoE ve MLA temelli,
mobil-odaklı 2.64B parametreli bir dil modelidir. Bu kart **eğitim öncesi**
durumu yansıtır ve kullanım amacı, sınırlar ve bilinen boşlukları belgeler.

## Amaçlanan Kullanım
- Cihaz içi/edge çıkarım araştırması ve prototipleme
- Mobil/embedded dağıtım denemeleri
- Verimli yönlendirme ve low-bit inference çalışmaları

## Kapsam Dışı / Hedeflenmeyen Kullanım
- Bağımsız doğrulama olmadan güvenlik-kritik kullanım
- Tıbbi, hukuki veya savunma kararları (insan denetimi olmadan)
- Veri lisanslarına ve gizlilik kurallarına aykırı kullanım

## Eğitim Verisi (Güncel Envanter)
Eğitim veri setleri `datasets/SOURCES.md` dosyasında listelenmiştir.
Üretim eğitiminden önce **snapshot, hash ve lisans doğrulaması** zorunludur.

## Değerlendirme Durumu
- Benchmarklar: **Henüz tamamlanmadı** (eğitim öncesi).
- Planlanan: HumanEval / MBPP / GSM8K (stabil baseline sonrası).

## Sınırlamalar
- Performans metrikleri şu an **hedef/tahmin** seviyesindedir.
- Low-bit kernel yolu **deneysel** ve opt-in’dir.
- Sertifikalı güvenlik değerlendirmesi yoktur; dikkatli kullanım gerekir.

## Sorumlu Kullanım
Kullanım `USE_POLICY.md` ile sınırlandırılır. Yerel yasalar, gizlilik
düzenlemeleri ve kurumsal kurallar gözetilmelidir.

## İletişim
Araştırma ve işbirliği için README Contact bölümüne bakın.
