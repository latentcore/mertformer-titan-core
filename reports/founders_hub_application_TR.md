# Microsoft Founders Hub Başvuru Taslağı (TR)

## Şirket
- Ad: MertFormer
- Web: Private repo
- Aşama: Prototype / Pre‑train

## Ürün Özeti
MertFormer Titan, cihaz‑içi inference için optimize edilmiş 2.64B tasarım hedefli edge‑native bir kodlama modelidir (mevcut build runtime toplamı: ~3.70B). 1.58‑bit kuantizasyon ve LiquidRouter MoE ile maliyet ve bellek kullanımını düşürmeyi hedefler.

## Problem
Bulut tabanlı AI pahalıdır ve gizlilik‑odaklı/low‑connectivity ortamlara uygun değildir. Kurumlar, cihaz‑içi kodlama copilotu çalıştırmakta zorlanır.

## Çözüm
Mobil‑first mimari ile cihaz‑içi kod üretimi; düşük gecikme, veri egemenliği ve maliyet verimliliği.

## Hedef Pazar
Regüle sektörler, savunma sınıfı iş akışları ve düşük bağlantılı bölgelerde yerel inference ihtiyacı olan kurumlar.

## Farklılaştırıcılar
- Donanım‑farkındalıklı mimari (post‑quantization değil).
- LiquidRouter ile uzman çökmesi riskini azaltan zamansal yönlendirme.
- Reproducibility gate’leri ve forensik loglama.

## Çekiş (Traction)
- Mimari ve eğitim pipeline uygulandı; tam eğitim koşusu ve benchmark kanıtları pending.
- Gate suite ve asset stack hazır.
- HumanEval/MBPP runner’ları hazır.

## İş Modeli
Kurumsal on‑device inference lisanslama ve özel dağıtımlar.

## Ekip
- Kurucu: Sistem odaklı mühendislik lideri.

## Finansman
- Mevcut finansman: Paylaşılmadı.
- Kredi kullanım alanı: Eğitim altyapısı, benchmark doğrulama ve dağıtım validasyonu.

## Riskler ve Önlemler
- Risk: Eğitim maliyeti ve yakınsama.
- Önlem: Failure budget ve telemetry‑driven execution.
