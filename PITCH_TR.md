# MertFormer Titan: Stratejik Özet (VC Brief)

> **Claim boundary:** Bu pitch stratejik arka plan dokümanıdır. Benchmark,
> production, deployment, AGI veya model üstünlüğü iddiası değildir. Performans
> ve mobil cihaz dili, checkpoint-bound kanıt oluşana kadar target/estimate
> statüsündedir.

## 💎 Temel Tezi
Kurumsal yapay zeka şu an bulut bağımlı bir maliyet (Opex) yüküdür. **MertFormer Titan**, bulut bağımlılığını, bulut inference maliyetini ve dış veri maruziyetini azaltmayı hedefleyen, 1.58-bit "Edge-Native" (Cihaz Özgü) bir mimari yönüdür ve son kullanıcı cihazlarında iddialı, düşük maliyetli kodlama modeli verimlilik hedefi sunar.

## ⚡ Ürün
Otonom yazılım geliştirme süreçleri için optimize edilmiş 2.64B tasarım hedefli kodlama motoru (mevcut build runtime toplamı: ~3.67B).
- **Hız**: Mobil NPU latency, fiziksel cihaz ölçümü yapılana kadar hedef profildir.
- **Verimlilik**: Üçlü (Ternary) BitLinear ağırlıkları büyük ağırlık belleği azaltımı hedefler; tam runtime bellek her export yolunda ölçülmelidir.

## 🏰 Stratejik Hendek (The Moat)
1. **Matematiksel Akış**: Tescilli `LiquidRouter` (causal Conv1d + state buffer) zamansal bağlamlı token yönlendirmesi sağlar; CfC dinamikleri `LiquidMixer/LiquidCell` içinde çalışır.
2. **Ekonomik Taban**: Kontrollü kurumsal dağıtımlar için daha düşük maliyetli özel inference hedefi.
3. **Founder Alpha**: Atipik yürütme hızına sahip, sistem odaklı kurucu tarafından inşa edilen yüksek alfalı mimari.

## 🚀 Vizyon
3B tasarım hedefi altında kanıt-temelli bir cihaz-içi kodlama modeli doğrulama hattı kurmak; yalnızca ölçülmüş kazananları üretim veya girişim ölçeğinde iş birliğine taşımak.

## V2 Refactor Note
Build 30 V2 adds dedup, parallel MoE dispatch, CfC fast path, and stricter training gates.
