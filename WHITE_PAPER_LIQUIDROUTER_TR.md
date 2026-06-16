# Teknik Makale: LiquidRouter Mimarisi
**Seyrek Uzmanlar Karışımı İçin Zamansal Yönlendirme (Truth-Locked Build30 V2)**

## 1. Özet
MertFormer Titan içinde **LiquidRouter**, MoE kapılama hattında zamansal token yönlendirmesi için kullanılır. Mevcut implementasyonda LiquidRouter, `layers/moe.py` içinde **causal depthwise Conv1d + rolling state buffer** mekanizmasıdır. Bu yol, `layers/liquid.py` içindeki CfC (`LiquidMixer/LiquidCell`) yolundan ayrı değerlendirilmelidir.

## 2. Problem: Statik MoE Kararsızlığı
Geleneksel MoE yönlendiricileri bazı uzmanlara aşırı yük bindirebilir ve ardışık tokenlar arasında dengesiz uzman geçişleri üretebilir. Bu durum edge çalışma zamanında yönlendirme oynaklığını artırabilir.

## 3. Build30 V2 Mevcut Yönlendirme Mekanizması
LiquidRouter iki sinyali birleştirir:
- **Main path:** token-anlık projeksiyon ile uzman logitleri.
- **Fluid path:** kısa geçmiş penceresi (`history_window`) üzerinde causal depthwise Conv1d + runtime state.

Uzman seçimi MoE tarafında **token-choice top-k** olarak uygulanır.

### Matematiksel Özet (Implementasyon Uyumlu)
Token özellikleri `x_t` olsun.
- Main logits: `g_main(t) = W_main * x_t`
- Fluid logits: `g_fluid(t) = W_fluid * Conv1d_causal(x_{t-k+1:t})`
- Router çıktısı: `g(t) = g_main(t) + g_fluid(t)`
- Seçim: token-choice `top-k(g(t))`

**V2 notu:** MoE dispatch artık opsiyonel paralel gather/scatter yolunu destekler.

## 4. CfC Ayrımı (Kritik)
CfC dinamikleri mimaride vardır; ancak router çekirdeği değildir:
- **Router:** Conv1d + state buffer (`layers/moe.py`)
- **CfC yolu:** `LiquidMixer/LiquidCell` (`layers/liquid.py`)

Build30 V2 için bu ayrım bilinçli bir tasarım kararına karşılık gelir ve dış iletişimde net korunmalıdır.

## 5. Donanım Niyeti ve İddia Sınırı
- Hedef niyet: yönlendirme geçiş kararsızlığını azaltmak ve edge runtime davranışını iyileştirmek.
- Gecikme/enerji üstünlüğü iddiaları gerçek cihaz ölçümü olmadan yalnızca **hedef/tahmin** düzeyindedir.
- Bağımsız kanıt olmadan “dünyada ilk” veya “ampirik olarak en iyi” iddiası yapılmaz.

## 6. Sonuç
LiquidRouter, MertFormer Titan’ın seyrek MoE yığınında zamansal Conv yönlendirme bileşenidir. Build30 V2 bunu offline-first edge kısıtlarına uyumlu, implementasyon-gerçekliğinde ve claim-safe bir mekanizma olarak tanımlar.

## 7. Deney (Ölçülmüş Ablasyon) ve Sınırlar

**Claim modu: ölçülmüş (küçük ölçek, yönsel). Bu bir benchmark iddiası DEĞİLDİR.**
**Kanonik sonuçlar: [`ABLATION_TR.md`](ABLATION_TR.md).**

**Önemli kapsam ayrımı:** ablasyon `cfg.use_liquid` bayrağını — yani `layers/liquid.py` içindeki
**CfC LiquidMixer**'ı — açıp kapatır. Bu makalenin anlattığı **Conv1d LiquidRouter** (`layers/moe.py`)
ve MoE yığını iki kolda da AÇIK kalır; sinyal CfC mixer'ı ölçer, router'ı **doğrudan test etmez**.

**12-seed nihai (2026-06-15) — önceki tek-seed pilotu EZER.** 12 seed × AÇIK/KAPALI, small preset
(hidden 384, 8 katman), 2-basamaklı toplama, Kaggle T4, ~900 sn/koşu:
- Held-out ID exact-acc: KAPALI %96.32 vs AÇIK %94.69 (Δ −1.63 pp, p=0.305, d=−0.43, %95 GA [−4.63, +1.17]).
- OOD (3-basamak): %0 / %0 (taban — ayırt edici değil). AÇIK duvar-saatinde ~%30 daha az adım; +%2.8 param.
- **Hüküm:** gözle görülür doğruluk faydası yok, maliyet kesin (~%30 yavaş), ama görev tavan/taban
  doygun + test underpowered + iso-zaman → **Liquid'in ölçekteki değeri konusunda sonuçsuz, çürütülmedi.**
  45K'yı Liquid'e bağlamak için pozitif kanıt yok.

**Liquid hız/gecikme: İDDİA YOK** — doğrulanmış bir 45K koşusu ölçek-temsili veri üretene kadar
(pilot/H200 sayıları confounded). Bkz. `ABLATION_TR.md`.

**Sınırlar (dürüst):** küçük toy görev, T4, ~15 dk/seed — 3.67B model hakkında kanıt değil. Önceki
tek-seed pilot (Δ(off−on)=+0.50) büyük ölçüde tek şanslı seed'di ve ezildi. **Toy-ölçek veriyle arXiv
gönderimi yapılmaz — iskelet şimdi, gönderim ölçülmüş 45K koşusundan sonra.**

Kanıt (sha256-zincirli): [`reports/ablations/liquid_ablation_final_20260615/`](reports/ablations/liquid_ablation_final_20260615/)
(`final_summary.json`, `MANIFEST.json`, grafikler). 2026-06-14 tek-seed pilotu
(`reports/ablations/liquid_ablation_results.json`, `liquid_ablation_pilot_curve.png`) **superseded**
tarihçe olarak korunur.
