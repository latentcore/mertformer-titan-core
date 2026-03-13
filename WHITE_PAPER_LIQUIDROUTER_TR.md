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
