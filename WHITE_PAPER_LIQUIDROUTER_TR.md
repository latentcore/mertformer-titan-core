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

## 7. Deney (Pilot Sinyali) ve Sınırlar

**Claim modu: ölçülmüş (yalnız pilot sinyali). Bu bir benchmark iddiası DEĞİLDİR.**

**Önemli kapsam ayrımı:** Aşağıdaki $0 pilot, `cfg.use_liquid` bayrağını — yani `layers/liquid.py`
içindeki **CfC LiquidMixer**'ı (katlar [2,4,6]) — açıp kapatır. Bu makalenin anlattığı **Conv1d
LiquidRouter** (`layers/moe.py`) ve MoE yığını, iki kolda da AÇIK kalır. Dolayısıyla bu sinyal CfC
mixer'ı ölçer; router'ı doğrudan test etmez.

- Kurulum: ~100M proxy MertFormer, saf next-token CE (teacher/KD yok), Kaggle T4×2, $0.
- Aynı veri + aynı init (seed 1234); tek fark `use_liquid`. 500 adım, seq 256, batch 8.
- Sonuç: liquid ON mean_last10 = 11.489 vs OFF = 11.993 → **Δ(off−on) = +0.50** (CfC mixer yön olarak
  yardım ediyor; daha düşük loss).

**Sınırlar (dürüst):** Tek seed; fark (0.50) sabit-lr/warmup'sız eğrilerin adım-adım gürültüsü
içindedir → ölçülmüş bir etki büyüklüğü değil, yön sinyalidir. Küçük corpus (35.634 token / 128k vocab);
end-loss rastgele tabana (ln 128000 ≈ 11.76) yakın seyreder. Bu, ablation hattının çalıştığını ve bir
yön gösterdiğini kanıtlar; eğitilmiş yetenek, benchmark veya modelin "çalıştığı" anlamına GELMEZ. Bunun
için daha büyük, çok-seed, ölçülmüş bir koşu gerekir (45K mimari-doğrulama koşusu). **Tek seedle arXiv
gönderimi yapılmaz — iskelet şimdi, gönderim ölçülmüş 45K koşusundan sonra.**

Kanıt: `reports/ablations/liquid_ablation_results.json` (tam 500-adım eğriler),
`reports/ablations/liquid_ablation_pilot_curve.png` (grafik),
`reports/outreach/liquid_ablation_pilot_note_2026-06-15.md` (kamuya açık not).
