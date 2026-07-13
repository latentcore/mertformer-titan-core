# Ablasyonlar — Kanonik Sonuçlar

> Dil: [English](ABLATION.md) | **Türkçe**
> İddia modu: `measured` (küçük ölçek, yönsel). Bu, 3.67B model hakkında bir **benchmark iddiası
> DEĞİLDİR.** Kanonik 3.67B koşusu henüz yapılmadı.

MertFormer Titan ablasyon sonuçları için tek kanonik yüzey. Ham çıktılar ve SHA256 manifesti
[`reports/ablations/liquid_ablation_final_20260615/`](reports/ablations/liquid_ablation_final_20260615/)
altındadır.

## LiquidRouter / CfC mixer — 12-seed ablasyon (NİHAİ, 2026-06-15)

**Kurulum (toy ölçek):** `use_liquid` AÇIK vs KAPALI, 12 seed × 2 kol = 24 koşu. Small preset
(hidden 384, 8 katman, 6/2 head, 8 expert top-2, `liquid_idx=[1,4]`, vocab 14), 2-basamaklı
toplama, Kaggle T4, ~900 sn/koşu, AMP kapalı. Held-out ID = dağılım-içi (2-basamak); OOD =
3-basamak büyüklük genellemesi. Doğruluk kaynağı: `final_summary.json` + `MANIFEST.json`
(sha256-zincirli).

**Sonuç (primary veri):**

| Metrik | Liquid KAPALI | Liquid AÇIK | Δ (AÇIK−KAPALI) | istatistik |
|---|---|---|---|---|
| Held-out ID exact-acc | **%96.32** | **%94.69** | −1.63 pp | p=0.305, Cohen's d=−0.43, %95 GA [−4.63, +1.17] |
| OOD (3-basamak) exact-acc | %0.0 | %0.0 | +0.00 pp | p=nan (ikisi de tabanda) |
| Final train loss | 1.2689 | 1.2704 | +0.0015 | AÇIK bir tık kötü |
| Adım (eşit duvar-saati) | ~6056 | ~4363 | ×0.72 | AÇIK ~%30 daha az adım |
| Parametre | 42.10M | 43.28M | +1.18M (+%2.8) | küçük param maliyeti |

`decoupling_detected = false`. Seed varyansı yüksek (örn. AÇIK: seed0 = %100, seed1 = %83.8).

### Bu deney neyi söyleyebilir, neyi söyleyemez (ölçüm geçerliliği)
- **Dinamik aralık çökük.** ID tavanda (~%95 her iki kol), OOD tabanda (%0 her iki kol). Küçük bir
  mimari etki ancak ortada görünür; bu görevde o orta bant yok. Yani `OOD = 0/0`, Liquid lehine ya
  da aleyhine **kanıt değildir** (floor effect ayırt edemez).
- **Kıyas iso-zaman, iso-adım değil.** KAPALI aynı saniyede ~%40 fazla adım attı; "Liquid katkısız"
  *bilimsel* iddiası iso-adım/iso-token isterdi; burada yalnız *dağıtım* iddiası ("sabit zaman
  bütçesinde Liquid kaybeder") destekleniyor. Koşucu bu uyarıyı kendisi basıyor (COMPUTE ASYMMETRY).
- **Underpowered.** Bu seed varyansıyla 12 seed küçük-ama-gerçek bir etkiyi yakalayamaz; `p=0.305`
  "reddedilemedi" demek, "sıfır kanıtlandı" değil. Minimum saptanabilir etki hesaplanmadı.
- **Pilot inancının mekanizması.** Önceki tek-seed pilot sinyali (Δ(off−on)=+0.50) büyük ölçüde tek
  şanslı seed'di (seed0 AÇIK = %100); 12-seed ortalaması %94.69.

### Hüküm
Bu toy ölçekte, sabit zaman bütçesinde, **Liquid gözle görülür bir doğruluk faydası göstermedi ve
duvar-saatinde ~%30 daha yavaştı**, küçük (+%2.8) param maliyetiyle. Ama görev tavan/taban doygun,
test underpowered ve kıyas iso-zaman olduğu için dürüst okuma şu: **Liquid'in ölçekteki değeri
konusunda sonuçsuz — maliyet kesin, fayda ölçülmedi (çürütülmedi).** Liquid'i 45K'da tutmak ayrı bir
karar; bu deney onun lehine pozitif kanıt sağlamıyor.

### Liquid hız/gecikme: İDDİA YOK
Bu repo, doğrulanmış bir 45K koşusu gerçek, ölçek-temsili veri üretene kadar **Liquid için hiçbir
mutlak hız/gecikme iddiası yapmaz.** Pilot ve H200 sayıları confounded'dır (torch.compile warmup,
çalışma sırası, T4 fast-path yok). Tek kontrollü gözlem yönseldir: toy ölçekte, T4'te, iso-zaman,
Liquid AÇIK daha yavaştı — üretim iddiası değildir.

## Diğer ablasyonlar (beklemede — eğitim donanımı gerekir)
`ablations/`, `no_moe`, `dense_only`, `bitlinear_off` için iskeletler tutar. Bunlar gerçek GPU
eğitimi ister ve koşulmadı. Bileşen değeri, ölçekte ölçülene kadar hipotezdir.

`layers/moe.py` içinde varsayılan kapalı iki bileşen daha var (`use_structural_plasticity`,
`use_cross_expert_sync_bus`) ve yukarıdakiler gibi hiç ablasyona girmedi: `structural_plasticity`
(kullanım EMA'sına göre periyodik uzman budama/büyütme) ve `cross_expert_sync_bus` (opsiyonel,
attention'dan bağımsız çapraz-uzman koordinasyon sinyali). İkisinin de `ablations/` altında henüz
iskeleti yok, ikisi de kanonik eğitim yolunda çalışmıyor. Liquid'in 12-seed ablasyonundan önceki
disiplinle aynı: kodda var olması değer kanıtı değildir. 2026-07-13'te işaretlendi; iskelet/koşu
planlı değil — eğitim donanımı bu soruya harcanabilir hale gelince eklenir.
