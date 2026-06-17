# Ablation: Liquid Kapalı

**Amaç**: Liquid katmanlarin stabilite ve routing üzerindeki etkisini ölçmek.

**Config değişikliği**:
- `use_liquid: false`

**Durum**: **12-seed çok-seed ablasyonu ile ezildi** (bkz. `ABLATION.md`): OFF %96.32 / ON %94.69 ID exact-accuracy, Δ−1.63 pp, p=0.305 — **ölçülen fayda yok, inconclusive** (~%30 daha yavaş). Aşağıdaki tek-seed $0 Kaggle pilotu (2026-06-14) büyük ölçüde tek şanslı seed'di; köken için tutuldu, iddia değil.

**Pilot sinyali (ölçülmüş, claim-safe)**:
- Kurulum: ~100M proxy, T4 x2, 500 adım, saf next-token CE; aynı veri + aynı init (seed 1234), sadece `use_liquid` değişiyor. (`use_liquid`, `layers/liquid.py` içindeki CfC LiquidMixer'ı (katlar [2,4,6]) açıp kapatır; MoE + Conv1d LiquidRouter iki kolda da AÇIK — bu, router'ı değil CfC mixer'ı ölçer.)
- Sonuç: `liquid ON mean_last10 = 11.489` vs `liquid OFF mean_last10 = 11.993`; `Δ(off−on) = +0.50` (tek-seed; **ezildi** — yukarıdaki 12-seed koşu ölçülen fayda göstermiyor).
- Sınır: oynak eğriler (sabit lr, warmup/decay yok), tek seed, küçük corpus (35.634 tok / 128k vocab). Bu yalnızca pilot sinyalidir; daha büyük ölçülmüş bir koşuya kadar benchmark iddiası değildir.
- Kanıt: `reports/ablations/liquid_ablation_results.json` (tam 500-adım eğriler) + `reports/ablations/liquid_ablation_pilot_curve.png` (grafik) + `reports/ablations/liquid_ablation_kaggle_20260614.json` (özet).
- Tam ölçekli ablasyon (45K, çok-seed) hâlâ eğitim donanımı gerektirir; orada `ablations/results.md`'ye kaydedin.
