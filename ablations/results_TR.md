# Ablasyon Sonuçları

Durum: **Ölçüm bekliyor** (tam eğitim donanımı gerekir).

Tabloyu nasıl dolduracağınız:
- Her ablation konfigürasyonunu çalıştırın (bkz: `ablations/*/README.md`).
- Final loss, konverjans notları ve stabilite sorunlarını kaydedin.
- Eğitim sonrası benchmark farklarını (HumanEval/MBPP/GSM8K) ekleyin.

| Ablasyon | Hedef | Durum | Notlar |
| --- | --- | --- | --- |
| no_moe | Dense-only temel çizgiyi ölç | Beklemede | Eğitim donanımı gerekir |
| no_liquid | CfC LiquidMixer'ın etkisini ölç (`use_liquid`, katlar [2,4,6]; iki kolda da MoE+router açık) | **12-seed verdict** (bkz. `ABLATION.md`) | 12-seed çok-seed: OFF %96.32 / ON %94.69 ID exact-acc, Δ−1.63 pp, p=0.305, inconclusive — **ölçülen fayda yok, ~%30 yavaş**. Önceki tek-seed Kaggle pilotu (2026-06-14, Δ(off−on)=+0.50, "yönsel fayda") büyük ölçüde tek şanslı bir seed'di ve **ezildi**. Pilot verisi `reports/ablations/liquid_ablation_results.json` + grafik `reports/ablations/liquid_ablation_pilot_curve.png` |
| dense_only | MoE + Liquid kapalı | Beklemede | Eğitim donanımı gerekir |
| bitlinear_off | BitNet kapalı temel çizgi | Beklemede | Eğitim donanımı gerekir |
