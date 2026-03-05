# Ablasyon Sonuçları

Durum: **Ölçüm bekliyor** (tam eğitim donanımı gerekir).

Tabloyu nasıl dolduracaginiz:
- Her ablation konfigürasyonunu çalıştirin (bkz: `ablations/*/README.md`).
- Final loss, konverjans notları ve stabilite sorunlarını kaydedin.
- Eğitim sonrası benchmark farklarini (HumanEval/MBPP/GSM8K) ekleyin.

| Ablasyon | Hedef | Durum | Notlar |
| --- | --- | --- | --- |
| no_moe | Dense-only temel çizgiyi olc | Beklemede | Eğitim donanımı gerekir |
| no_liquid | Liquid katmanlarinin etkisini olc | Beklemede | Eğitim donanımı gerekir |
| dense_only | MoE + Liquid kapalı | Beklemede | Eğitim donanımı gerekir |
| bitlinear_off | BitNet kapalı temel çizgi | Beklemede | Eğitim donanımı gerekir |
