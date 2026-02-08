# Ablasyon Sonuclari

Durum: **Ölçüm bekliyor** (tam eğitim donanımı gerekir).

Tabloyu nasil dolduracaginiz:
- Her ablation konfigürasyonunu çalıştirin (bkz: `ablations/*/README.md`).
- Final loss, konverjans notlari ve stabilite sorunlarini kaydedin.
- Eğitim sonrasi benchmark farklarini (HumanEval/MBPP/GSM8K) ekleyin.

| Ablasyon | Hedef | Durum | Notlar |
| --- | --- | --- | --- |
| no_moe | Dense-only temel cizgiyi olc | Beklemede | Eğitim donanımı gerekir |
| no_liquid | Liquid katmanlarinin etkisini olc | Beklemede | Eğitim donanımı gerekir |
| dense_only | MoE + Liquid kapali | Beklemede | Eğitim donanımı gerekir |
| bitlinear_off | BitNet kapali temel cizgi | Beklemede | Eğitim donanımı gerekir |
