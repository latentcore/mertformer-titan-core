# Ablasyon Sonuclari

Durum: **Olcum bekliyor** (tam egitim donanimi gerekir).

Tabloyu nasil dolduracaginiz:
- Her ablation konfigurasyonunu calistirin (bkz: `ablations/*/README.md`).
- Final loss, konverjans notlari ve stabilite sorunlarini kaydedin.
- Egitim sonrasi benchmark farklarini (HumanEval/MBPP/GSM8K) ekleyin.

| Ablasyon | Hedef | Durum | Notlar |
| --- | --- | --- | --- |
| no_moe | Dense-only temel cizgiyi olc | Beklemede | Egitim donanimi gerekir |
| no_liquid | Liquid katmanlarinin etkisini olc | Beklemede | Egitim donanimi gerekir |
| dense_only | MoE + Liquid kapali | Beklemede | Egitim donanimi gerekir |
| bitlinear_off | BitNet kapali temel cizgi | Beklemede | Egitim donanimi gerekir |
