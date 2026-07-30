# GERÇEK MATRİSİ (TRUTH MATRIX) — iddia → kanıt

Yük taşıyan her iddia ve kanıt sınıfı. English: [TRUTH_MATRIX.md](TRUTH_MATRIX.md).
Kanıt sınıfları: **measured** (koşuldu, tekrarlanabilir) · **target** (tasarım hedefi, henüz ölçülmedi) · **vision** (araştırma yönü).

| İddia | Sınıf | Kanıt |
|---|---|---|
| Mimari: 18L / 2048 / GQA(16:8) / MoE 8-top2 / Liquid[4,10,16] / BitNet b1.58 | measured | kod: `config/config.py`, `model/transformers.py`, `layers/` |
| Ölçülen runtime param = 3,672,982,022 (~3.67B) | measured | `reports/param_accounting_report.md`, `reports/FACTS.json` |
| Tasarım hedefi param = 2.64B | target | `economics/flops_estimator.py` `DEFAULT_PARAMS` |
| Test suite: 726 passed, 5 skipped (offline) | measured | `pytest` — bkz. [REPRODUCE.md](REPRODUCE.md) |
| Checkpoint save→restore→resume bütünlüğü (K4) | measured (yerel, toy ölçek) | `scripts/checkpoint_restore_drill.py`, `resume_compat_check.py`; **45K ölçeğinde henüz kanıtlanmadı** |
| CfC/MoE fast-path sayısal parite (≤%1) | measured (toy ölçek) | `scripts/cfc_moe_tolerance_check.py` + rapor |
| Liquid katmanları doğruluğu artırır | **DESTEKLENMİYOR** | [ABLATION_TR.md](ABLATION_TR.md): OFF %96.32 / ON %94.69, Δ−1.63pp, p=0.305, d=−0.43 — sonuçsuz; maliyet (~%30 yavaş) kesin |
| Liquid hız / gecikme avantajı | **iddia yok** | [ABLATION_TR.md](ABLATION_TR.md) §"İDDİA YOK" — tüm hız sayıları doğrulanmış 45K koşusuna kadar confounded |
| GPT-3.5-sınıfı yetenek / cihaz-içi throughput / NPU gecikmesi | target/vision | ölçülmedi — 45K koşusu + cihaz profili gerektirir |
| Kanonik 3.67B model eğitildi & yakınsıyor | **doğrulanmadı** | hiç eğitilmedi; gerçek checkpoint yok — bkz. [STATUS_TR.md](STATUS_TR.md) |
| Değiştirilmeyip-belgelenen bulgular (z-loss 2e-6, dt=1.0, GPU-perf) | measured (mekanizma) | [DECISIONS_TR.md](DECISIONS_TR.md) — 45K'yı confound etmemek için kasıtlı ertelendi |

Üretilen çapraz referans: `reports/final_truth_matrix.md`, `reports/source_of_truth_map.md`.
