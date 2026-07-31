# DURUM (STATUS) — MertFormer Titan

Kanonik, elle tutulan durum giriş noktası. Üretilen ayrıntı `reports/` altındadır
(reviewer'ın ilk okuması gereken tek kaynak budur). English: [STATUS.md](STATUS.md).

## Tek bakış
- **Aşama:** pre-training **closure-complete** — kanonik model **henüz eğitilmedi**.
- **Build:** `721 passed, 9 skipped` (offline-first `pytest`). Bkz. [REPRODUCE.md](REPRODUCE.md).
- **Readiness:** `train_allowed = true` · `decision_reason_code = READY_REMOTE_BOOTSTRAP` · `start_gate = START_ALLOWED`.
- **Crash-sınıfı bug:** yok (kanonik model + orchestrator sorunsuz import oluyor).

## Mimari (ölçülen, benchmark değil)
- 18 katman, hidden 2048, 16 head / 8 KV (**GQA**), 8 expert top-2 (her 3. katmanda MoE), [4,10,16] katmanlarında Liquid/CfC mixer, BitNet b1.58 ternary.
- **Ölçülen runtime param:** `3,672,982,022` (~3.67B). **Tasarım hedefi:** 2.64B. İki etiket de bilinçli — bkz. [reports/param_accounting_report.md](reports/param_accounting_report.md).

## Ölçülen vs ölçülmeyen
- **Ölçülen:** repo-içi self-test'ler, offline smoke harness, 12-seed Liquid ablation (bkz. [ABLATION_TR.md](ABLATION_TR.md) — hüküm: **ölçülen doğruluk faydası yok, ~%30 daha yavaş, toy ölçekte sonuçsuz; hız iddiası yok**).
- **Ölçülmeyen (tek gerçek boşluk):** kanonik 3.67B model hiç eğitilmedi — yani "öğreniyor mu / yakınsıyor mu / genelliyor mu" **doğrulanmadı**. Bu bir kod düzenlemesi değil, donanım bağımlılığıdır.

## Tek kalan engel: gerçek bir 45K GPU koşusu
- H100/H200 + compute + gün gerektirir. Yerel K4 drill'leri (checkpoint save→restore→resume) + import smoke yeşildir.
- Lane engelleri: `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`, `online_teacher:MISSING_HF_TOKEN`.
- Başlatma (hedef donanım): `bash zero_touch_start.sh`. Bkz. [REPRODUCE.md](REPRODUCE.md).
- **Ara-ölçek ÖLÇEKLEME koşusu YOK** (250M / 500M) — öncesi ya da sonrası; doğrudan kanonik 3.67B 45K (küçük modelin kendi dinamiği var, 45K'yı öngörmez). Korunan tek ön-koşu: kanonik mimarinin ücretsiz `TITAN_MAX_STEPS=2` import/K4 smoke'u (alt-ölçek model değil). Bkz. [DECISIONS_TR.md](DECISIONS_TR.md).
  - *BACKLOG_TR.md'nin 2026-07-02 "100–300M pilot" maddesiyle çelişmez:* o pilot (`config/model/mertformer_pilot_stabilization.yaml`, ölçülen 172.67M) **yalnızca kararlılık** (LR/ıraksama) güvenlik kontrolüdür; **kabiliyet iddiası ve ölçekleme ekstrapolasyonu içermez**. Mühürlü karar, ara-ölçek koşuyu *kabiliyet/ölçekleme vekili* olarak kullanmayı yasaklar; aday bir öğrenme oranının grad_norm'u sonlu tuttuğunu 45K'ya $150–570 ve günlerce GPU harcamadan önce bir saatte kanıtlamayı yasaklamaz.

## Kanonik yüzeyler
- [TRUTH_MATRIX_TR.md](TRUTH_MATRIX_TR.md) — iddia → kanıt. · [BACKLOG_TR.md](BACKLOG_TR.md) — ertelenen iş.
- [GOVERNANCE_TR.md](GOVERNANCE_TR.md) — politika/kontrat indeksi. · [REPRODUCE.md](REPRODUCE.md) — doğrulama & başlatma.
- [DECISIONS_TR.md](DECISIONS_TR.md) — bilinçli kararlar (değiştirilmeyip-belgelenen bulgular dahil).
- Üretilen ayrıntı: `reports/closure_57_matrix.md`, `reports/repo_closure_scorecard.md`, `reports/final_truth_matrix.md`.

## 45K öncesi kararlılık pass'i — 2026-07-08 (aday düzeltmeler, DOĞRULANMADI)
2026-07-02 laptop-preflight koşu geri-bildirimi artık **kodda uygulandı**: LR rejimi (`1.5e-3` → `3e-4`, router ×1.5 farkı kaldırıldı, warmup 0.10 → 0.15, üçü de env ile taranır), EMA-göreli Liquid spike eşiği, `generate()` Liquid-state taşıma + parite testi, checkpoint-bağlı held-out perplexity harness'ı (`eval/held_out_ppl.py`), ölçülen 172.67M kararlılık pilot config'i, genel loss-ıraksama devre kesici ve WSD-scheduler cosine clamp'i.

**Bunlar gerçek bir GPU koşusunda yeniden doğrulanmayı bekleyen aday düzeltmelerdir.** Temiz bir RTX-5070 yeniden-koşusu söyleyene kadar ıraksama sorunu çözülmüş *değildir*. Yeni ölçülen-kabiliyet iddiası eklenmedi; readiness, iddia sınırları ve dondurulmuş mimari değişmedi. Tam kayıt için bkz. [BACKLOG_TR.md](BACKLOG_TR.md) ve [DECISIONS_TR.md](DECISIONS_TR.md) — ön-onaylı **olmayan** tek ekleme (ıraksama guard'ı) dahil.
