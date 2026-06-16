# DURUM (STATUS) — MertFormer Titan

Kanonik, elle tutulan durum giriş noktası. Üretilen ayrıntı `reports/` altındadır
(reviewer'ın ilk okuması gereken tek kaynak budur). English: [STATUS.md](STATUS.md).

## Tek bakış
- **Aşama:** pre-training **closure-complete** — kanonik model **henüz eğitilmedi**.
- **Build:** `370 passed, 4 skipped` (offline-first `pytest`). Bkz. [REPRODUCE.md](REPRODUCE.md).
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
- **Ara-ölçek koşu YOK** (250M / 500M) — öncesi ya da sonrası; doğrudan kanonik 3.67B 45K (küçük modelin kendi dinamiği var, 45K'yı öngörmez). Korunan tek ön-koşu: kanonik mimarinin ücretsiz `TITAN_MAX_STEPS=2` import/K4 smoke'u (alt-ölçek model değil). Bkz. [DECISIONS_TR.md](DECISIONS_TR.md).

## Kanonik yüzeyler
- [TRUTH_MATRIX_TR.md](TRUTH_MATRIX_TR.md) — iddia → kanıt. · [BACKLOG_TR.md](BACKLOG_TR.md) — ertelenen iş.
- [GOVERNANCE_TR.md](GOVERNANCE_TR.md) — politika/kontrat indeksi. · [REPRODUCE.md](REPRODUCE.md) — doğrulama & başlatma.
- [DECISIONS_TR.md](DECISIONS_TR.md) — bilinçli kararlar (değiştirilmeyip-belgelenen bulgular dahil).
- Üretilen ayrıntı: `reports/closure_57_matrix.md`, `reports/repo_closure_scorecard.md`, `reports/final_truth_matrix.md`.
