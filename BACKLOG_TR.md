# BACKLOG — ertelenen iş

Kanonik backlog giriş noktası. English: [BACKLOG.md](BACKLOG.md). Tohum ayrıntı: [V2_BACKLOG_SEED.md](V2_BACKLOG_SEED.md).

## Sert engel (kod düzenlemesi değil)
- **45K GPU koşusu (K1)** — kanonik 3.67B modeli H100/H200'de eğit; ilk gerçek loss eğrisini, checkpoint'i, model card'ı üret. "target/vision" iddialarını "measured"a çeviren tek şey budur. **Ara 250M/500M koşusu YOK** — doğrudan 45K (bkz. [DECISIONS_TR.md](DECISIONS_TR.md)). Bkz. [STATUS_TR.md](STATUS_TR.md).

## 45K sonrası — doğrulanmış bulgular, koşu öncesi kasıtlı DEĞİŞTİRİLMEDİ (koşu confound olmasın)
Her biri mekanizmasıyla [DECISIONS_TR.md](DECISIONS_TR.md)'de belgeli:
- **z-loss efektif ağırlığı ≈ 2e-6** (1e-4 × 0.02 çift-çarpım, niyetin ~50× altı). İstenen efektif ağırlığı seç ve bir kez uygula; `cfc_moe_tolerance_check.py` ile yeniden doğrula.
- **Liquid `dt` 1.0 sabit** → kanonik CfC sürekli-zaman değil, gated RNN. Ya değişken `dt` bağla ve ablate et, ya yeniden etiketle.
- **Yalnız GPU-perf:** sequential MoE dispatch `capacity_mask` atlıyor (çıktı doğru, boşa FLOP); micro-step başına `.item()` host-device senkronları. Sadece gerçek GPU'da ölçülür.
- **`liquid_warmup_steps`** env-override yok (hardcoded 10000) — diğer tunable'larla parite için ekle.
- **`mark_weights_updated()`** çağrılmayan cache-invalidation kancası — eval cache'in buna ihtiyacı var mı doğrula (körlemesine silme).

## Kozmetik / housekeeping (güvenli, davranışsal değil)
- `mark_weights_updated` ölü-metot incelemesi; `_compute_weight_version` private torch `._version`'a dayanıyor (sürümler arası guard); RoPE cache lazy-grow yorum netliği; `iter_packed_sequences` greedy-buffer notu.
- Tam liste için `V2_BACKLOG_SEED.md` Track A–F (compile policy, distributed contract, optimizer matrix, vb.).

## Kapsam dışı (belgeli, repo'da takip edilmiyor)
- `reports/closure_57_matrix.md`'deki AGI/ASI yetenek satırları **kapsam-dışı pending**'dir (benchmark / uzun-vadeli kanıt gerektirir) — bkz. [INTERNAL_AGI_GAP.md](INTERNAL_AGI_GAP.md).
- `orchestrator/` cognitive runtime + flag-off katmanlar (`cognitive_extensions.py`, `world_model_head.py`, `lifelong_safety.py`, `qinn.py`) kanonik 45K yolunda **inert** — kapsam-dışı, eğitilen modele dahil değil. Belgeli (silinmedi); bkz. [ARCHITECTURE.md](ARCHITECTURE.md) "out-of-scope surfaces".

## 45K öncesi — laptop preflight koşu geri-bildirimi (2026-07-02, gerçek koşu sinyali)
Kaynak: `evidence/2026-07-02-laptop-preflight/` (RTX 5070, 8 GB, commit `5fc5068`). Preflight altyapıyı doğruladı (step 500'de atomik checkpoint; guard'lar canlı; step 981'de graceful kesme-kaydı) VE kesin bir negatif eğitim-dinamiği bulgusu üretti: koşu **ıraksadı** (loss ~10.4 → ~15.0; grad_norm → `inf`, `1e11`–`1e16` bandında, yalnız `clip=2.0` yaşattı; MoE entropi 0.99 → 0.74). Bunlar 45K öncesi kararlılık kalemleridir — **koşu-geri-bildirimi olarak belgelenir; her biri gerçek bir koşuda uygulanıp yeniden doğrulanana kadar frozen eğitim yolu DEĞİŞMEZ:**
- **LR rejimi:** `1.5e-3` bu mimaride bu ölçekte ampirik olarak ölümcül → `3e-4`'ten tara, router ×1.5 LR çarpanını kaldır, warmup uzat; hedef clip-hit `< %5` ("clip'e rağmen yaşamak" değil).
- **Liquid spike eşiği:** mutlak `loss>5.0` ölçek-körü (bu loss aralığında hiç serbest bırakmaz → Liquid katmanı fiilen eğitilmez). **Göreli** yap (`EMA×1.5`) + cooldown.
- **`generate()` Liquid state:** Liquid gizli durumunu decode boyunca taşı (router state gibi) + full-forward↔incremental-decode **parite testi** ekle.
- **Held-out perplexity harness:** sabit korpus + sabit seed + sabit script (bugünkü eval bir koşunun kendi loss eğrisi dışında bir şey öğrenip öğrenmediğini söyleyemiyor).
- Gerçek 45K'dan önce **100–300M pilot** (kapı: clip-hit `<%5`, kalıcı Liquid freeze yok, MoE collapse yok, held-out ppl monoton).
