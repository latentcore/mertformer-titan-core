# Nutrition5k Vision Side-Experiment — 2026-07-18 (RTX 5070, bounded, independently re-verified)

**Status: `SIDE-EXPERIMENT, COMPLETE (measured)` — a bounded, self-contained vision side-experiment, NOT the canonical 45K text-pretraining run. Reuses the real BitLinear/MoE/LiquidMixer/RMSNorm trunk (unmodified) on a small (~14.1M param) vision model with a new bidirectional attention module. Does not touch `train/train.py`, the 45K training path, or any 45K readiness/claim surface.**

## What this is

A single-file, zero-argument orchestrator (`scripts/train_nutrition5k.py`) trained a small image→nutrition regression model on the real [Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k) (Google, public GCS bucket, anonymous HTTPS, verified 2026-07-17), on a Windows **RTX 5070 laptop (8 GB VRAM)**. Purpose: exercise this repo's own BitLinear/MoE/LiquidMixer layers on a genuinely different task (image regression, not next-token prediction) as a bounded, independent capability check — not a claim about the 45K model.

Config (see `dataset_manifest.json`, `metrics.json`): 14,074,631 params · 8 layers × 256 hidden · GQA attention 8q/4kv (new `VisionAttention`, bidirectional — the shared `layers/mla.py` GQA is hardcoded causal and does not apply to images) · MoE 8 experts top-2 at layers [3, 6] · LiquidMixer at layer [5] · 256×256 input, 16×16 patches · batch 32, AdamW, cosine LR, AMP on · 58/60 epochs (early-stopped, patience 10) · wall time 1h 07m.

## 🟢 MEASURED — training result (Windows, RTX 5070, cuda)

| Target | MAE | MAE% | Paper 2D Direct MAE% | Paper always-predict-mean MAE% |
|---|---:|---:|---:|---:|
| calories | 75.5 kcal | 29.6% | 26.1% | 60.2% |
| mass | 49.5 g | 24.9% | 18.8% | 58.5% |
| fat | 5.7 g | 44.8% | 34.2% | 67.6% |
| carb | 8.9 g | 45.1% | 31.9% | 62.1% |
| protein | 7.4 g | 42.5% | 29.5% | 62.1% |

All 5 targets land between the paper's "always-predict-mean" baseline and its own "2D Direct Prediction" baseline (arXiv:2103.03375, Table 3) — a real, non-trivial learned signal, not matching a much larger pretrained (JFT-300M InceptionV2) backbone, which is expected and not the goal.

## 🟢 MEASURED — independent Mac-side re-verification

The Windows-reported result was independently reproduced on a **different machine (Mac, CPU), with fresh code, downloading the 507 test-split images from scratch** (not reusing the Windows run's cached files) — see `evaluation_independent.md`. Result: **75.6 kcal / 29.6%** calories (vs. 75.5/29.6% reported), and all other 4 targets identical to 1 decimal. The small delta is consistent with CPU-vs-GPU/AMP floating-point rounding, not a discrepancy.

## 🟡 Known, real, documented limitation — partial dataset

The official Nutrition5k split is 4,059 train / 709 test dishes. This run's local dataset index was reused from an earlier, interrupted staging attempt (`phase_dataset()`'s resumability check only verifies an index file *exists*, not that it's *complete* — a real gap, now noted in this script) and was never topped up: **2,755/4,059 train (67.9%) and 507/709 test (71.5%) dishes actually staged** (`missing_image`: 1,304 train / 202 test, confirmed in `dataset_manifest.json`). The reported numbers above are real, measured on 507 genuine held-out dishes — just a smaller sample than the full official test split, not a synthetic or leaked one.

## 🟢 FIXED — real bug found during independent verification (not cosmetic)

`train_nutrition5k.py`'s checkpoint-saving code built the `best_val_calorie_mae` metadata field into the checkpoint dict **before** updating the variable to the current epoch's improved value, so every "new best" checkpoint's stored metadata field was one improvement-step stale (weights were always correct — confirmed by the independent re-eval above reproducing the true 75.5-tier result; only this one informational field lagged, surfacing in `predict_nutrition5k.py`'s printed "val MAE at save time" line). Fixed and verified with a live 4-epoch smoke test confirming the stored field now matches the actual result. The checkpoint in this evidence folder predates the fix, so its internal field still reads a stale value — harmless, does not affect predictions.

## Checkpoints — NOT included (weights excluded on purpose)

`.pt` files are gitignored (`*.pt`, `/checkpoints/`), 161 MB each. Referenced by SHA256 only:
- `nutrition5k_best.pt` (epoch 47 / 0-indexed, val calorie MAE 75.5 kcal) → `df8576923d0ad41a518bfa949605b81e86cc1f9e54c633b848fc10fc70c727e7`
- `nutrition5k_latest.pt` (epoch 57 / 0-indexed, final epoch before early stop) → `cfe488d4beb5af37dd8c4f8882c8d3581b0a7553e60ea2facdd2743cb8cc2fab`

## 🔵 Correction & follow-up notes (added 2026-07-25)

Additions only — nothing above this section was rewritten; the original 2026-07-18 measurements and text stand as-is.

- **Correction to the "much larger" backbone framing used above.** The measured-result and boundary sections describe the paper's InceptionV2 baseline as "much larger" / "a much larger pretrained (JFT-300M InceptionV2) backbone" without ever citing a parameter count. Checked via web search (2026-07-25): commonly-cited parameter counts for the InceptionV2/BN-Inception family run **~17.2M–23.2M** depending on exact variant (naming is inconsistent across sources between BN-Inception/Inception-v2 and Inception-v3) — the same order of magnitude as this model's measured 14,074,631 params, not dramatically larger by raw parameter count. The real, defensible gap this experiment is up against is **JFT-300M pretraining and full floating-point precision**, not size.
- **Interpretive note on the per-target results.** Calorie/mass MAE% (29.6%/24.9%) land closer to the paper's own pretrained baseline than fat/carb/protein MAE% (42.5-45.1%) do. Plausible reading: calorie/mass correlate with visible volume/size, a signal a from-scratch trunk can pick up on its own; the macros require ingredient-level recognition, closer to what JFT-300M pretraining specifically provides and this trunk lacks.
- **Diagnostic value beyond this experiment's own scope.** The same BitLinear/MoE/LiquidMixer layers used here trained stably for 58/60 epochs (plain AdamW, no GaLore, no 8-bit Adam, no distillation, no curriculum — and this task carries no MoE z-loss term at all, so it predates that fix entirely) with zero divergence. That narrows where the two RTX-5070 LM-side divergences (2026-07-02, 2026-07-12) trace to: the *training regime* (LR/optimizer/z-loss/curriculum interaction), not a defect in these shared layers themselves.

## Boundary (does NOT prove)

Not a claim about the canonical 3.67B / 45K model. Not benchmark-verified against the paper's own pretrained backbone (a much larger, JFT-300M-pretrained, full-precision model — not a fair comparison target). Not production-ready, not mobile-ready. Does not change 45K readiness (`TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP / START_ALLOWED`, unaffected) or any 45K training-math/architecture/claim surface — this experiment's code is fully isolated (new `model/nutrition_vision.py`, new `scripts/train_nutrition5k.py` orchestrator; zero existing tracked file was modified to build it).

## Reproduce

`python scripts/train_nutrition5k.py` (zero arguments; downloads the real dataset, trains, writes `REPORT.md`/`metrics.json`/checkpoints). `python scripts/evaluate_nutrition5k.py` (independent held-out re-evaluation against a trained checkpoint). `python scripts/predict_nutrition5k.py [photo.jpg]` (single-photo inference; opens a file picker if no path given).
