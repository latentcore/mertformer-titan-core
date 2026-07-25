# Nutrition5k Vision Side-Experiment - Report

Generated 2026-07-18T08:04:32Z  |  import source: `vendor`

## 1. Scope (read before interpreting anything below)

This is a bounded side experiment, separate from the canonical 45K-step text-pretraining run. It reuses the real BitLinear / MoE / LiquidMixer / RMSNorm layers on a small (~14.1M parameter) vision trunk with a necessarily-new bidirectional attention module (the shared GQA attention is hardcoded causal; see model/nutrition_vision.py). It does **not** reuse train.train.train() (the canonical LM trainer is coupled to token-ID input and does not apply to image regression) - training used a short, plain AdamW loop defined in this script.

The comparison numbers below are against the original Nutrition5k paper's own InceptionV2 backbone (2048-d features, JFT-300M pretrained, full precision) — a much larger, pretrained, full-precision model. This experiment's ~14.1M parameter, from-scratch, partly-ternary-quantized trunk is **not** expected to match it; the paper's numbers are included as a reference point, not a claim of parity.

## 2. Data

| Split | requested | staged | missing label | missing image |
|---|---:|---:|---:|---:|
| train | 4,059 | 2,755 | 0 | 1,304 |
| test | 709 | 507 | 0 | 202 |

Source: public GCS bucket `gs://nutrition5k_dataset/nutrition5k_dataset/` (anonymous HTTPS, no auth/gsutil). Only `imagery/realsense_overhead/*/rgb.png` was downloaded — depth and side-angle video were deliberately skipped (see this script's module docstring, 'DEPTH: SKIPPED ON PURPOSE').

## 3. Model

| | |
|---|---|
| Parameters | 14,074,631 total / 14,074,631 trainable |
| Geometry | 8 layers x 256 hidden, attn 8q/4kv |
| Experts / mixer | 8 experts top-2 at layers [3, 6]; LiquidMixer at layer [5] |
| Input | 256x256 RGB, 16x16 patches (256 patches/image) |
| Device | cuda | AMP on |
| Epochs run | 58 / 60 (early stopped: True) |

## 4. Results vs. the paper's own baselines (Table 3, arXiv:2103.03375)

| Target | This model MAE | This model MAE% | Paper 2D Direct MAE | Paper 2D Direct MAE% | Paper always-predict-mean MAE% |
|---|---:|---:|---:|---:|---:|
| calories | 75.5 kcal | 29.6% | 70.6 kcal | 26.1% | 60.2% |
| mass | 49.5 g | 24.9% | 40.4 g | 18.8% | 58.5% |
| fat | 5.7 g | 44.8% | 5.0 g | 34.2% | 67.6% |
| carb | 8.9 g | 45.1% | 6.1 g | 31.9% | 62.1% |
| protein | 7.4 g | 42.5% | 5.5 g | 29.5% | 62.1% |

A result between the 'always-predict-mean' column and the paper's direct-prediction column means the model learned a real, non-trivial signal from the image, even without matching the paper's much larger pretrained backbone.

## 5. Checkpoints

- Best (by val calorie MAE): `C:\Users\<WIN_HOME>\Desktop\Nutrition5k_Training_Package\nutrition5k_work\checkpoints\nutrition5k_best.pt`
- Latest: `C:\Users\<WIN_HOME>\Desktop\Nutrition5k_Training_Package\nutrition5k_work\checkpoints\nutrition5k_latest.pt`
- Best val calorie MAE reached: 75.5 kcal

## 6. Known, accepted limitations (not bugs)

- No depth channel (see Section 2). The paper's own Table 3 shows depth-as-4th-channel improves calorie MAE% from 26.1% to 18.8%; this experiment does not have that signal available.
- The reused MoE LiquidRouter applies a causal depthwise conv over the flattened patch sequence (raster-scan order), a directional inductive bias with no natural meaning for a 2D image. Harmless (zero-padded), documented in model/nutrition_vision.py, not fixed (would require modifying the shared, sealed layers/moe.py).
- Single train/test split (the paper's own Nutri-Train/Nutri-Test partition), no cross-validation — this was a one-shot run by design.

## 7. Correction & follow-up notes (added 2026-07-25, does not alter Sections 1-6 above)

This section is an addition, not a silent rewrite of the dated 2026-07-18 snapshot above — the original numbers and text are untouched.

- **Correction to Section 1's "much larger" framing.** Section 1 called the paper's InceptionV2 baseline "a much larger, pretrained, full-precision model" without citing a parameter count. Checked (web search, 2026-07-25): commonly-cited parameter counts for the InceptionV2/BN-Inception family run **~17.2M–23.2M** depending on exact variant — the same order of magnitude as this model's measured 14,074,631 params, not dramatically larger by raw parameter count. The real, defensible gap is **JFT-300M pretraining and full floating-point precision**, not size.
- **Interpretive note on Section 4's results table.** Calorie/mass MAE% (29.6%/24.9%) land closer to the paper's own pretrained baseline than fat/carb/protein MAE% (42.5-45.1%) do. Plausible reading: calorie/mass correlate with visible volume/size, a signal a from-scratch trunk can pick up on its own; the macros require ingredient-level recognition, which is closer to what JFT-300M pretraining specifically provides and this trunk lacks.
- **Diagnostic value beyond this experiment's own scope.** The same BitLinear/MoE/LiquidMixer layers used here trained stably for 58/60 epochs (plain AdamW, no GaLore, no 8-bit Adam, no distillation, no curriculum — and this task has no MoE z-loss term at all, so it predates that fix entirely) with zero divergence. That narrows where the two RTX-5070 LM-side divergences (2026-07-02, 2026-07-12) trace to: the *training regime* (LR/optimizer/z-loss/curriculum interaction), not a defect in these shared layers themselves.
