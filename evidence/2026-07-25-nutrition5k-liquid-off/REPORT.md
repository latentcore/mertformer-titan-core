# Nutrition5k Vision Side-Experiment - Report

Generated 2026-07-25T10:02:36Z  |  import source: `repo`

## 1. Scope (read before interpreting anything below)

This is a bounded side experiment, separate from the canonical 45K-step text-pretraining run. It reuses the real BitLinear / MoE / LiquidMixer / RMSNorm layers on a small (~13.8M parameter) vision trunk with a necessarily-new bidirectional attention module (the shared GQA attention is hardcoded causal; see model/nutrition_vision.py). It does **not** reuse train.train.train() (the canonical LM trainer is coupled to token-ID input and does not apply to image regression) - training used a short, plain AdamW loop defined in this script.

The comparison numbers below are against the original Nutrition5k paper's own InceptionV2 backbone (2048-d features, JFT-300M pretrained, full precision) — a much larger, pretrained, full-precision model. This experiment's ~13.8M parameter, from-scratch, partly-ternary-quantized trunk is **not** expected to match it; the paper's numbers are included as a reference point, not a claim of parity.

## 2. Data

| Split | requested | staged | missing label | missing image |
|---|---:|---:|---:|---:|
| train | 4,059 | 2,755 | 0 | 1,304 |
| test | 709 | 507 | 0 | 202 |

Source: public GCS bucket `gs://nutrition5k_dataset/nutrition5k_dataset/` (anonymous HTTPS, no auth/gsutil). Only `imagery/realsense_overhead/*/rgb.png` was downloaded — depth and side-angle video were deliberately skipped (see this script's module docstring, 'DEPTH: SKIPPED ON PURPOSE').

## 3. Model

| | |
|---|---|
| Parameters | 13,811,719 total / 13,811,719 trainable |
| Geometry | 8 layers x 256 hidden, attn 8q/4kv |
| Experts / mixer | 8 experts top-2 at layers [3, 6]; LiquidMixer at layer [] |
| Input | 256x256 RGB, 16x16 patches (256 patches/image) |
| Device | cuda | AMP on |
| Epochs run | 46 / 60 (early stopped: True) |

## 4. Results vs. the paper's own baselines (Table 3, arXiv:2103.03375)

| Target | This model MAE | This model MAE% | Paper 2D Direct MAE | Paper 2D Direct MAE% | Paper always-predict-mean MAE% |
|---|---:|---:|---:|---:|---:|
| calories | 73.4 kcal | 28.7% | 70.6 kcal | 26.1% | 60.2% |
| mass | 50.3 g | 25.4% | 40.4 g | 18.8% | 58.5% |
| fat | 5.8 g | 44.9% | 5.0 g | 34.2% | 67.6% |
| carb | 8.8 g | 44.9% | 6.1 g | 31.9% | 62.1% |
| protein | 7.5 g | 42.8% | 5.5 g | 29.5% | 62.1% |

A result between the 'always-predict-mean' column and the paper's direct-prediction column means the model learned a real, non-trivial signal from the image, even without matching the paper's much larger pretrained backbone.

## 5. Checkpoints

- Best (by val calorie MAE): `/kaggle/working/mertformer_batch/orchestrator/mertformer_batch_output/01_nutrition5k_liquid_off/repo/scripts/nutrition5k_work/checkpoints/nutrition5k_best.pt`
- Latest: `/kaggle/working/mertformer_batch/orchestrator/mertformer_batch_output/01_nutrition5k_liquid_off/repo/scripts/nutrition5k_work/checkpoints/nutrition5k_latest.pt`
- Best val calorie MAE reached: 73.4 kcal

## 6. Known, accepted limitations (not bugs)

- No depth channel (see Section 2). The paper's own Table 3 shows depth-as-4th-channel improves calorie MAE% from 26.1% to 18.8%; this experiment does not have that signal available.
- The reused MoE LiquidRouter applies a causal depthwise conv over the flattened patch sequence (raster-scan order), a directional inductive bias with no natural meaning for a 2D image. Harmless (zero-padded), documented in model/nutrition_vision.py, not fixed (would require modifying the shared, sealed layers/moe.py).
- Single train/test split (the paper's own Nutri-Train/Nutri-Test partition), no cross-validation — this was a one-shot run by design.
