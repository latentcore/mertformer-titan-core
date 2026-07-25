# Nutrition5k Vision Side-Experiment — N4, MoE-OFF Ablation — 2026-07-25 (Kaggle T4×2)

**Status: `SIDE-EXPERIMENT, COMPLETE (measured)` — a bounded ablation of the 2026-07-18 baseline experiment, NOT the canonical 45K text-pretraining run. Trains the same `model/nutrition_vision.py` vision trunk with the MoE layers removed (`MOE_LAYER_IDS = ()`), Liquid kept ON. Does not touch `train/train.py`, the 45K training path, or any 45K readiness/claim surface.**

## What this is

BACKLOG item N4 ("MoE-OFF ablation on Nutrition5k, same hardware constraint as N3"). Run on Kaggle (2× T4, single-GPU used) as job `02_nutrition5k_moe_off` of the same 5-job batch as N3 (see `evidence/2026-07-25-nutrition5k-liquid-off/README.md`), via the same standalone orchestrator, which patched `MOE_LAYER_IDS = (3, 6)` → `()` on a private per-job copy of `scripts/train_nutrition5k.py` — the shared, tracked copy was never modified.

Config (see `dataset_manifest.json`, `metrics.json`): 8,559,365 params · 8 layers × 256 hidden · GQA attention 8q/4kv · **MoE experts at layers [] (removed)** · LiquidMixer at layer [5] (kept) · 256×256 input, 16×16 patches · AMP on · 47/60 epochs (early-stopped) · wall time 1h 05m.

## 🟢 MEASURED — training result (Kaggle, T4, cuda)

| Target | MAE | MAE% | 2026-07-18 baseline (MoE ON) MAE% | Paper 2D Direct MAE% | Paper always-predict-mean MAE% |
|---|---:|---:|---:|---:|---:|
| calories | 79.0 kcal | 30.9% | 29.6% | 26.1% | 60.2% |
| mass | 53.6 g | 27.0% | 24.9% | 18.8% | 58.5% |
| fat | 6.1 g | 47.8% | 44.8% | 34.2% | 67.6% |
| carb | 9.3 g | 47.3% | 45.1% | 31.9% | 62.1% |
| protein | 7.8 g | 44.8% | 42.5% | 29.5% | 62.1% |

## 🟡 Interpretation — real-task evidence, MoE contributes measurably here

Removing MoE **worsened** every single target relative to the baseline (calorie MAE% 30.9% vs 29.6%, and all four other targets similarly higher), unlike N3 (Liquid-OFF), which stayed within noise of baseline. Read together, N3+N4 suggest MoE contributes real, measurable value to this task while Liquid does not — the opposite conclusions for the two components, from the same experimental setup. This is a genuinely different comparative signal from the toy-scale ablations (which only tested Liquid, not MoE) and is worth noting as new evidence, without reopening any sealed 45K architecture decision (MoE 8e2 is not up for review here — this is a small, separate side-task).

## Checkpoints — NOT included (weights excluded on purpose)

`.pt` files are gitignored (`*.pt`, `/checkpoints/`), currently sitting on this laptop at `~/Downloads/mertformer_batch/orchestrator/mertformer_batch_output/02_nutrition5k_moe_off/collected/checkpoints/` (not deleted, not moved — disk on this machine is at 100%/1.5GB free as of this evidence pass, retention location still an open call). Referenced by SHA256 only:
- `nutrition5k_best.pt` (best val calorie MAE 79.0 kcal, 102,860,957 bytes) → `311f1348bef00788582711131809e2782d353e83595144d525e60748246e6d9d`
- `nutrition5k_latest.pt` (final epoch before early stop, 102,861,913 bytes) → `55d9479728bfad159bb1c589ccc3533a9122aadf6439728bd061908d2c5affe1`

## Known, accepted limitations (not bugs) — same as the 2026-07-18 baseline

Same partial-dataset caveat as the baseline experiment (2,755/4,059 train, 507/709 test dishes staged — see `dataset_manifest.json`), same single-split/no-cross-validation design.

## Boundary (does NOT prove)

Not a claim about the canonical 3.67B / 45K model or its MoE 8e2 architecture decision. Not benchmark-verified against the paper's own JFT-300M-pretrained backbone. Does not change 45K readiness (`TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP / START_ALLOWED`, unaffected). Orchestrator code that ran this job (`kaggle_batch_runner.py`) lives only in this repo's `scripts/` as tooling — it is not part of the 45K training path.

## Reproduce

`python scripts/train_nutrition5k.py` with `MOE_LAYER_IDS = ()` patched in a private copy (never in the tracked file — see `scripts/kaggle_batch_runner.py::patch_constant()`).
