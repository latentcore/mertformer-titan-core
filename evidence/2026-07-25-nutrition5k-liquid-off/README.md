# Nutrition5k Vision Side-Experiment — N3, Liquid-OFF Ablation — 2026-07-25 (Kaggle T4×2)

**Status: `SIDE-EXPERIMENT, COMPLETE (measured)` — a bounded ablation of the 2026-07-18 baseline experiment, NOT the canonical 45K text-pretraining run. Trains the same `model/nutrition_vision.py` vision trunk with the LiquidMixer layer removed (`LIQUID_LAYER_IDS = ()`), MoE kept ON. Does not touch `train/train.py`, the 45K training path, or any 45K readiness/claim surface.**

## What this is

BACKLOG item N3 ("Liquid-OFF ablation on Nutrition5k, requires real training hardware, not runnable from a Mac session"). Run on Kaggle (2× T4, single-GPU used — job did not request DDP) as job `01_nutrition5k_liquid_off` of a 5-job unattended batch, via a standalone orchestrator (`kaggle_batch_runner.py`, see `scripts/kaggle_batch_runner.py` in this repo as of this commit) that patched `LIQUID_LAYER_IDS = (5,)` → `()` on a private per-job copy of `scripts/train_nutrition5k.py` before running — the shared, tracked copy of that script was never modified.

Config (see `dataset_manifest.json`, `metrics.json`): 13,811,719 params · 8 layers × 256 hidden · GQA attention 8q/4kv · MoE 8 experts top-2 at layers [3, 6] · **LiquidMixer at layer [] (removed)** · 256×256 input, 16×16 patches · AMP on · 46/60 epochs (early-stopped) · wall time 27m 31s.

## 🟢 MEASURED — training result (Kaggle, T4, cuda)

| Target | MAE | MAE% | 2026-07-18 baseline (Liquid ON) MAE% | Paper 2D Direct MAE% | Paper always-predict-mean MAE% |
|---|---:|---:|---:|---:|---:|
| calories | 73.4 kcal | 28.7% | 29.6% | 26.1% | 60.2% |
| mass | 50.3 g | 25.4% | 24.9% | 18.8% | 58.5% |
| fat | 5.8 g | 44.9% | 44.8% | 34.2% | 67.6% |
| carb | 8.8 g | 44.9% | 45.1% | 31.9% | 62.1% |
| protein | 7.5 g | 42.8% | 42.5% | 29.5% | 62.1% |

## 🟡 Interpretation — real-task evidence, consistent with the existing toy-scale finding

Removing Liquid slightly **improved** calorie MAE% (28.7% vs 29.6% baseline) and left the other four targets within noise of the baseline (±0.5pp). This is consistent with — and adds real-task evidence to — the existing 12-seed toy-scale Liquid ablation (`ABLATION.md`: OFF 96.32% / ON 94.69%, Δ−1.63pp, p=0.305, inconclusive, "does not earn its cost"). It does **not** overturn the sealed Liquid-keep decision for the 45K canonical run (`DECISIONS.md` "Eight launch-time decisions locked...", 2026-07-19) — that decision is scoped to the 45K text-pretraining architecture, this is a separate, small, non-canonical vision side-experiment. Recorded as an additional data point, not a re-opened decision.

## Checkpoints — NOT included (weights excluded on purpose)

`.pt` files are gitignored (`*.pt`, `/checkpoints/`), currently sitting on this laptop at `~/Downloads/mertformer_batch/orchestrator/mertformer_batch_output/01_nutrition5k_liquid_off/collected/checkpoints/` (not deleted, not moved to git or elsewhere — disk on this machine is at 100%/1.5GB free as of this evidence pass, so long-term retention location is still an open call, not made here). Referenced by SHA256 only:
- `nutrition5k_best.pt` (best val calorie MAE 73.4 kcal, 165,952,569 bytes) → `356f75505a0da0a03b0f1daefaaed975411c85531560ee605e5452b29db4f48e`
- `nutrition5k_latest.pt` (final epoch before early stop, 165,953,917 bytes) → `e9526da28b6c61ef0fdf76ed561ee835143cf832fca68c8dcd8294e0c62452ba`

## Known, accepted limitations (not bugs) — same as the 2026-07-18 baseline

Same partial-dataset caveat as the baseline experiment (2,755/4,059 train, 507/709 test dishes staged — see `dataset_manifest.json`), same single-split/no-cross-validation design, same MoE-router-on-raster-scan-order caveat (documented in `model/nutrition_vision.py`).

## Boundary (does NOT prove)

Not a claim about the canonical 3.67B / 45K model. Not benchmark-verified against the paper's own JFT-300M-pretrained backbone. Does not change 45K readiness (`TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP / START_ALLOWED`, unaffected) or reopen the sealed Liquid-keep decision. Orchestrator code that ran this job (`kaggle_batch_runner.py`) lives only in this repo's `scripts/` as tooling — it is not part of the 45K training path.

## Reproduce

`python scripts/train_nutrition5k.py` with `LIQUID_LAYER_IDS = ()` patched in a private copy (never in the tracked file — see `scripts/kaggle_batch_runner.py::patch_constant()` for how this job's orchestrator did it without touching the shared script).
