# Nutrition5k Independent Evaluation (Mac-side, fresh code)

Checkpoint: `checkpoints/nutrition5k_best.pt` (epoch 47, sha256 `df857692…` — see README.md) — re-run independently on a Mac (CPU) via `evaluate_nutrition5k.py`, downloading the 507 test-split images fresh rather than reusing any cached copy.
Real held-out test dishes evaluated: 507 (official Nutri-Test split)

| Target | This model MAE | This model MAE% | Paper 2D Direct MAE% | Paper always-predict-mean MAE% |
|---|---:|---:|---:|---:|
| calories | 75.6 kcal | 29.6% | 26.1% | 60.2% |
| mass | 49.5 g | 24.9% | 18.8% | 58.5% |
| fat | 5.8 g | 44.8% | 34.2% | 67.6% |
| carb | 8.9 g | 45.1% | 31.9% | 62.1% |
| protein | 7.4 g | 42.5% | 29.5% | 62.1% |
