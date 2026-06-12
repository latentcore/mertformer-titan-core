# $0 Kaggle Pilot — LiquidRouter ON vs OFF

The cheapest meaningful experiment this repo can run: train a small (~80–100M) MertFormer
**twice** — once with LiquidRouter ON, once OFF — on a **free** Kaggle T4/P100, with pure
next-token cross-entropy (no 70B teacher, no KD, no paid compute), and compare the loss curves.

This is the single domino that unlocks the GPU-gated backlog (items 1–12, 78–82): it produces
the first *measured* signal and the first loss curves the repo can honestly commit.

> **Honesty:** this is a **pilot signal**, not a benchmark claim. A few-hundred-step run on a
> tiny corpus tells you the *direction* (does Liquid help?), not a publishable number.

## What it does
`scripts/run_liquid_ablation.py` builds `MertFormer()` directly with a patched ~80–100M config
(the `train_smoke.py` pattern), bypassing the teacher/KD machinery, so it costs $0 and needs
only local data + tokenizer. Both variants use **identical init (seed 1234) and identical data**,
so the only difference is `use_liquid`. Output: `reports/ablations/liquid_ablation_results.json`
+ a printed `Δ(off − on)` verdict.

- Data: real text from `datasets/offline_demo/train.jsonl` + the local TR tokenizer
  (`data/tokenizer/tr`). Falls back to synthetic tokens if either is missing (signal still valid).
- Size: `hidden=256, layers=8, 4 experts top-2, seq=256` (the 128k vocab dominates → ~100M).

## Run it on Kaggle (free GPU)
In a Kaggle notebook with GPU enabled (T4 x2 or P100), after cloning the repo:

```bash
pip install -q -r requirements.txt
python scripts/run_liquid_ablation.py --steps 500 --device cuda --batch-size 8 --seq-len 256
```

Expected: a few hours, $0. Read the verdict and `reports/ablations/liquid_ablation_results.json`.
On Kaggle write checkpoints under `/kaggle/working/` if you add checkpointing.

## Smoke it locally first (proves it runs, ~1 min)
```bash
python scripts/run_liquid_ablation.py --steps 3 --device cpu --batch-size 2 --seq-len 64
```
3 steps is far too few to *learn* — the loss/verdict at 3 steps is noise. The smoke only proves
the path runs end-to-end (both variants, CE computed, JSON written).

## After the pilot (feed the backlog)
1. Commit the loss curves + `liquid_ablation_results.json` (items 7, 78).
2. Record the verdict in `reports/FACTS.json` / the ablation scaffold (`ablations/no_liquid/`).
3. If Liquid clearly helps, keep it default; if not, mark it experimental (item 80) — and that
   decision is what gates whether the expensive 45K / 8×B300 run is worth funding (item 81).

## Knobs
`--steps` (per variant), `--device {auto,cpu,mps,cuda}`, `--batch-size`, `--seq-len`, `--lr`,
`--synthetic` (force synthetic data), `--out` (results path).
