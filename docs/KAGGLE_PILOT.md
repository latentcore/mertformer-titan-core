# $0 Kaggle Pilot — LiquidRouter ON vs OFF (step by step)

Train a small (~80–100M) MertFormer **twice** — Liquid ON, then OFF — on a **free** Kaggle GPU,
with pure next-token cross-entropy (no 70B teacher, no KD, no paid compute), and compare the loss.
This is the single domino that produces the repo's first *measured* signal.

> **Honesty:** a few-hundred-step run on a tiny corpus tells you the **direction** (does Liquid
> help?), not a publishable number. It is a pilot signal, not a benchmark claim.

---

## 0. One-time Kaggle setup
1. Create a free account at kaggle.com and **verify your phone** (required to enable GPU + internet).
2. **New → Notebook.**
3. Right sidebar → **Session options**:
   - **Accelerator: GPU T4 x2** (or P100).
   - **Internet: On.**

## 1. Get the repo onto Kaggle (it is PRIVATE)
Pick ONE:

**Option A — upload as a private Dataset (no token, simplest):**
1. On your Mac: zip the repo (include the gitignored tokenizer/data so the pilot gets *real* text):
   ```bash
   cd <the folder that CONTAINS your mertformer-titan-core checkout>
   zip -r mertformer.zip mertformer-titan-core -x '*/.git/*' '*/.titan-venv/*'
   ```
2. Kaggle → **Datasets → New Dataset →** upload `mertformer.zip` (set it Private).
3. In the notebook: **Add Input →** your dataset. It mounts at `/kaggle/input/<name>/`.
4. First cell:
   ```python
   !cp -r /kaggle/input/<your-dataset-name>/mertformer-titan-core /kaggle/working/repo
   %cd /kaggle/working/repo
   ```

**Option B — git clone with a token:**
1. GitHub → Settings → Developer settings → **Fine-grained PAT**, read-only on this repo.
2. Kaggle → notebook **Add-ons → Secrets →** add `GH_TOKEN` = your token.
3. First cell:
   ```python
   from kaggle_secrets import UserSecretsClient
   tok = UserSecretsClient().get_secret("GH_TOKEN")
   !git clone https://latentcore:{tok}@github.com/latentcore/mertformer-titan-core.git /kaggle/working/repo
   %cd /kaggle/working/repo
   ```
   (With a plain `git clone` the tokenizer/data are gitignored → the pilot uses the tracked
   `datasets/validation.jsonl` + a free `gpt2` tokenizer download — still real text. Option A also
   ships the local TR tokenizer if you want that instead.)

## 2. Deps
Kaggle's image already has `torch` + `transformers`. If any import fails, run once:
```python
!pip install -q accelerate || pip install -q -r requirements.txt
```

## 3. Run the pilot (the actual command)
```python
!python scripts/run_liquid_ablation.py --steps 500 --device cuda --batch-size 8 --seq-len 256
```
Expect a few hours, **$0**. It prints `Δ(off − on)` and writes
`reports/ablations/liquid_ablation_results.json`.

## 4. Get the results out
```python
import json; print(open("reports/ablations/liquid_ablation_results.json").read())
# (the JSON file in /kaggle/working is downloadable from the notebook's Output tab)
```

---

## Smoke it locally first (proves it runs, ~1 min, no GPU)
```bash
.titan-venv/bin/python scripts/run_liquid_ablation.py --steps 3 --device cpu --batch-size 2 --seq-len 64
```
3 steps is far too few to *learn* — the loss/verdict there is noise; it only proves the path runs.

## What data/tokenizer it actually uses
- **Text:** `datasets/offline_demo/train.jsonl` if present (local), else the tracked
  `datasets/validation.jsonl` (ships in the repo → available on a fresh Kaggle clone).
- **Tokenizer:** local TR tokenizer if present → else `gpt2` (free ~1MB download) → else char-level
  (no download). Both variants use **identical data + identical init (seed 1234)**, so the only
  difference is `use_liquid`.

## After the pilot (feed the backlog)
1. Commit the loss curves + `liquid_ablation_results.json` (items 7, 78).
2. Record the verdict in `reports/FACTS.json` / `ablations/no_liquid/`.
3. If Liquid clearly helps → keep default; if not → mark experimental (item 80). That decision gates
   whether the expensive 45K / 8×B300 run is worth funding (item 81).

## Knobs
`--steps` (per variant), `--device {auto,cpu,mps,cuda}`, `--batch-size`, `--seq-len`, `--lr`,
`--synthetic` (force synthetic data), `--out` (results path).
