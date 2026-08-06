# ChessFormerAI — searchless chess, RTX 5070, 2026-08-02

A real, checkpoint-bound training run of **ChessFormerAI** — a small, independent
side project that mirrors this repo's own architecture family (BitNet b1.58,
GQA, sparse MoE, Liquid/CfC) at a scale a single consumer GPU can actually
train, built as a standalone package (`ChessFormerAI/chessformer`, not part of
this repo) developed against a read-only mirror of this repo's `layers/`.

**This is not the canonical MertFormer Titan model.** It shares an architecture
family, not a checkpoint, a parameter count, or a training run. It does **not**
close this repo's own 45K/H100/H200 gap (see [STATUS.md](../../STATUS.md)) —
that gap is unaffected by anything in this folder.

## What actually happened

- **4,592,740 parameters** ("tiny" preset), `use_bitnet=true, use_moe=true,
  use_liquid=true` — all three resolved into the instantiated model (see
  `model_report.json`'s note on the parameter-count cross-check).
- **32,150 of 100,365 planned steps** (~32%), stopped by the operator, not by
  divergence or a crash. Loss fell smoothly from 6.12 to 3.82 with no spikes
  — see `training_curve.png`.
- **16.2M Lichess-eval positions**, packed with a hash-based train/val/test
  split (see `dataset_provenance.json`).

## Real measured results

The original run's own `holdout`/`puzzles`/`elo` stages saw `should_stop()`
already `true` by the time they ran and produced no real numbers (all
`null`/`not_run` in the run's own `reports/`). The three eval reports in this
folder are from a **separate, retroactive, inference-only pass** against the
saved `best.pt` checkpoint (step 30,000, the lowest-val-loss checkpoint) —
no retraining, no resuming, just loading weights and evaluating.

| Metric | Result | Comparable to |
|---|---|---|
| **Puzzle accuracy (strict)** | **45.78%** (2,289/5,000, reservoir sample) | DeepMind's Searchless Chess (arXiv:2402.04494): 9M-param model 85.5%, 136M 92.1%, 270M 93.5% — **this is the directly comparable metric**, same database, same protocol |
| Holdout masked policy top-1 | 39.4% (val) / 50.1% (test) | — |
| Holdout WDL accuracy | 85.4% (val) / 86.0% (test) | — |
| Elo estimate | **1509** (95% CI 1452–1567), 140 games vs Stockfish 18, `UCI_LimitStrength` | **Not comparable** to the 2895 Lichess-blitz figure in the DeepMind paper — different rating pool (Stockfish's internal UCI_Elo scale vs. Lichess-versus-humans). See `elo_report.json`'s own `interpretation` block. |

**Honest framing, not a record claim.** DeepMind's smallest (9M) model was
trained on ~15 billion Stockfish action-value examples on TPU-scale compute.
This run used 16.2 million positions (~925x less data) on a single 8GB laptop
GPU for a few hours, on a model roughly half the parameter count. Reaching
~54% of the 9M model's puzzle accuracy (45.78% vs. 85.5%) under those
constraints is a data/compute-efficiency data point, not a benchmark result —
`chessformer/config.py`'s own `MIN_POSITIONS_PER_PARAM` autoscaler picked the
"tiny" preset specifically *because* the dataset couldn't support a larger
model honestly; that is a documented safety mechanism, not a limitation being
hidden here.

Zero illegal moves were played across all 140 Elo-ladder games and all 5,000
puzzle attempts — every move is drawn from a legality-masked policy and
re-checked before being pushed (`chessformer/board.py`), so illegal moves are
structurally impossible regardless of what the raw (unmasked) policy would
have picked. `elo_report.json`'s `raw_illegal_top1_events: 6018` (of 12,159
total plies) is a separate, honest diagnostic of what the network learned
*unaided* by the mask — not a safety gap.

## What is NOT in this folder

- **Checkpoint weights.** `.pt` files (~52.7 MB each) stay local; `EVIDENCE_MANIFEST.json`
  carries their SHA256 for reference, not the weights themselves.
- **Raw PGN game records** from the Elo ladder — `elo_report.json`'s own
  `pgn_dir` field records where they live locally; not included in this
  package to keep it small. Available on request / regenerable by re-running
  the same retroactive evaluation against the same checkpoint.
- Any claim about this run improving, validating, or substituting for the
  canonical 45K run.

## Files

- `retroactive_eval.py` — the script that produced `holdout_report.json`/`puzzle_report.json`/
  `elo_report.json`: loads `best.pt` (step 30,000) and calls the existing
  `chessformer.eval.holdout`/`puzzles`/`benchmark` functions directly, inference-only, no
  retraining or resuming. Included verbatim for full reproducibility.
- `training_report.json` — steps, loss, throughput, why the operator stopped it
- `model_report.json` — architecture, resolved BitNet/MoE/Liquid, parameter accounting
- `holdout_report.json`, `puzzle_report.json`, `elo_report.json` — the three retroactive eval passes
- `dataset_provenance.json` — source, split policy, parse stats
- `environment_snapshot.json` — training hardware (RTX 5070) + eval hardware note
- `architecture_parity_summary.json` — the 4 drift/bug fixes vs. `scripts/chess_5080_onefile.py`, self-verified against `layers/` via `torch.allclose`
- `EVIDENCE_MANIFEST.json` — checkpoint SHA256 references
- `training_curve.png` — loss / accuracy / grad-norm over the run
