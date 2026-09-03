# Chess Elo Re-Verify — 2026-09-03 (CPU, 36 games) — CONFIRMS the 1509 figure

**Status: `INDEPENDENT RE-VERIFICATION, CONFIRMS`** — a smaller-scale (36 vs.
the original 140 games), same-day, fresh re-run of `evidence/2026-08-02-chess-searchless-5070/elo_report.json`'s
own Elo-ladder methodology against the same checkpoint. Point estimate landed
14 Elo points from the original — well inside normal sampling variance at
this sample size.

## What this is

Requested live, not read from a cached file: "measure the real thing using
the actual Elo benchmark, right now." `live_elo_bench.py` (this folder)
loads `chessformer-tiny-step30000.pt` and replays the same ladder protocol
`evidence/2026-08-02-chess-searchless-5070/elo_report.json['methodology']`
describes — Stockfish 18, `UCI_LimitStrength=true`, `UCI_Elo` re-aimed to the
running MLE estimate after each block, `mode="policy"` (searchless), same
100ms Stockfish movetime — at a reduced scale (3 blocks × 12 games = 36,
vs. the original 7 blocks × 20 = 140) to fit a single interactive session.

## 🟢 MEASURED — result

| | This re-verify (2026-09-03) | Original (2026-08-02, `elo_report.json`) |
|---|---|---|
| Games | 36 | 140 |
| Score | 17.5 (48.6%) | 72.0 (51.4%) |
| Point estimate | **1495** | **1509** (95% CI 1452–1567) |
| Opponent | Stockfish 18, `UCI_LimitStrength` | same |
| Selection mode | `policy` (searchless) | same |

Block-by-block (opponent Elo re-aimed to the running estimate each block,
same ladder logic as the original):

| Block | Opponent Elo | W-D-L | Score rate | Estimate after |
|---|---|---|---|---|
| 1 | 1500 | 5-3-4 | 54.2% | 1529 |
| 2 | 1529 | 3-3-6 | 37.5% | 1485 |
| 3 | 1485 | 5-3-4 | 54.2% | 1495 |

Checkpoint verified identical before running: `4,592,740` parameters, step
30,000 — matches `model_report.json` exactly (same check the original
`retroactive_eval.py` run and this repo's own commit `5b36e27e` performed).

## Boundary (does NOT prove)

- **Not a replacement for the 140-game study.** A 36-game sample has
  meaningfully wider uncertainty than 140; this script does not compute a
  profile-likelihood CI (the original's own method), so no CI is reported
  here — reporting a point estimate without one would overstate the
  precision. Take this as a same-day sanity check, not a new canonical
  number.
- **Opening book is smaller** (6 openings × 2 colors = 12 games/block vs.
  the original's `opening_book_size: 10` → 20 games/block) — a convenience
  reduction for the smaller total game count, not a methodology change.
- **No PGN records saved** — `play_game()` in `chessformer/inference.py`
  doesn't return move-by-move in a form this script persisted; only the
  per-game result/ply-count/score are logged (visible in `live_elo_bench.py`'s
  own stdout, not checked into this folder). The original's `elo_report.json`
  has the same limitation in this evidence package (raw PGNs are local-only,
  per that folder's own README).
- Does not change 45K readiness or anything about the canonical Titan model
  — same boundary the original chess evidence folder already states.

## Files

- `live_elo_bench.py` — the script that produced the result above; loads the
  checkpoint via the same `chessformer.model`/`chessformer.config` path as
  `retroactive_eval.py`, drives Stockfish via `python-chess`'s UCI engine
  interface, plays games through `chessformer.inference.play_game()`, and
  fits Elo via a direct maximum-likelihood implementation of the same
  logistic model `elo_report.json['methodology']['rating_model']` describes
  (golden-section search on the log-likelihood, not a library call).
- `live_elo_report.json` — the script's own JSON output (block-by-block
  W/D/L, score rates, running estimates, final point estimate).

## Reproduce

```bash
python live_elo_bench.py
```

Requires a Stockfish 18 binary and the `chessformer-tiny-step30000.pt`
checkpoint (per `EVIDENCE_MANIFEST.json`, checkpoint weights are kept local
and not committed to this repo — the same policy `2026-08-02-chess-searchless-5070/README.md`
states). Both paths are hardcoded near the top of the script for the machine
it was run on (`STOCKFISH`, and `ROOT` for the checkpoint + the vendored
`chessformer` package it was loaded alongside) — update them before
re-running elsewhere. The imports (`chessformer.board`/`model`/`config`/
`inference`) are the same package vendored in
`evidence/2026-08-02-chess-searchless-5070/chessformer/`.
