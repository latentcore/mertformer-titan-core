# Chess vs. Stockfish 18 fixed at 1700 Elo — 2026-09-04 — 30% score, 2 wins

**Status: `MEASURED, ANSWERS A SPECIFIC QUESTION`** — not a new Elo estimate,
not a ladder run. This answers one narrow question: *can chessformer-tiny
(measured ~1509 Elo, see `evidence/2026-08-02-chess-searchless-5070/` and
`evidence/2026-09-03-chess-elo-reverify/`) beat Stockfish 18 fixed at
`UCI_Elo=1700` even once?*

## 🟢 MEASURED — result

10 games, Stockfish 18 held at a constant `UCI_Elo=1700` throughout (no
ladder — same opponent strength for all 10), `mode="policy"` (searchless),
100ms Stockfish movetime, same 5-opening book (2 colors each) as the
2026-09-03 re-verify.

| | Result |
|---|---|
| Games | 10 |
| Score | **3.0 / 10 (30%)** |
| W-D-L | **2 - 2 - 6** |
| Checkpoint | `chessformer-tiny-step30000.pt`, verified 4,592,740 params (assert in script) |

Per-game log:

| # | Opening | Model color | Result | Plies |
|---|---|---|---|---|
| 1 | start | White | draw | 93 |
| 2 | start | Black | loss | 167 |
| 3 | 1.e4 e5 | White | **win** | 43 |
| 4 | 1.e4 e5 | Black | loss | 15 |
| 5 | 1.e4 c5 | White | draw | 117 |
| 6 | 1.e4 c5 | Black | loss | 73 |
| 7 | 1.d4 d5 | White | loss | 124 |
| 8 | 1.d4 d5 | Black | **win** | 104 |
| 9 | 1.d4 Nf6 | White | loss | 68 |
| 10 | 1.d4 Nf6 | Black | loss | 121 |

Full per-game JSON: `stockfish_1700_test_report.json`.

## What this answers

Elo math predicts: expected score at a ~191-point rating gap (1509 vs 1700)
is `1/(1+10^(191/400)) ≈ 25%`. Measured: **30%**, inside normal sampling
variance for n=10. **Not impossible — happened twice in 10 games**, plus 2
draws.

## Boundary (does NOT prove)

- **n=10 is small.** This is not a new canonical Elo figure and doesn't
  supersede the 1509 (140 games) or 1495 (36 games) estimates — it's a
  fixed-opponent spot-check at one specific rating, run to answer one
  specific question.
- Game 4 (Black, 1.e4 e5, loss in 15 plies) was a notably fast loss — no
  root-cause analysis was done here; flagged for anyone re-reading the raw
  log, not investigated further in this pass.
- Same limitations as the other evidence in this repo: no PGN saved (only
  result/ply-count/score per game, visible in this folder's own JSON), CPU
  inference, `mode="policy"` (no search added on either side beyond
  Stockfish's own engine strength at the configured Elo).

## Files

- `stockfish_1700_test.py` — the script that produced the result above;
  loads the checkpoint the same way as `evidence/2026-09-03-chess-elo-reverify/live_elo_bench.py`,
  fixes Stockfish at `UCI_Elo=1700` (no ladder), plays 10 games.
- `stockfish_1700_test_report.json` — full per-game log.

## Reproduce

```bash
python stockfish_1700_test.py
```

Requires a Stockfish 18 binary and the `chessformer-tiny-step30000.pt`
checkpoint (checkpoint weights are kept local, not committed — same policy
as the other chess evidence folders in this repo). Both paths are
hardcoded near the top of the script for the machine it was run on
(`STOCKFISH`, `ROOT`) — update them before re-running elsewhere.
