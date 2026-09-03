#!/usr/bin/env python3
"""Live, fresh Elo re-measurement of chessformer-tiny vs Stockfish 18,
replicating the methodology in elo_report.json (UCI_LimitStrength ladder,
policy/searchless mode, MLE Elo fit) at a smaller scale (time-budgeted).
"""
import json
import math
import sys
import time
from pathlib import Path

import chess
import chess.engine
import torch

# <REDACTED_LOCAL_PATH> -- local checkpoint directory (chessformer-tiny-step30000.pt
# + a copy of the vendored chessformer/ package, same layout as this repo's
# own evidence/2026-08-02-chess-searchless-5070/chessformer/). Not committed
# here per EVIDENCE_MANIFEST.json's checkpoint policy; point this at wherever
# you've placed the checkpoint locally.
ROOT = Path(r"<REDACTED_LOCAL_PATH>")
sys.path.insert(0, str(ROOT))
from chessformer.board import VOCAB_SIZE
from chessformer.model import ChessFormer
from chessformer.config import ModelConfig
from chessformer.inference import play_game

# <REDACTED_LOCAL_PATH> -- local Stockfish 18 binary (downloaded from
# https://github.com/official-stockfish/Stockfish/releases/tag/sf_18 for
# this run); point this at your own local Stockfish executable.
STOCKFISH = r"<REDACTED_LOCAL_PATH>"
MOVETIME_MS = 100
GAMES_PER_BLOCK = 12  # 6 openings x 2 colors
N_BLOCKS = 3
START_ELO = 1500
ELO_RANGE = (1320, 3190)

# Small opening book (well-known short lines), UCI.
OPENINGS = [
    [],  # start position
    ["e2e4", "e7e5"],
    ["e2e4", "c7c5"],
    ["d2d4", "d7d5"],
    ["d2d4", "g8f6"],
    ["c2c4", "e7e5"],
]

device = torch.device("cpu")
payload = torch.load(ROOT / "chessformer-tiny-step30000.pt", map_location=device, weights_only=False)
cfg_dict = dict(payload["model_config"])
if isinstance(cfg_dict.get("liquid_layers_idx"), list):
    cfg_dict["liquid_layers_idx"] = tuple(cfg_dict["liquid_layers_idx"])
cfg = ModelConfig(**cfg_dict)
model = ChessFormer(cfg, VOCAB_SIZE).to(device)
model.load_state_dict(payload["model_state"])
model.eval()
print(f"loaded checkpoint, step={payload.get('state', {}).get('step')}, "
      f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
engine.configure({"UCI_LimitStrength": True, "Threads": 1, "Hash": 64})


def make_opponent_fn(elo: int):
    engine.configure({"UCI_Elo": elo})

    def fn(board: chess.Board):
        result = engine.play(board, chess.engine.Limit(time=MOVETIME_MS / 1000.0))
        return result.move.uci() if result.move else None

    return fn


def elo_negloglik(r: float, games: list) -> float:
    nll = 0.0
    for opp_elo, score in games:
        e = 1.0 / (1.0 + 10 ** ((opp_elo - r) / 400.0))
        e = min(max(e, 1e-9), 1 - 1e-9)
        if score == 1.0:
            nll -= math.log(e)
        elif score == 0.0:
            nll -= math.log(1 - e)
        else:
            nll -= 0.5 * math.log(e) + 0.5 * math.log(1 - e)
    return nll


def fit_elo(games: list) -> float:
    lo, hi = 400.0, 3200.0
    for _ in range(60):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if elo_negloglik(m1, games) < elo_negloglik(m2, games):
            hi = m2
        else:
            lo = m1
    return round((lo + hi) / 2, 1)


all_games = []  # (opponent_elo, score)
running_estimate = float(START_ELO)
block_reports = []
t_start = time.time()

for block_idx in range(N_BLOCKS):
    opp_elo = int(min(max(round(running_estimate), ELO_RANGE[0]), ELO_RANGE[1]))
    wins = draws = losses = 0
    block_t0 = time.time()
    game_no = 0
    for opening in OPENINGS:
        for model_white in (True, False):
            game_no += 1
            opp_fn = make_opponent_fn(opp_elo)
            res = play_game(
                model, device,
                opponent_move_fn=opp_fn,
                model_plays_white=model_white,
                opening_moves=opening,
                max_plies=300,
                mode="policy",
            )
            score = res["model_score"]
            all_games.append((opp_elo, score))
            if score == 1.0:
                wins += 1
            elif score == 0.0:
                losses += 1
            else:
                draws += 1
            print(f"  block{block_idx+1} g{game_no}/{GAMES_PER_BLOCK} "
                  f"opp_elo={opp_elo} white={'model' if model_white else 'sf'} "
                  f"result={res['result']} plies={res['plies']} score={score}", flush=True)
    block_score = wins + 0.5 * draws
    running_estimate = fit_elo(all_games)
    block_reports.append({
        "block": block_idx + 1, "opponent_elo": opp_elo, "games": GAMES_PER_BLOCK,
        "wins": wins, "draws": draws, "losses": losses,
        "score": block_score, "score_rate": round(block_score / GAMES_PER_BLOCK, 4),
        "elapsed_sec": round(time.time() - block_t0, 2),
        "estimate_after": running_estimate,
    })
    print(f"block {block_idx+1} done: W{wins} D{draws} L{losses}  "
          f"running Elo estimate = {running_estimate}", flush=True)

engine.quit()

total_games = len(all_games)
total_score = sum(s for _, s in all_games)
final_estimate = fit_elo(all_games)

report = {
    "schema": "chessformer_live_elo_rebench_v1",
    "date": "2026-09-03",
    "games_played": total_games,
    "total_score": total_score,
    "overall_score_rate": round(total_score / total_games, 4),
    "elapsed_sec": round(time.time() - t_start, 2),
    "final_estimate_elo": final_estimate,
    "blocks": block_reports,
    "methodology": {
        "opponent": "Stockfish 18, UCI_LimitStrength=true, UCI_Elo re-aimed to running estimate each block",
        "movetime_ms": MOVETIME_MS,
        "selection_mode": "policy (searchless)",
        "opening_book_size": len(OPENINGS),
        "note": "Smaller-scale live rebench (36 games vs original 140) for a fresh, "
                "same-day sanity check against the documented 1509 figure -- not a "
                "replacement for the full study.",
    },
}
out_path = Path(__file__).resolve().parent / "live_elo_report.json"
out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\n=== FINAL ===")
print(json.dumps(report, indent=2))
