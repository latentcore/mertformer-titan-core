#!/usr/bin/env python3
"""chessformer-tiny vs Stockfish 18 fixed at UCI_Elo=1700, 10 games.

Same methodology (opponent config, opening book style, mode='policy') as
evidence/2026-09-03-chess-elo-reverify/live_elo_bench.py, but fixed-opponent
(not a ladder) since the question here is specifically "can it beat 1700
Elo even once" -- not a fresh overall Elo estimate.
"""
import json
import sys
import time
from pathlib import Path

import chess
import chess.engine
import torch

# <REDACTED_LOCAL_PATH> -- local checkpoint directory (chessformer-tiny-step30000.pt
# + a copy of the vendored chessformer/ package), same layout as this repo's
# own evidence/2026-08-02-chess-searchless-5070/chessformer/. Not committed
# here per EVIDENCE_MANIFEST.json's checkpoint policy; point this at wherever
# you've placed the checkpoint locally.
ROOT = Path(r"<REDACTED_LOCAL_PATH>")
sys.path.insert(0, str(ROOT))
from chessformer.board import VOCAB_SIZE
from chessformer.model import ChessFormer
from chessformer.config import ModelConfig
from chessformer.inference import play_game

# <REDACTED_LOCAL_PATH> -- local Stockfish 18 binary (same one used in the
# 2026-09-03 re-verify); point this at your own local Stockfish executable.
STOCKFISH = r"<REDACTED_LOCAL_PATH>"
MOVETIME_MS = 100
OPPONENT_ELO = 1700
N_GAMES = 10

OPENINGS = [
    [],
    ["e2e4", "e7e5"],
    ["e2e4", "c7c5"],
    ["d2d4", "d7d5"],
    ["d2d4", "g8f6"],
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
n_params = sum(p.numel() for p in model.parameters())
print(f"loaded checkpoint, step={payload.get('state', {}).get('step')}, params={n_params:,}", flush=True)
assert n_params == 4592740, f"PARAM COUNT MISMATCH: {n_params} != 4592740 (wrong checkpoint!)"

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
engine.configure({"UCI_LimitStrength": True, "UCI_Elo": OPPONENT_ELO, "Threads": 1, "Hash": 64})
print(f"Stockfish configured: UCI_LimitStrength=True, UCI_Elo={OPPONENT_ELO}", flush=True)


def opponent_fn(board: chess.Board):
    result = engine.play(board, chess.engine.Limit(time=MOVETIME_MS / 1000.0))
    return result.move.uci() if result.move else None


games = []
t0 = time.time()
game_no = 0
for opening in OPENINGS:
    for model_white in (True, False):
        game_no += 1
        t_g = time.time()
        res = play_game(
            model, device,
            opponent_move_fn=opponent_fn,
            model_plays_white=model_white,
            opening_moves=opening,
            max_plies=300,
            mode="policy",
        )
        elapsed = round(time.time() - t_g, 2)
        entry = {
            "game": game_no,
            "opening": opening,
            "model_color": "white" if model_white else "black",
            "result": res["result"],
            "plies": res["plies"],
            "model_score": res["model_score"],
            "elapsed_sec": elapsed,
        }
        games.append(entry)
        print(f"  game {game_no}/{N_GAMES} opening={opening or 'start'} "
              f"model={'white' if model_white else 'black'} result={res['result']} "
              f"plies={res['plies']} score={res['model_score']} ({elapsed}s)", flush=True)

engine.quit()

wins = sum(1 for g in games if g["model_score"] == 1.0)
draws = sum(1 for g in games if g["model_score"] == 0.5)
losses = sum(1 for g in games if g["model_score"] == 0.0)
total_score = sum(g["model_score"] for g in games)

report = {
    "schema": "chessformer_fixed_opponent_test_v1",
    "date": "2026-09-04",
    "purpose": "Answer: can chessformer-tiny (measured ~1509 Elo) beat Stockfish 18 fixed at UCI_Elo=1700 even once?",
    "checkpoint_verified_params": n_params,
    "opponent": {"engine": "Stockfish 18", "UCI_LimitStrength": True, "UCI_Elo": OPPONENT_ELO, "movetime_ms": MOVETIME_MS},
    "mode": "policy (searchless)",
    "games_played": N_GAMES,
    "wins": wins,
    "draws": draws,
    "losses": losses,
    "score": total_score,
    "score_rate": round(total_score / N_GAMES, 4),
    "elapsed_sec_total": round(time.time() - t0, 2),
    "games": games,
}
out_path = Path(__file__).resolve().parent / "stockfish_1700_test_report.json"
out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\n=== FINAL ===")
print(json.dumps({k: v for k, v in report.items() if k != "games"}, indent=2))
