"""Evaluation: holdout metrics, Lichess puzzles and a Stockfish Elo ladder."""
from .holdout import evaluate_all_splits, evaluate_split
from .puzzles import evaluate_puzzles, iter_puzzles, sample_puzzles, solve_puzzle
from .elo import estimate_rating, expected_score, log_likelihood, run_elo_ladder, OPENING_BOOK
from .engine import StockfishOpponent, ensure_stockfish, engine_version, find_stockfish
from .benchmark import run_stockfish_benchmark

__all__ = [
    "evaluate_split",
    "evaluate_all_splits",
    "evaluate_puzzles",
    "iter_puzzles",
    "sample_puzzles",
    "solve_puzzle",
    "estimate_rating",
    "expected_score",
    "log_likelihood",
    "run_elo_ladder",
    "OPENING_BOOK",
    "StockfishOpponent",
    "ensure_stockfish",
    "engine_version",
    "find_stockfish",
    "run_stockfish_benchmark",
]
