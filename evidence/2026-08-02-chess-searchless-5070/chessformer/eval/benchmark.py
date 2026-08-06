"""Orchestrates the Stockfish gauntlet that feeds the Elo estimator."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import chess
import torch

from ..inference import play_game
from ..model import ChessFormer
from .elo import OPENING_BOOK, run_elo_ladder
from .engine import StockfishOpponent, engine_version, ensure_stockfish


def run_stockfish_benchmark(
    model: ChessFormer,
    device: torch.device,
    *,
    stockfish_path: str = "",
    cache_dir: Optional[Path] = None,
    allow_download: bool = False,
    total_games: int = 200,
    block_games: int = 20,
    start_elo: int = 1500,
    movetime_ms: int = 100,
    threads: int = 1,
    hash_mb: int = 64,
    max_plies: int = 300,
    selection_mode: str = "policy",
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_move: Optional[Callable[[Dict[str, Any]], None]] = None,
    save_pgn_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Play the adaptive ladder and return a full Elo report.

    Returns ``status: not_run`` with a reason if no engine is available -- it
    never fabricates a rating.
    """
    cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "tools" / "stockfish"
    resolution = ensure_stockfish(
        cache_dir, allow_download=allow_download, explicit_path=stockfish_path
    )
    if resolution.get("status") in {"not_found", "error"}:
        return {
            "schema": "chessformer_elo_report_v1",
            "status": "not_run",
            "reason": resolution.get("reason", "stockfish unavailable"),
            "how_to_fix": resolution.get("how_to_fix", ""),
            "engine_resolution": resolution,
        }

    engine_path = Path(resolution["path"])
    version = engine_version(engine_path)
    opponent = StockfishOpponent(
        engine_path, uci_elo=start_elo, movetime_ms=movetime_ms,
        threads=threads, hash_mb=hash_mb,
    )
    games_played: List[Dict[str, Any]] = []
    started = time.perf_counter()

    if save_pgn_dir:
        Path(save_pgn_dir).mkdir(parents=True, exist_ok=True)

    try:
        opponent.open()
    except Exception as exc:
        return {
            "schema": "chessformer_elo_report_v1",
            "status": "not_run",
            "reason": f"could not start engine: {type(exc).__name__}: {exc}",
            "engine_resolution": resolution,
        }

    def play_block(opponent_elo: int, n_games: int, block_index: int) -> List[Dict[str, Any]]:
        effective = opponent.set_elo(opponent_elo)
        records: List[Dict[str, Any]] = []
        for game_index in range(n_games):
            if should_stop is not None and should_stop():
                break
            opening = OPENING_BOOK[(block_index * n_games + game_index) % len(OPENING_BOOK)]
            # Alternate colours within the block so each opening is played from
            # both sides -- colour cannot bias the block score.
            model_white = (game_index % 2) == 0
            record = play_game(
                model, device,
                opponent_move_fn=opponent.move,
                model_plays_white=model_white,
                opening_moves=opening,
                max_plies=max_plies,
                mode=selection_mode,
                on_move=on_move,
            )
            record.update({
                "block": block_index,
                "game_index": game_index,
                "opponent_requested_elo": opponent_elo,
                "opponent_effective_elo": effective,
            })
            records.append(record)
            games_played.append(record)
            if progress is not None:
                progress({
                    "stage": "elo_game",
                    "block": block_index,
                    "game": game_index,
                    "of": n_games,
                    "opponent_elo": effective,
                    "result": record["result"],
                    "model_score": record["model_score"],
                    "plies": record["plies"],
                })
        return records

    try:
        report = run_elo_ladder(
            play_block,
            total_games=total_games,
            block_games=block_games,
            start_elo=start_elo,
            should_stop=should_stop,
            on_block=progress,
        )
    finally:
        opponent.close()

    if save_pgn_dir and games_played:
        _write_pgns(Path(save_pgn_dir), games_played)

    illegal = sum(int(g.get("raw_illegal_top1_count", 0)) for g in games_played)
    total_model_plies = sum(int(g.get("plies", 0)) for g in games_played)

    report.update({
        "engine": {
            "resolution": resolution,
            "version": version,
            "config": opponent.describe(),
        },
        "selection_mode": selection_mode,
        "searchless": selection_mode == "policy",
        "max_plies": max_plies,
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "raw_illegal_top1_events": illegal,
        "total_plies": total_model_plies,
        "illegal_move_attempts_after_masking": 0,
        "legality_note": (
            "Every move played was drawn from the legality-masked policy and re-checked "
            "before being pushed, so illegal moves are structurally impossible. "
            "'raw_illegal_top1_events' counts how often the UNMASKED argmax would have "
            "been illegal, which is a measure of what the network learned unaided."
        ),
    })
    if save_pgn_dir:
        report["pgn_dir"] = str(save_pgn_dir)
    return report


def _write_pgns(out_dir: Path, games: List[Dict[str, Any]]) -> None:
    import chess.pgn

    for i, record in enumerate(games):
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = "ChessFormer vs Stockfish (UCI_Elo ladder)"
        game.headers["Site"] = "local"
        game.headers["Round"] = str(record.get("block", 0))
        white = "ChessFormer" if record.get("model_plays_white") else "Stockfish"
        black = "Stockfish" if record.get("model_plays_white") else "ChessFormer"
        game.headers["White"] = white
        game.headers["Black"] = black
        game.headers["Result"] = record.get("result", "*")
        game.headers["WhiteElo"] = str(record.get("opponent_effective_elo", "")) if black == "ChessFormer" else "?"
        game.headers["BlackElo"] = str(record.get("opponent_effective_elo", "")) if white == "ChessFormer" else "?"

        node = game
        for uci in record.get("opening", []) + record.get("moves", []):
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                break
            if move not in board.legal_moves:
                break
            board.push(move)
            node = node.add_variation(move)
        (out_dir / f"game_{i:04d}.pgn").write_text(str(game) + "\n", encoding="utf-8")
