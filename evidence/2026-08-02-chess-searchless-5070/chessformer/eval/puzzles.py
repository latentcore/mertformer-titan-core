"""Lichess puzzle benchmark.

This is the one metric here that is directly comparable to published numbers.
arXiv:2402.04494 reports puzzle accuracy of 85.5% (9M params), 92.1% (136M) and
93.5% (270M) on a 10k-puzzle set drawn from the same Lichess puzzle database.

PROTOCOL (matching the paper's definition)
------------------------------------------
Each puzzle row gives a FEN and a move list. The **first** move in the list is
the opponent's move played *into* the puzzle position; the solver moves second.
A puzzle counts as solved only if the model plays every one of its moves in the
line correctly -- one wrong move fails the whole puzzle. Opponent replies are
taken from the solution line, not chosen by the model.

One deliberate relaxation, also standard: if the model's move delivers immediate
checkmate it is accepted even when it differs from the recorded solution, since
an alternative mate is not an error. Both the strict and relaxed counts are
reported so the number can be read either way.
"""
from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import chess
import numpy as np
import torch

from ..inference import select_move
from ..model import ChessFormer

RATING_BANDS: Sequence[tuple] = (
    (0, 1000), (1000, 1400), (1400, 1800), (1800, 2200), (2200, 2600), (2600, 10000),
)


@dataclass
class Puzzle:
    puzzle_id: str
    fen: str
    moves: List[str]
    rating: int
    themes: str = ""


def iter_puzzles(path: Path, limit: Optional[int] = None) -> Iterator[Puzzle]:
    """Stream puzzles from the zstd-compressed CSV."""
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    count = 0
    with Path(path).open("rb") as raw:
        try:
            reader = dctx.stream_reader(raw, read_across_frames=True)
        except TypeError:
            reader = dctx.stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore", newline="")
        csv_reader = csv.reader(text)
        header = next(csv_reader, None)
        if header is None:
            return
        # Column order has been stable, but resolve by name where possible.
        idx = {name.strip().lower(): i for i, name in enumerate(header)}
        i_id = idx.get("puzzleid", 0)
        i_fen = idx.get("fen", 1)
        i_moves = idx.get("moves", 2)
        i_rating = idx.get("rating", 3)
        i_themes = idx.get("themes", 7)

        for row in csv_reader:
            if len(row) <= max(i_fen, i_moves, i_rating):
                continue
            try:
                rating = int(row[i_rating])
            except (ValueError, IndexError):
                continue
            moves = row[i_moves].split()
            if len(moves) < 2:
                continue
            yield Puzzle(
                puzzle_id=row[i_id] if i_id < len(row) else "",
                fen=row[i_fen],
                moves=moves,
                rating=rating,
                themes=row[i_themes] if i_themes < len(row) else "",
            )
            count += 1
            if limit is not None and count >= limit:
                return


def sample_puzzles(
    path: Path, sample_size: int, *, seed: int = 0, scan_limit: int = 400_000
) -> List[Puzzle]:
    """Reservoir-sample so the set is not biased toward the file's ordering."""
    rng = np.random.default_rng(seed)
    reservoir: List[Puzzle] = []
    for i, puzzle in enumerate(iter_puzzles(path, limit=scan_limit)):
        if len(reservoir) < sample_size:
            reservoir.append(puzzle)
        else:
            j = int(rng.integers(0, i + 1))
            if j < sample_size:
                reservoir[j] = puzzle
    return reservoir


def solve_puzzle(
    model: ChessFormer,
    device: torch.device,
    puzzle: Puzzle,
    *,
    mode: str = "policy",
) -> Dict[str, Any]:
    """Play the model through one puzzle line."""
    try:
        board = chess.Board(puzzle.fen)
    except ValueError:
        return {"status": "bad_fen", "solved": False, "solved_relaxed": False}

    moves = puzzle.moves
    # First listed move is the opponent's; it creates the puzzle position.
    try:
        board.push(chess.Move.from_uci(moves[0]))
    except (ValueError, AssertionError):
        return {"status": "bad_setup_move", "solved": False, "solved_relaxed": False}

    strict = True
    relaxed = True
    model_moves = 0
    first_error: Optional[Dict[str, Any]] = None

    for ply in range(1, len(moves)):
        expected = moves[ply]
        if ply % 2 == 1:
            # Model's turn.
            if board.is_game_over():
                break
            try:
                trace = select_move(model, board, device, mode=mode, top_k_trace=1)
            except ValueError:
                strict = relaxed = False
                break
            model_moves += 1
            played = trace.move
            if played != expected:
                strict = False
                # Accept an alternative immediate mate.
                probe = board.copy(stack=False)
                probe.push(chess.Move.from_uci(played))
                if not probe.is_checkmate():
                    relaxed = False
                    if first_error is None:
                        first_error = {
                            "ply": ply, "expected": expected, "played": played,
                            "fen": board.fen(),
                        }
                    break
            board.push(chess.Move.from_uci(played if relaxed and not strict else expected))
        else:
            try:
                board.push(chess.Move.from_uci(expected))
            except (ValueError, AssertionError):
                break

    return {
        "status": "ok",
        "solved": bool(strict),
        "solved_relaxed": bool(relaxed),
        "model_moves": model_moves,
        "rating": puzzle.rating,
        "puzzle_id": puzzle.puzzle_id,
        "first_error": first_error,
    }


def evaluate_puzzles(
    model: ChessFormer,
    device: torch.device,
    puzzle_path: Path,
    *,
    sample_size: int = 5000,
    seed: int = 0,
    mode: str = "policy",
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    puzzle_path = Path(puzzle_path)
    if not puzzle_path.exists():
        return {
            "schema": "chessformer_puzzle_report_v1",
            "status": "not_run",
            "reason": f"puzzle database not present at {puzzle_path}",
        }

    started = time.perf_counter()
    puzzles = sample_puzzles(puzzle_path, sample_size, seed=seed)
    if not puzzles:
        return {
            "schema": "chessformer_puzzle_report_v1",
            "status": "not_run",
            "reason": "no puzzles could be parsed from the database",
        }

    was_training = model.training
    model.eval()
    solved = relaxed_solved = attempted = 0
    band_stats = {f"{lo}-{hi}": {"n": 0, "solved": 0} for lo, hi in RATING_BANDS}
    failures: List[Dict[str, Any]] = []

    try:
        for i, puzzle in enumerate(puzzles):
            if should_stop is not None and should_stop():
                break
            result = solve_puzzle(model, device, puzzle, mode=mode)
            if result["status"] != "ok":
                continue
            attempted += 1
            solved += int(result["solved"])
            relaxed_solved += int(result["solved_relaxed"])
            for lo, hi in RATING_BANDS:
                if lo <= puzzle.rating < hi:
                    key = f"{lo}-{hi}"
                    band_stats[key]["n"] += 1
                    band_stats[key]["solved"] += int(result["solved"])
                    break
            if not result["solved"] and len(failures) < 20 and result.get("first_error"):
                failures.append(result["first_error"])
            if progress is not None and (i + 1) % 100 == 0:
                progress({
                    "stage": "puzzles",
                    "done": i + 1,
                    "total": len(puzzles),
                    "accuracy": round(solved / max(1, attempted), 4),
                })
    finally:
        model.train(was_training)

    bands = {}
    for key, payload in band_stats.items():
        if payload["n"] == 0:
            continue
        bands[key] = {
            "puzzles": payload["n"],
            "solved": payload["solved"],
            "accuracy": round(payload["solved"] / payload["n"], 4),
        }

    return {
        "schema": "chessformer_puzzle_report_v1",
        "status": "completed",
        "source": str(puzzle_path),
        "sample_size_requested": sample_size,
        "puzzles_attempted": attempted,
        "solved_strict": solved,
        "accuracy_strict": round(solved / max(1, attempted), 4),
        "solved_relaxed": relaxed_solved,
        "accuracy_relaxed": round(relaxed_solved / max(1, attempted), 4),
        "by_rating_band": bands,
        "selection_mode": mode,
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "sample_failures": failures,
        "protocol": {
            "definition": (
                "A puzzle is solved only if the model plays every one of its moves in the "
                "solution line correctly; opponent replies come from the solution."
            ),
            "relaxation": "an alternative move giving immediate checkmate is accepted (relaxed count)",
            "sampling": f"reservoir sample, seed={seed}",
        },
        "comparability": {
            "deepmind_270m_puzzle_accuracy": 0.935,
            "deepmind_136m_puzzle_accuracy": 0.921,
            "deepmind_9m_puzzle_accuracy": 0.855,
            "source": "arXiv:2402.04494 Table 1",
            "note": (
                "Same puzzle database and same solved-if-exact protocol, so these are "
                "comparable in kind. The puzzle sample differs, so small differences are "
                "not meaningful; sampling error at n=5000 is roughly +/-1%."
            ),
        },
    }
