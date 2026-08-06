"""Elo estimation against strength-limited Stockfish.

METHOD
------
Games are played in blocks against Stockfish configured with
``UCI_LimitStrength`` + ``UCI_Elo = R_opp``. After each block the model's
rating is re-estimated by maximum likelihood under the standard Elo model::

    E(R, R_opp) = 1 / (1 + 10 ** ((R_opp - R) / 400))
    logL(R)     = sum_g [ s_g * log E_g + (1 - s_g) * log (1 - E_g) ]

with a draw scored as ``s = 0.5`` (this is the convention Elo itself is defined
on). The next block is played against an opponent set to the current estimate,
which concentrates games where they are most informative -- a block played
against an opponent 800 points away carries almost no information.

The reported interval is a **profile-likelihood 95% CI**: the set of ratings
whose log-likelihood is within 1.92 (``chi2(1, 0.95)/2``) of the maximum.

WHAT THIS NUMBER IS AND IS NOT
------------------------------
It is an internal measurement against Stockfish's own strength-limiting scale.
It is **not** a Lichess rating and not a FIDE rating, and it is not directly
comparable to the 2895 Lichess-blitz-versus-humans figure from
arXiv:2402.04494 -- that was measured against human opponents on a different
rating pool. Puzzle accuracy (``puzzles.py``) is the metric that *is* directly
comparable.

If the model scores 0% or 100% across every block, the likelihood is unbounded
and no finite estimate exists; that is reported as such rather than papered over
with a made-up number.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .engine import UCI_ELO_MAX, UCI_ELO_MIN

# Ten standard openings, played from both sides, so colour and opening choice
# cannot bias the result.
OPENING_BOOK: Tuple[Tuple[str, ...], ...] = (
    (),                                            # start position
    ("e2e4", "e7e5"),                              # open game
    ("e2e4", "c7c5"),                              # sicilian
    ("e2e4", "e7e6"),                              # french
    ("e2e4", "c7c6"),                              # caro-kann
    ("d2d4", "d7d5"),                              # closed
    ("d2d4", "g8f6"),                              # indian
    ("c2c4", "e7e5"),                              # english
    ("g1f3", "d7d5"),                              # reti
    ("d2d4", "d7d5", "c2c4", "e7e6"),              # QGD
)

LOG10 = math.log(10.0)
CHI2_95_HALF = 1.9207  # chi2(1, 0.95) / 2


def expected_score(rating: float, opponent: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0))


def log_likelihood(rating: float, observations: Sequence[Tuple[float, float]]) -> float:
    """``observations`` is a list of (opponent_elo, score in [0,1]) per game."""
    total = 0.0
    for opponent, score in observations:
        e = expected_score(rating, opponent)
        e = min(1.0 - 1e-12, max(1e-12, e))
        total += score * math.log(e) + (1.0 - score) * math.log(1.0 - e)
    return total


def estimate_rating(
    observations: Sequence[Tuple[float, float]],
    *,
    lo: float = 0.0,
    hi: float = 4000.0,
    step: float = 1.0,
) -> Dict[str, Any]:
    """MLE rating plus a profile-likelihood 95% CI."""
    if not observations:
        return {"status": "no_games", "rating": None}

    total_score = sum(s for _, s in observations)
    n = len(observations)
    if total_score <= 0.0:
        return {
            "status": "unbounded_below",
            "rating": None,
            "games": n,
            "score_rate": 0.0,
            "note": (
                "the model scored 0 in every game, so the likelihood is maximized as "
                "rating -> -inf; no finite Elo can be estimated from this data"
            ),
            "opponent_elo_range": [min(o for o, _ in observations), max(o for o, _ in observations)],
        }
    if total_score >= n:
        return {
            "status": "unbounded_above",
            "rating": None,
            "games": n,
            "score_rate": 1.0,
            "note": (
                "the model won every game, so the likelihood is maximized as "
                "rating -> +inf; only a lower bound can be stated"
            ),
            "opponent_elo_range": [min(o for o, _ in observations), max(o for o, _ in observations)],
        }

    grid = np.arange(lo, hi + step, step, dtype=np.float64)
    lls = np.array([log_likelihood(float(r), observations) for r in grid])
    best_idx = int(np.argmax(lls))
    best_rating = float(grid[best_idx])
    best_ll = float(lls[best_idx])

    threshold = best_ll - CHI2_95_HALF
    inside = np.flatnonzero(lls >= threshold)
    ci_low = float(grid[inside[0]]) if inside.size else best_rating
    ci_high = float(grid[inside[-1]]) if inside.size else best_rating

    return {
        "status": "estimated",
        "rating": round(best_rating, 1),
        "ci95_low": round(ci_low, 1),
        "ci95_high": round(ci_high, 1),
        "ci95_width": round(ci_high - ci_low, 1),
        "log_likelihood": round(best_ll, 4),
        "games": n,
        "score_rate": round(total_score / n, 4),
        "method": "profile-likelihood MLE under the standard Elo logistic model",
        "ci_note": f"95% CI = ratings within {CHI2_95_HALF} log-likelihood of the maximum",
        "hit_grid_edge": bool(ci_low <= lo + step or ci_high >= hi - step),
    }


@dataclass
class LadderBlock:
    opponent_elo: int
    effective_opponent_elo: int
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    elapsed_sec: float
    estimate_after: Optional[float] = None
    game_records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opponent_elo": self.opponent_elo,
            "effective_opponent_elo": self.effective_opponent_elo,
            "games": self.games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "score": round(self.score, 3),
            "score_rate": round(self.score / max(1, self.games), 4),
            "elapsed_sec": round(self.elapsed_sec, 2),
            "estimate_after": self.estimate_after,
        }


def next_opponent_elo(
    current_estimate: Optional[float], last_block: Optional[LadderBlock], default: int
) -> int:
    """Aim the next block at the current estimate; nudge on a shutout block."""
    if current_estimate is None:
        if last_block is None:
            return int(default)
        rate = last_block.score / max(1, last_block.games)
        # A shutout carries no gradient information, so step decisively.
        if rate <= 0.0:
            return int(max(UCI_ELO_MIN, last_block.effective_opponent_elo - 400))
        if rate >= 1.0:
            return int(min(UCI_ELO_MAX, last_block.effective_opponent_elo + 400))
        return int(last_block.effective_opponent_elo)
    return int(min(UCI_ELO_MAX, max(UCI_ELO_MIN, round(current_estimate))))


def run_elo_ladder(
    play_block: Callable[[int, int, int], List[Dict[str, Any]]],
    *,
    total_games: int = 200,
    block_games: int = 20,
    start_elo: int = 1500,
    target_ci_width: float = 120.0,
    should_stop: Optional[Callable[[], bool]] = None,
    on_block: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Drive the adaptive ladder.

    ``play_block(opponent_elo, n_games, block_index)`` must return one record
    per game containing at least ``model_score`` in {0, 0.5, 1}.
    """
    observations: List[Tuple[float, float]] = []
    blocks: List[LadderBlock] = []
    estimate: Optional[float] = None
    played = 0
    block_index = 0
    stopped_early = False
    started = time.perf_counter()

    while played < total_games:
        if should_stop is not None and should_stop():
            stopped_early = True
            break

        opponent = next_opponent_elo(estimate, blocks[-1] if blocks else None, start_elo)
        n = min(block_games, total_games - played)
        # Even game count keeps colours balanced.
        if n > 1 and n % 2 == 1:
            n -= 1
        if n <= 0:
            break

        block_started = time.perf_counter()
        records = play_block(opponent, n, block_index)
        if not records:
            stopped_early = True
            break

        effective = int(records[0].get("opponent_effective_elo", opponent))
        wins = sum(1 for r in records if r["model_score"] == 1.0)
        draws = sum(1 for r in records if r["model_score"] == 0.5)
        losses = sum(1 for r in records if r["model_score"] == 0.0)
        score = float(sum(r["model_score"] for r in records))

        for record in records:
            observations.append((float(effective), float(record["model_score"])))
        played += len(records)

        current = estimate_rating(observations)
        estimate = current.get("rating")

        block = LadderBlock(
            opponent_elo=opponent, effective_opponent_elo=effective, games=len(records),
            wins=wins, draws=draws, losses=losses, score=score,
            elapsed_sec=time.perf_counter() - block_started,
            estimate_after=estimate, game_records=records,
        )
        blocks.append(block)
        block_index += 1

        if on_block is not None:
            on_block({"stage": "elo_block", **block.to_dict(), "estimate": current})

        if (
            current.get("status") == "estimated"
            and current.get("ci95_width", 1e9) <= target_ci_width
            and played >= max(block_games * 3, 40)
        ):
            break

    final = estimate_rating(observations)
    total_score = sum(s for _, s in observations)

    return {
        "schema": "chessformer_elo_report_v1",
        "status": "completed" if observations else "not_run",
        "stopped_early": stopped_early,
        "games_played": len(observations),
        "games_requested": total_games,
        "total_score": round(total_score, 2),
        "overall_score_rate": round(total_score / max(1, len(observations)), 4),
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "estimate": final,
        "blocks": [b.to_dict() for b in blocks],
        "methodology": {
            "opponent": "Stockfish with UCI_LimitStrength=true and UCI_Elo set per block",
            "rating_model": "E = 1/(1+10^((Ropp-R)/400)); draws scored 0.5",
            "estimator": "maximum likelihood over all games",
            "interval": "profile-likelihood 95% CI (delta logL = 1.9207)",
            "ladder": "opponent Elo re-aimed at the running estimate after each block",
            "colour_policy": "each opening played from both sides within a block",
            "opening_book_size": len(OPENING_BOOK),
            "uci_elo_clamp_range": [UCI_ELO_MIN, UCI_ELO_MAX],
        },
        "interpretation": {
            "is_a_lichess_rating": False,
            "is_a_fide_rating": False,
            "comparable_to_deepmind_2895": False,
            "note": (
                "This is an internal measurement against Stockfish's UCI_Elo scale. "
                "The 2895 figure in arXiv:2402.04494 is Lichess blitz versus human "
                "opponents and lives on a different rating pool; the two numbers are "
                "not interchangeable. Puzzle accuracy is the directly comparable metric."
            ),
        },
    }
