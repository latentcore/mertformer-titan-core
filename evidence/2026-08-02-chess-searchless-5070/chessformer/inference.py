"""Move selection.

Two modes:

``policy``
    Pure single-forward-pass move selection: encode the board, mask illegal
    moves, take the argmax. This is the "searchless" setting -- the regime
    DeepMind's result is about -- and it is the default.

``value_1ply``
    Optionally re-rank the top-k policy candidates by the value head evaluated
    after the move. This is a shallow lookahead, so any strength measured with
    it must be reported as *not* searchless. The evaluation harness records
    which mode produced every number.

Legality is structural, not hoped for: the mask comes from ``python-chess``
and :func:`select_move` re-checks the chosen move before returning it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import chess
import numpy as np
import torch
import torch.nn.functional as F

from .board import (
    ID_TO_MOVE,
    MOVE_TO_ID,
    VOCAB_SIZE,
    encode_position,
    legal_move_ids,
    mirror_move_id,
)
from .model import ChessFormer

SELECTION_MODES = ("policy", "value_1ply")


@dataclass
class MoveTrace:
    move: str
    win_prob: float
    latency_ms: float
    mode: str
    top_moves: List[Dict[str, Any]] = field(default_factory=list)
    raw_top1_is_legal: bool = True
    legal_move_count: int = 0
    flipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "move": self.move,
            "win_prob": round(self.win_prob, 6),
            "latency_ms": round(self.latency_ms, 3),
            "mode": self.mode,
            "top_moves": self.top_moves,
            "raw_top1_is_legal": bool(self.raw_top1_is_legal),
            "legal_move_count": int(self.legal_move_count),
            "flipped": bool(self.flipped),
        }


def _encode_batch(boards: Sequence[chess.Board], device: torch.device):
    pieces, metas, flips, legals = [], [], [], []
    for board in boards:
        p, m, flipped = encode_position(board)
        pieces.append(p)
        metas.append(m)
        flips.append(flipped)
        legals.append(legal_move_ids(board, flipped=flipped))
    piece_t = torch.from_numpy(np.stack(pieces)).long().to(device)
    meta_t = torch.from_numpy(np.stack(metas)).long().to(device)
    mask = torch.zeros((len(boards), VOCAB_SIZE), dtype=torch.bool, device=device)
    for row, ids in enumerate(legals):
        if ids.size:
            mask[row, torch.from_numpy(ids.astype(np.int64)).to(device)] = True
    return piece_t, meta_t, mask, flips, legals


@torch.no_grad()
def evaluate_boards(
    model: ChessFormer, boards: Sequence[chess.Board], device: torch.device
) -> Dict[str, torch.Tensor]:
    """Batched forward pass. Returns masked policy log-probs and win probs."""
    piece_t, meta_t, mask, flips, _ = _encode_batch(boards, device)
    out = model(piece_t, meta_t)
    logits = out["policy_logits"].float()
    masked = logits.masked_fill(~mask, float("-inf"))
    return {
        "logits": logits,
        "masked_logits": masked,
        "log_probs": F.log_softmax(masked, dim=-1),
        "win_prob": model.predict_win_prob(out["value_logits"]),
        "legal_mask": mask,
        "flips": flips,
    }


@torch.no_grad()
def select_move(
    model: ChessFormer,
    board: chess.Board,
    device: torch.device,
    *,
    mode: str = "policy",
    temperature: float = 0.0,
    top_k_trace: int = 5,
    value_candidates: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> MoveTrace:
    """Choose a move for ``board``. Raises if the position is terminal."""
    if mode not in SELECTION_MODES:
        raise ValueError(f"unknown selection mode {mode!r}; known: {SELECTION_MODES}")
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("no legal moves: position is terminal")

    started = time.perf_counter()
    was_training = model.training
    model.eval()
    try:
        result = evaluate_boards(model, [board], device)
        masked = result["masked_logits"][0]
        flipped = result["flips"][0]
        win_prob = float(result["win_prob"][0])

        raw_top1 = int(result["logits"][0].argmax())
        raw_top1_is_legal = bool(result["legal_mask"][0, raw_top1])

        k = min(max(1, top_k_trace), int(result["legal_mask"][0].sum()))
        top = torch.topk(masked, k=k)
        probs = F.softmax(masked, dim=-1)

        def to_uci(net_id: int) -> str:
            return ID_TO_MOVE[mirror_move_id(net_id) if flipped else net_id]

        top_moves = [
            {"move": to_uci(int(idx)), "prob": round(float(probs[idx]), 6)}
            for idx in top.indices.tolist()
        ]

        if temperature and temperature > 0.0:
            scaled = F.softmax(masked / float(temperature), dim=-1)
            generator = rng or np.random.default_rng()
            choice = int(generator.choice(VOCAB_SIZE, p=scaled.cpu().numpy().astype(np.float64)
                                          / float(scaled.sum())))
            chosen_id = choice
        else:
            chosen_id = int(masked.argmax())

        if mode == "value_1ply" and value_candidates > 1:
            chosen_id = _value_rerank(
                model, board, device, masked, flipped, value_candidates
            )

        move_uci = to_uci(chosen_id)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            # Structural guarantee, checked rather than assumed.
            raise RuntimeError(
                f"selected move {move_uci} is not legal in {board.fen()!r}; "
                "legality masking is broken"
            )

        return MoveTrace(
            move=move_uci,
            win_prob=win_prob,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            mode=mode,
            top_moves=top_moves,
            raw_top1_is_legal=raw_top1_is_legal,
            legal_move_count=len(legal),
            flipped=flipped,
        )
    finally:
        model.train(was_training)


@torch.no_grad()
def _value_rerank(
    model: ChessFormer,
    board: chess.Board,
    device: torch.device,
    masked_logits: torch.Tensor,
    flipped: bool,
    candidates: int,
) -> int:
    """Re-rank top policy candidates by the opponent's win prob after the move."""
    legal_count = int(torch.isfinite(masked_logits).sum())
    k = min(max(1, candidates), legal_count)
    top = torch.topk(masked_logits, k=k)

    net_ids = [int(i) for i in top.indices.tolist()]
    boards, kept = [], []
    for net_id in net_ids:
        uci = ID_TO_MOVE[mirror_move_id(net_id) if flipped else net_id]
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            continue
        child = board.copy(stack=False)
        child.push(move)
        if child.is_game_over():
            outcome = child.outcome()
            # Immediate mate is strictly best; nothing needs evaluating.
            if outcome is not None and outcome.winner is not None and outcome.winner == board.turn:
                return net_id
        boards.append(child)
        kept.append(net_id)

    if not boards:
        return int(masked_logits.argmax())

    child_eval = evaluate_boards(model, boards, device)
    # Child win prob is from the *child's* mover, i.e. our opponent.
    our_value = 1.0 - child_eval["win_prob"]
    policy_prior = F.softmax(masked_logits, dim=-1)[torch.tensor(kept, device=device)]
    score = our_value + 0.25 * policy_prior
    return kept[int(score.argmax())]


@torch.no_grad()
def play_game(
    model: ChessFormer,
    device: torch.device,
    *,
    opponent_move_fn=None,
    model_plays_white: bool = True,
    opening_moves: Sequence[str] = (),
    max_plies: int = 300,
    mode: str = "policy",
    temperature: float = 0.0,
    on_move=None,
) -> Dict[str, Any]:
    """Play one game. ``opponent_move_fn(board) -> uci`` drives the other side."""
    board = chess.Board()
    applied_opening: List[str] = []
    for uci in opening_moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break
        board.push(move)
        applied_opening.append(uci)

    moves: List[str] = []
    illegal_attempts = 0
    latencies: List[float] = []

    while not board.is_game_over(claim_draw=True) and len(moves) < max_plies:
        model_turn = (board.turn == chess.WHITE) == model_plays_white
        if model_turn:
            trace = select_move(model, board, device, mode=mode, temperature=temperature)
            latencies.append(trace.latency_ms)
            if not trace.raw_top1_is_legal:
                illegal_attempts += 1
            uci = trace.move
        else:
            if opponent_move_fn is None:
                break
            uci = opponent_move_fn(board)
            if uci is None:
                break
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break
        board.push(move)
        moves.append(uci)
        if on_move is not None:
            on_move({"fen": board.fen(), "move": uci, "ply": len(moves),
                     "model_turn": model_turn})

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result, score = "*", 0.5   # ply cap reached: scored as a draw
        termination = "ply_limit"
    else:
        result = outcome.result()
        termination = str(outcome.termination).split(".")[-1]
        if outcome.winner is None:
            score = 0.5
        else:
            score = 1.0 if (outcome.winner == chess.WHITE) == model_plays_white else 0.0

    return {
        "result": result,
        "model_score": score,
        "model_plays_white": model_plays_white,
        "termination": termination,
        "plies": len(moves),
        "moves": moves,
        "opening": applied_opening,
        "final_fen": board.fen(),
        "raw_illegal_top1_count": illegal_attempts,
        "mean_latency_ms": round(float(np.mean(latencies)), 3) if latencies else 0.0,
    }
