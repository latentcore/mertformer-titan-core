"""Board tokenization, move vocabulary and value binning.

The move vocabulary is built exactly as
``vendor/upstream/scripts/chess_5080_onefile.py::build_move_vocab`` does, so
``MOVE_VOCAB`` (4208 entries) and ``MOVE_VOCAB_HASH`` match the upstream
checkpoint contract byte for byte.

SIDE-TO-MOVE CANONICALIZATION
-----------------------------
When it is Black's turn we mirror the board vertically and swap piece colours,
so the network always sees a "White to move" position. This is standard
practice (Leela, NNUE) and roughly halves the state space the model has to
cover -- a real win when the training budget is a laptop GPU rather than a TPU
pod. Encoding, the policy target and the decoded move all go through the same
mirror, and ``tests/test_board.py`` pins the round-trip.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import chess
import numpy as np

from .config import META_CARDINALITIES, NUM_META_TOKENS, NUM_SQUARE_TOKENS, SEQ_LEN

# ---------------------------------------------------------------------------
# Move vocabulary
# ---------------------------------------------------------------------------
_PROMO_PIECES = ("q", "r", "b", "n")
PROMO_TO_IDX: Dict[Optional[int], int] = {
    None: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}
NUM_PROMO_SLOTS = 5


def build_move_vocab() -> List[str]:
    """Every from!=to square pair, plus all legal promotion targets."""
    moves: List[str] = []
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq == to_sq:
                continue
            moves.append(chess.square_name(from_sq) + chess.square_name(to_sq))

    promos: List[str] = []
    for file_idx in range(8):
        white_from = chess.square(file_idx, 6)
        black_from = chess.square(file_idx, 1)
        for delta in (-1, 0, 1):
            to_file = file_idx + delta
            if 0 <= to_file < 8:
                white_to = chess.square(to_file, 7)
                black_to = chess.square(to_file, 0)
                for promo in _PROMO_PIECES:
                    promos.append(chess.square_name(white_from) + chess.square_name(white_to) + promo)
                    promos.append(chess.square_name(black_from) + chess.square_name(black_to) + promo)

    ordered: List[str] = []
    seen = set()
    for item in moves + promos:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


MOVE_VOCAB: List[str] = build_move_vocab()
MOVE_TO_ID: Dict[str, int] = {uci: idx for idx, uci in enumerate(MOVE_VOCAB)}
ID_TO_MOVE: Dict[int, str] = {idx: uci for uci, idx in MOVE_TO_ID.items()}
VOCAB_SIZE: int = len(MOVE_VOCAB)
MOVE_VOCAB_HASH: str = hashlib.sha256("\n".join(MOVE_VOCAB).encode("utf-8")).hexdigest()

# Factorized-head lookup tables: move id -> (from square, to square, promo slot).
MOVE_FROM_SQ = np.zeros(VOCAB_SIZE, dtype=np.int64)
MOVE_TO_SQ = np.zeros(VOCAB_SIZE, dtype=np.int64)
MOVE_PROMO = np.zeros(VOCAB_SIZE, dtype=np.int64)
for _idx, _uci in enumerate(MOVE_VOCAB):
    _m = chess.Move.from_uci(_uci)
    MOVE_FROM_SQ[_idx] = _m.from_square
    MOVE_TO_SQ[_idx] = _m.to_square
    MOVE_PROMO[_idx] = PROMO_TO_IDX[_m.promotion]

# Vertical mirror of every move id, used by the perspective flip.
MOVE_MIRROR = np.zeros(VOCAB_SIZE, dtype=np.int64)
for _idx, _uci in enumerate(MOVE_VOCAB):
    _m = chess.Move.from_uci(_uci)
    _mirrored = chess.Move(
        chess.square_mirror(_m.from_square),
        chess.square_mirror(_m.to_square),
        promotion=_m.promotion,
    )
    MOVE_MIRROR[_idx] = MOVE_TO_ID[_mirrored.uci()]


def mirror_move_id(move_id: int) -> int:
    return int(MOVE_MIRROR[int(move_id)])


# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------
_MATERIAL_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def piece_to_id(piece: Optional[chess.Piece]) -> int:
    """0 = empty, 1..6 = white P..K, 7..12 = black P..K (upstream mapping)."""
    if piece is None:
        return 0
    offset = 0 if piece.color == chess.WHITE else 6
    return offset + piece.piece_type


def material_bucket(board: chess.Board, color: bool) -> int:
    score = 0
    for piece_type, value in _MATERIAL_VALUES.items():
        score += len(board.pieces(piece_type, color)) * value
    return min(39, score)


def encode_board_state(
    board: chess.Board, legal_move_count: Optional[int] = None
) -> Tuple[List[int], List[int]]:
    """Upstream-compatible encoding: no perspective flip, absolute colours.

    Kept for the compat surface. The training/inference path uses
    :func:`encode_position`.
    """
    if legal_move_count is None:
        legal_move_count = board.legal_moves.count()
    piece_ids = [piece_to_id(board.piece_at(sq)) for sq in chess.SQUARES]
    ep_square = board.ep_square
    ep_file = 0 if ep_square is None else chess.square_file(ep_square) + 1
    meta_ids = [
        int(board.turn),
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
        ep_file,
        min(15, board.halfmove_clock // 4),
        min(31, board.fullmove_number // 2),
        int(board.is_check()),
        min(31, legal_move_count // 2),
        material_bucket(board, chess.WHITE),
        material_bucket(board, chess.BLACK),
    ]
    return piece_ids, meta_ids


# The Lichess eval database stores 4-field FENs (placement, side, castling,
# en passant) -- no halfmove clock and no fullmove number. Every training
# position therefore has clock buckets of 0. If we fed *real* clocks at
# inference time the model would see meta slots 6 and 7 in a range it never
# encountered during training, which is a silent train/serve skew. We neutralize
# both slots on both paths instead; ``include_clocks=True`` exists only for
# datasets that actually carry them.
INCLUDE_CLOCKS_DEFAULT = False
_HALFMOVE_SLOT = 6
_FULLMOVE_SLOT = 7


def encode_position(
    board: chess.Board,
    legal_move_count: Optional[int] = None,
    canonical_perspective: bool = True,
    include_clocks: bool = INCLUDE_CLOCKS_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Encode a position, optionally from the side-to-move's perspective.

    Returns ``(piece_ids[64] int8, meta_ids[12] int8, flipped)``. When
    ``flipped`` is True every move id must be passed through
    :func:`mirror_move_id` to move between board space and network space.
    """
    flipped = bool(canonical_perspective and board.turn == chess.BLACK)
    view = board.mirror() if flipped else board
    # ``Board.mirror()`` swaps colours and flips ranks, so ``view.turn`` is
    # WHITE whenever we flipped; legality counts are identical either way.
    if legal_move_count is None:
        legal_move_count = board.legal_moves.count()

    piece_ids = np.fromiter(
        (piece_to_id(view.piece_at(sq)) for sq in chess.SQUARES),
        dtype=np.int8,
        count=NUM_SQUARE_TOKENS,
    )
    ep_square = view.ep_square
    ep_file = 0 if ep_square is None else chess.square_file(ep_square) + 1
    meta = np.array(
        [
            int(view.turn),
            int(view.has_kingside_castling_rights(chess.WHITE)),
            int(view.has_queenside_castling_rights(chess.WHITE)),
            int(view.has_kingside_castling_rights(chess.BLACK)),
            int(view.has_queenside_castling_rights(chess.BLACK)),
            ep_file,
            min(15, view.halfmove_clock // 4) if include_clocks else 0,
            min(31, view.fullmove_number // 2) if include_clocks else 0,
            int(view.is_check()),
            min(31, int(legal_move_count) // 2),
            material_bucket(view, chess.WHITE),
            material_bucket(view, chess.BLACK),
        ],
        dtype=np.int8,
    )
    return piece_ids, meta, flipped


def legal_move_ids(board: chess.Board, flipped: bool = False) -> np.ndarray:
    """Legal move ids in *network* space (mirrored when ``flipped``)."""
    ids: List[int] = []
    for move in board.legal_moves:
        mid = MOVE_TO_ID.get(move.uci())
        if mid is None:
            raise RuntimeError(f"legal move outside vocabulary: {move.uci()}")
        ids.append(MOVE_MIRROR[mid] if flipped else mid)
    return np.asarray(sorted(ids), dtype=np.int16)


def position_key(board: chess.Board) -> bytes:
    """Dedup key: piece placement + side + castling + ep, ignoring clocks."""
    return " ".join(board.fen().split(" ")[:4]).encode("utf-8")


# ---------------------------------------------------------------------------
# Value: centipawns -> win probability -> discretized bins
# ---------------------------------------------------------------------------
# Lichess' own centipawn -> win% mapping. Keeping their constant means the
# targets we train on are on the same scale as the evals in their database.
_WIN_PCT_K = 0.00368208
# Centipawn clamp. At |cp| = 2500 the win probability is ~0.99989, which already
# lands in the outermost of 128 bins; clamping here (rather than at a mate
# score) keeps the value strictly inside (0, 1) so the HL-Gauss target never
# degenerates and the loss stays finite.
CP_CLAMP = 2500.0
PROB_EPS = 1e-4


def cp_to_win_prob(cp: float) -> float:
    """Centipawns (side-to-move POV) -> win probability, strictly inside (0, 1)."""
    cp = max(-CP_CLAMP, min(CP_CLAMP, float(cp)))
    p = 1.0 / (1.0 + math.exp(-_WIN_PCT_K * cp))
    return min(1.0 - PROB_EPS, max(PROB_EPS, p))


def mate_to_win_prob(mate_in: int) -> float:
    """Forced mate -> saturated win probability, sign following ``mate_in``."""
    if mate_in == 0:
        return 0.5
    # Deeper mates are marginally less certain in a bounded-depth search, but
    # they are all overwhelmingly won; clamp near the ends rather than at them
    # so the HL-Gauss target keeps a little mass inside the range.
    magnitude = 0.995 - min(0.045, 0.0015 * (abs(mate_in) - 1))
    return magnitude if mate_in > 0 else 1.0 - magnitude


def win_prob_to_bin(win_prob: float, num_bins: int) -> int:
    idx = int(win_prob * num_bins)
    return max(0, min(num_bins - 1, idx))


def bin_centers(num_bins: int) -> np.ndarray:
    return (np.arange(num_bins, dtype=np.float32) + 0.5) / float(num_bins)


def hl_gauss_target(
    win_prob: float | np.ndarray, num_bins: int, sigma_ratio: float = 0.75
) -> np.ndarray:
    """HL-Gauss (arXiv:2403.03950) target: a Gaussian over bins, not one-hot.

    Mass is assigned by the difference of the Gaussian CDF at each bin edge,
    then renormalized, so the distribution stays proper even at the extremes.
    """
    win_prob = np.atleast_1d(np.asarray(win_prob, dtype=np.float64))
    edges = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)
    sigma = max(1e-6, sigma_ratio / float(num_bins))
    z = (edges[None, :] - win_prob[:, None]) / (sigma * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z))
    probs = np.diff(cdf, axis=1)
    total = probs.sum(axis=1, keepdims=True)
    # Degenerate only if the Gaussian is numerically entirely outside [0,1];
    # fall back to a one-hot on the nearest bin in that case.
    bad = (total <= 1e-12).ravel()
    if bad.any():
        for i in np.flatnonzero(bad):
            probs[i] = 0.0
            probs[i, win_prob_to_bin(float(win_prob[i]), num_bins)] = 1.0
            total[i] = 1.0
    return (probs / total).astype(np.float32)


def win_prob_to_wdl_class(win_prob: float, draw_margin: float = 0.10) -> int:
    """0 = loss, 1 = draw, 2 = win, from the side-to-move's perspective."""
    if win_prob <= 0.5 - draw_margin:
        return 0
    if win_prob >= 0.5 + draw_margin:
        return 2
    return 1


__all__ = [
    "MOVE_VOCAB",
    "MOVE_TO_ID",
    "ID_TO_MOVE",
    "VOCAB_SIZE",
    "MOVE_VOCAB_HASH",
    "MOVE_FROM_SQ",
    "MOVE_TO_SQ",
    "MOVE_PROMO",
    "MOVE_MIRROR",
    "NUM_PROMO_SLOTS",
    "build_move_vocab",
    "mirror_move_id",
    "piece_to_id",
    "material_bucket",
    "encode_board_state",
    "encode_position",
    "legal_move_ids",
    "position_key",
    "cp_to_win_prob",
    "mate_to_win_prob",
    "win_prob_to_bin",
    "bin_centers",
    "hl_gauss_target",
    "win_prob_to_wdl_class",
    "SEQ_LEN",
    "NUM_META_TOKENS",
    "NUM_SQUARE_TOKENS",
    "META_CARDINALITIES",
]
