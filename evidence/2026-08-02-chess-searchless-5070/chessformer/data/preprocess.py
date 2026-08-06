"""Turn ``lichess_db_eval.jsonl.zst`` into packed memmap shards.

WHY PACKED SHARDS
-----------------
``scripts/chess_5080_onefile.py`` held the whole dataset as a Python list of
``ChessExample`` dataclasses, each carrying several Python ``list[int]``
objects. At its own ``strength_4060_24h`` setting (2.4 M positions) that is
millions of live Python objects: gigabytes of interpreter overhead, constant GC
pressure, and ``python-chess`` calls on the training hot path. That is the
"veri akisi zor / islemler arasi bekleme" symptom.

Here every position is decoded exactly once, offline, into flat ``numpy``
arrays that the trainer opens with ``mmap_mode='r'``. Training touches no
Python objects and no chess library at all.

RECORD FORMAT (verified against the live file, see data card)
-------------------------------------------------------------
``{"fen": <4-field FEN>, "evals": [{"knodes": int, "depth": int,
   "pvs": [{"cp"|"mate": int, "line": "<uci> <uci> ..."}]}]}``

* FENs carry only 4 fields -- no clocks (see ``board.INCLUDE_CLOCKS_DEFAULT``).
* ``cp``/``mate`` are **White-relative**. Determined empirically, not assumed:
  over 3,189 sampled positions with a material imbalance >= 2 and |cp| >= 60,
  the White-POV reading agrees with the material sign 69.1% of the time while
  the side-to-move reading agrees 49.7% (chance). We negate for Black to move.
* ``pvs`` are ordered best-first *for the side to move*.
"""
from __future__ import annotations

import io
import json
import math
import time
from dataclasses import dataclass, field
from hashlib import blake2b
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import chess
import numpy as np

from ..board import (
    MOVE_TO_ID,
    VOCAB_SIZE,
    cp_to_win_prob,
    encode_position,
    legal_move_ids,
    mate_to_win_prob,
    mirror_move_id,
    win_prob_to_wdl_class,
)
from ..config import NUM_META_TOKENS, NUM_SQUARE_TOKENS, write_json

SHARD_PREFIX = "shard_"
VALUE_LEVELS = 1024  # win-prob quantization stored per position (uint16)
MAX_PV_SLOTS = 4     # soft policy target width
SPLIT_BUCKETS = 10_000
SPLITS = ("train", "val", "test")

ProgressFn = Callable[[Dict[str, object]], None]


def position_bucket(key: bytes) -> int:
    return int.from_bytes(blake2b(key, digest_size=8).digest(), "big") % SPLIT_BUCKETS


def make_split_fn(
    train_positions: int, val_positions: int, test_positions: int
) -> Tuple[Callable[[bytes], str], Dict[str, object]]:
    """Deterministic, memory-free split assignment keyed on the position itself.

    Hashing the position (rather than shuffling rows) means a duplicate FEN can
    never straddle two splits, so the holdout stays clean without holding a
    30M-entry dedup set in RAM.

    Bucket widths are proportional to the requested split sizes, so a 20k-row
    smoke dataset gets a usable validation split just like a 32M-row real one.
    """
    total = max(1, int(train_positions) + int(val_positions) + int(test_positions))
    test_max = max(1, round(SPLIT_BUCKETS * int(test_positions) / total))
    val_max = test_max + max(1, round(SPLIT_BUCKETS * int(val_positions) / total))
    val_max = min(val_max, SPLIT_BUCKETS - 1)

    def split_for_key(key: bytes) -> str:
        bucket = position_bucket(key)
        if bucket < test_max:
            return "test"
        if bucket < val_max:
            return "val"
        return "train"

    policy = {
        "method": f"blake2b(position_key) % {SPLIT_BUCKETS}",
        "test_buckets": [0, test_max],
        "val_buckets": [test_max, val_max],
        "train_buckets": [val_max, SPLIT_BUCKETS],
        "note": "hash-based so a duplicated FEN always lands in the same split",
    }
    return split_for_key, policy


def split_for_key(key: bytes) -> str:
    """Default 99 / 0.5 / 0.5 assignment (used by tests and ad-hoc checks)."""
    fn, _ = make_split_fn(99_000, 500, 500)
    return fn(key)


@dataclass
class ParsedPosition:
    piece_ids: np.ndarray
    meta: np.ndarray
    pv_ids: np.ndarray       # int16, -1 padded, network space
    pv_weights: np.ndarray   # float32, sums to 1 over used slots
    legal_ids: np.ndarray    # int16, network space
    win_prob: float
    depth: int
    key: bytes


@dataclass
class ParseStats:
    lines_read: int = 0
    json_errors: int = 0
    fen_errors: int = 0
    dropped_low_depth: int = 0
    dropped_no_eval: int = 0
    dropped_no_legal_pv: int = 0
    dropped_terminal: int = 0
    accepted: int = 0
    multi_pv_positions: int = 0
    mate_positions: int = 0
    depth_sum: int = 0
    legal_moves_sum: int = 0
    flipped: int = 0

    def to_dict(self) -> Dict[str, object]:
        acc = max(1, self.accepted)
        return {
            "lines_read": self.lines_read,
            "accepted": self.accepted,
            "json_errors": self.json_errors,
            "fen_errors": self.fen_errors,
            "dropped_low_depth": self.dropped_low_depth,
            "dropped_no_eval": self.dropped_no_eval,
            "dropped_no_legal_pv": self.dropped_no_legal_pv,
            "dropped_terminal": self.dropped_terminal,
            "multi_pv_positions": self.multi_pv_positions,
            "mate_positions": self.mate_positions,
            "mean_depth": round(self.depth_sum / acc, 3),
            "mean_legal_moves": round(self.legal_moves_sum / acc, 3),
            "flipped_fraction": round(self.flipped / acc, 4),
        }


def _pv_score_white_pov(pv: Dict[str, object]) -> Optional[Tuple[float, bool]]:
    """(white-POV win probability, is_mate) or None if the pv has no score."""
    if "mate" in pv and pv["mate"] is not None:
        return mate_to_win_prob(int(pv["mate"])), True
    if "cp" in pv and pv["cp"] is not None:
        return cp_to_win_prob(float(pv["cp"])), False
    return None


def parse_record(
    record: Dict[str, object],
    *,
    min_depth: int,
    max_pvs: int,
    soft_tau: float,
    stats: ParseStats,
) -> Optional[ParsedPosition]:
    fen = record.get("fen")
    if not isinstance(fen, str):
        stats.fen_errors += 1
        return None
    try:
        board = chess.Board(fen)
    except Exception:
        stats.fen_errors += 1
        return None

    evals = record.get("evals") or []
    if not isinstance(evals, list) or not evals:
        stats.dropped_no_eval += 1
        return None

    # Deepest analysis wins; knodes breaks ties.
    best = max(evals, key=lambda e: (int(e.get("depth", 0) or 0), int(e.get("knodes", 0) or 0)))
    depth = int(best.get("depth", 0) or 0)
    if depth < min_depth:
        stats.dropped_low_depth += 1
        return None

    pvs = best.get("pvs") or []
    if not isinstance(pvs, list) or not pvs:
        stats.dropped_no_eval += 1
        return None

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        stats.dropped_terminal += 1
        return None

    white_to_move = board.turn == chess.WHITE
    legal_uci = {m.uci() for m in legal_moves}

    candidates: List[Tuple[int, float]] = []  # (board-space move id, stm win prob)
    saw_mate = False
    for pv in pvs[: max(1, max_pvs)]:
        if not isinstance(pv, dict):
            continue
        line = pv.get("line")
        if not isinstance(line, str) or not line:
            continue
        first = line.split(" ", 1)[0]
        if first not in legal_uci:
            continue
        scored = _pv_score_white_pov(pv)
        if scored is None:
            continue
        wp_white, is_mate = scored
        saw_mate = saw_mate or is_mate
        # White-relative -> side-to-move relative.
        wp_stm = wp_white if white_to_move else 1.0 - wp_white
        move_id = MOVE_TO_ID.get(first)
        if move_id is None:
            continue
        candidates.append((move_id, wp_stm))

    if not candidates:
        stats.dropped_no_legal_pv += 1
        return None

    # Deduplicate move ids, keeping the best score for each.
    best_by_move: Dict[int, float] = {}
    for move_id, wp in candidates:
        if move_id not in best_by_move or wp > best_by_move[move_id]:
            best_by_move[move_id] = wp
    items = sorted(best_by_move.items(), key=lambda kv: -kv[1])[:MAX_PV_SLOTS]

    # Soft target: exponential in win-probability loss relative to the best pv.
    # At the default tau=0.03 a ~5cp-worse move keeps ~86% of the best move's
    # weight while a ~100cp-worse move falls to ~5%, so near-equal alternatives
    # stay informative without teaching the model to like blunders.
    top_wp = items[0][1]
    tau = max(1e-3, float(soft_tau))
    raw = np.array([math.exp((wp - top_wp) / tau) for _, wp in items], dtype=np.float64)
    weights = (raw / raw.sum()).astype(np.float32)

    piece_ids, meta, flipped = encode_position(board, legal_move_count=len(legal_moves))
    legal = legal_move_ids(board, flipped=flipped)

    pv_ids = np.full(MAX_PV_SLOTS, -1, dtype=np.int16)
    pv_w = np.zeros(MAX_PV_SLOTS, dtype=np.float32)
    for slot, ((move_id, _wp), weight) in enumerate(zip(items, weights)):
        net_id = mirror_move_id(move_id) if flipped else move_id
        pv_ids[slot] = np.int16(net_id)
        pv_w[slot] = weight

    stats.accepted += 1
    stats.depth_sum += depth
    stats.legal_moves_sum += len(legal)
    if len(items) > 1:
        stats.multi_pv_positions += 1
    if saw_mate:
        stats.mate_positions += 1
    if flipped:
        stats.flipped += 1

    return ParsedPosition(
        piece_ids=piece_ids,
        meta=meta,
        pv_ids=pv_ids,
        pv_weights=pv_w,
        legal_ids=legal,
        win_prob=float(top_wp),
        depth=depth,
        key=" ".join(fen.split(" ")[:4]).encode("utf-8"),
    )


def iter_records(path: Path, *, encoding: str = "utf-8") -> Iterator[Dict[str, object]]:
    """Stream JSON records out of a (possibly truncated) .zst file."""
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    with Path(path).open("rb") as raw:
        try:
            reader = dctx.stream_reader(raw, read_across_frames=True)
        except TypeError:  # older zstandard
            reader = dctx.stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding=encoding, errors="ignore")
        while True:
            try:
                line = text.readline()
            except (OSError, ValueError, zstd.ZstdError):
                # Expected when the prefix ends mid-frame.
                return
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A truncated final line is normal for a prefix download.
                return


class ShardWriter:
    """Accumulates positions and flushes fixed-size shards to disk."""

    def __init__(self, root: Path, split: str, shard_size: int) -> None:
        self.root = Path(root) / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_size = int(shard_size)
        self.shard_index = 0
        self.total = 0
        self.shards: List[Dict[str, object]] = []
        self._reset()

    def _reset(self) -> None:
        self._pieces: List[np.ndarray] = []
        self._meta: List[np.ndarray] = []
        self._pv_ids: List[np.ndarray] = []
        self._pv_w: List[np.ndarray] = []
        self._legal: List[np.ndarray] = []
        self._value: List[int] = []
        self._wdl: List[int] = []
        self._depth: List[int] = []

    def add(self, pos: ParsedPosition) -> None:
        self._pieces.append(pos.piece_ids)
        self._meta.append(pos.meta)
        self._pv_ids.append(pos.pv_ids)
        self._pv_w.append(pos.pv_weights)
        self._legal.append(pos.legal_ids)
        level = int(min(VALUE_LEVELS - 1, max(0, round(pos.win_prob * (VALUE_LEVELS - 1)))))
        self._value.append(level)
        self._wdl.append(win_prob_to_wdl_class(pos.win_prob))
        self._depth.append(pos.depth)
        if len(self._value) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        n = len(self._value)
        if n == 0:
            return
        d = self.root / f"{SHARD_PREFIX}{self.shard_index:05d}"
        d.mkdir(parents=True, exist_ok=True)

        np.save(d / "pieces.npy", np.stack(self._pieces).astype(np.int8))
        np.save(d / "meta.npy", np.stack(self._meta).astype(np.int8))
        np.save(d / "pv_ids.npy", np.stack(self._pv_ids).astype(np.int16))
        np.save(d / "pv_w.npy", np.stack(self._pv_w).astype(np.float32))
        np.save(d / "value_level.npy", np.asarray(self._value, dtype=np.uint16))
        np.save(d / "wdl.npy", np.asarray(self._wdl, dtype=np.int8))
        np.save(d / "depth.npy", np.asarray(self._depth, dtype=np.uint8))

        lengths = np.asarray([len(x) for x in self._legal], dtype=np.int64)
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        np.save(d / "legal_offsets.npy", offsets)
        np.save(d / "legal_ids.npy", np.concatenate(self._legal).astype(np.int16))

        size_bytes = sum(p.stat().st_size for p in d.glob("*.npy"))
        write_json(d / "shard_meta.json", {
            "split": self.split,
            "index": self.shard_index,
            "positions": n,
            "bytes": int(size_bytes),
            "bytes_per_position": round(size_bytes / n, 2),
        })
        self.shards.append({"path": str(d), "positions": n, "bytes": int(size_bytes)})
        self.total += n
        self.shard_index += 1
        self._reset()

    def close(self) -> Dict[str, object]:
        self.flush()
        return {
            "split": self.split,
            "positions": self.total,
            "shards": len(self.shards),
            "bytes": int(sum(int(s["bytes"]) for s in self.shards)),
            "shard_list": self.shards,
        }


def build_dataset(
    source: Path,
    out_dir: Path,
    *,
    target_positions: int,
    min_depth: int = 12,
    max_pvs: int = 4,
    soft_tau: float = 0.05,
    shard_size: int = 1_000_000,
    val_positions: int = 100_000,
    test_positions: int = 100_000,
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, object]:
    """Stream ``source`` and write packed shards under ``out_dir``.

    Returns the manifest that becomes the data card. Split membership is
    decided by hashing the position, so val/test can never leak into train.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writers = {
        "train": ShardWriter(out_dir, "train", shard_size),
        "val": ShardWriter(out_dir, "val", max(1, min(shard_size, val_positions))),
        "test": ShardWriter(out_dir, "test", max(1, min(shard_size, test_positions))),
    }
    caps = {"train": int(target_positions), "val": int(val_positions), "test": int(test_positions)}
    split_fn, split_policy = make_split_fn(target_positions, val_positions, test_positions)
    stats = ParseStats()
    started = time.time()
    stopped_early = False

    for record in iter_records(source):
        stats.lines_read += 1
        if should_stop is not None and should_stop():
            stopped_early = True
            break

        pos = parse_record(
            record, min_depth=min_depth, max_pvs=max_pvs, soft_tau=soft_tau, stats=stats
        )
        if pos is None:
            continue

        split = split_fn(pos.key)
        writer = writers[split]
        if writer.total + len(writer._value) >= caps[split]:
            if all(w.total + len(w._value) >= caps[s] for s, w in writers.items()):
                break
            continue
        writer.add(pos)

        if progress is not None and stats.lines_read % 20000 == 0:
            elapsed = max(1e-6, time.time() - started)
            progress({
                "stage": "preprocess",
                "lines_read": stats.lines_read,
                "accepted": stats.accepted,
                "train": writers["train"].total + len(writers["train"]._value),
                "val": writers["val"].total + len(writers["val"]._value),
                "test": writers["test"].total + len(writers["test"]._value),
                "target": caps["train"],
                "positions_per_sec": round(stats.accepted / elapsed, 1),
                "elapsed_sec": round(elapsed, 1),
            })

    summary = {split: writers[split].close() for split in SPLITS}
    elapsed = time.time() - started

    manifest = {
        "schema": "chessformer_dataset_manifest_v1",
        "source": str(source),
        "out_dir": str(out_dir),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(elapsed, 2),
        "stopped_early": stopped_early,
        "parameters": {
            "target_positions": int(target_positions),
            "min_depth": int(min_depth),
            "max_pvs": int(max_pvs),
            "soft_tau": float(soft_tau),
            "shard_size": int(shard_size),
            "val_positions": int(val_positions),
            "test_positions": int(test_positions),
            "value_levels": VALUE_LEVELS,
            "max_pv_slots": MAX_PV_SLOTS,
        },
        "split_policy": split_policy,
        "parse_stats": stats.to_dict(),
        "splits": summary,
        "positions_total": sum(int(summary[s]["positions"]) for s in SPLITS),
        "bytes_total": sum(int(summary[s]["bytes"]) for s in SPLITS),
        "cp_perspective": {
            "value": "white_relative",
            "determined_by": "empirical material-sign correlation over 3189 sampled positions",
            "white_pov_agreement": 0.6911,
            "side_to_move_agreement": 0.4967,
        },
        "known_limitations": [
            "Source FENs carry only 4 fields, so halfmove/fullmove clocks are absent; "
            "meta slots 6 and 7 are forced to 0 on BOTH the training and inference paths "
            "(board.INCLUDE_CLOCKS_DEFAULT) to avoid train/serve skew.",
            "The model therefore has no 50-move-rule or repetition awareness from this data.",
        ],
    }
    write_json(out_dir / "dataset_manifest.json", manifest)
    return manifest
