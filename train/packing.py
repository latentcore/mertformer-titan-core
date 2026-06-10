"""Shared, deterministic sequence packing + per-sequence identity.

TR: Bu modul hem `scripts/precompute_logits_topk.py` (teacher) hem de
    `train/train.py` (student) tarafindan kullanilir. Tek bir deterministik
    packer + kimlik (identity) hash'i, teacher logit'leri ile student token'larinin
    byte-byte hizali kalmasini GARANTI eder. Hizasizlik = sessiz KD bozulmasi;
    bu modul onu yapisal olarak imkansiz kilar (load-time hard-assert).

EN: This module is imported by BOTH `scripts/precompute_logits_topk.py` (teacher)
    and `train/train.py` (student). A single deterministic packer + a per-sequence
    identity hash GUARANTEES teacher logits and student tokens stay byte-aligned.
    Misalignment = silent KD corruption; this module makes it structurally
    impossible via a load-time hard assert.

Design notes:
- ONE text extractor (`extract_row_text`) so precompute and train can never skip
  rows differently.
- ONE tokenization (`encode_row`, `add_special_tokens=True`).
- Greedy packing with an EOS separator between rows, padded to ``max_seq_len``.
- Per-sequence identity = blake2b over the int32-LE bytes of ``input_ids[:true_len]``
  (trailing pad excluded), so it is stable across platforms.
- Import-light on purpose (torch only for the optional tensor path): safe to import
  from both the precompute script and the trainer without circular deps.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

PACK_FORMAT = "packed_v1"
TOPK_PACKED_FORMAT = "topk_packed_v1"
# Single source of truth for "what text does a JSONL row contribute".
TEXT_FIELD = "text"


class LogitAlignmentError(RuntimeError):
    """Raised when a stored teacher-logit shard does not match the re-derived
    student sequence (length or token hash). Never silently realign."""


def extract_row_text(obj: dict) -> str:
    """TR: Tek kaynak satir-metin cikarici. EN: Single-source row text extractor.

    Stage JSONL writer emits only ``{"text": ...}``; collapsing to ``text`` here
    (used by BOTH precompute and train) removes the historical
    ``text|content|instruction`` skip-predicate divergence.
    """
    if not isinstance(obj, dict):
        return ""
    return str(obj.get(TEXT_FIELD, "") or "").strip()


def encode_row(tokenizer, text: str, max_seq_len: int) -> List[int]:
    """Canonical per-row tokenization, identical for teacher and student.

    No padding here (packing handles it). ``add_special_tokens=True`` is pinned so
    teacher and student never drift on tokenizer defaults.
    """
    if not text:
        return []
    enc = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_len,
    )
    ids = enc["input_ids"]
    # Some tokenizers return tensors; normalize to a flat python list.
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    return [int(t) for t in ids]


def sequence_identity(input_ids: Sequence[int], true_len: int) -> dict:
    """Serializable identity stamped into a shard item and asserted at train load.

    Uses ``input_ids[:true_len]`` so trailing pad never affects the hash, and a
    canonical int32 little-endian encoding so it is tokenizer/platform stable.
    """
    n = max(0, int(true_len))
    real = [int(t) for t in list(input_ids)[:n]]
    packed = struct.pack(f"<{len(real)}i", *real) if real else b""
    return {"len": n, "hash": hashlib.blake2b(packed, digest_size=16).hexdigest()}


def assert_sequence_identity(input_ids: Sequence[int], true_len: int, stored: Optional[dict]) -> None:
    """Hard-fail if the re-derived student sequence disagrees with the stored shard.

    Raises ``LogitAlignmentError`` on a missing identity, a length mismatch, or a
    token-hash mismatch. This is the structural guard against silent KD corruption.
    """
    if not stored or "hash" not in stored:
        raise LogitAlignmentError(
            "Teacher-logit shard carries no per-sequence identity; refusing to align "
            "by position (silent realignment is exactly the corruption this guards). "
            "Re-run precompute with a build that stamps sequence identity."
        )
    current = sequence_identity(input_ids, true_len)
    if int(stored.get("len", -1)) != current["len"] or stored.get("hash") != current["hash"]:
        raise LogitAlignmentError(
            "Teacher/student sequence mismatch (no silent realign): "
            f"stored len={stored.get('len')} hash={stored.get('hash')} vs "
            f"student len={current['len']} hash={current['hash']}. "
            "Tokenizer / packing / row order drifted between precompute and train."
        )


def count_real_tokens(true_len: int) -> int:
    return max(0, int(true_len))


def _make_seq(
    ids: List[int],
    true_len: int,
    row_span: Tuple[int, int],
    seq_index: int,
    max_seq_len: int,
    pad_id: int,
    consumed_through: int,
) -> dict:
    input_ids = list(ids[:max_seq_len])
    if len(input_ids) < max_seq_len:
        input_ids = input_ids + [pad_id] * (max_seq_len - len(input_ids))
    tl = min(int(true_len), max_seq_len)
    return {
        "format": PACK_FORMAT,
        "input_ids": input_ids,
        "true_len": tl,
        "row_span": [int(row_span[0]), int(row_span[1])],
        "seq_index": int(seq_index),
        "identity": sequence_identity(input_ids, tl),
        "consumed_through": int(consumed_through),
    }


def iter_packed_sequences(
    rows: Iterable[Tuple[int, str]],
    tokenizer,
    max_seq_len: int,
    eos_id: int,
    pad_id: int,
) -> Iterator[dict]:
    """Deterministic greedy packer shared by teacher precompute and student train.

    ``rows`` yields ``(raw_line_index, text)`` pairs (raw_line_index lets the caller
    persist a resume point). Each yielded dict (see ``_make_seq``) is a packed
    sequence of exactly ``max_seq_len`` tokens with:
      - ``input_ids``: pad-filled to ``max_seq_len`` (only the last sequence pads)
      - ``true_len``: number of real (non-pad) tokens
      - ``row_span``: (first, last) raw line index contributing
      - ``identity``: hash of the real tokens
      - ``consumed_through``: highest raw line index fully emitted by the end of this
        sequence (safe resume point AFTER this sequence).

    Packing is a pure function of (rows, max_seq_len, eos_id, pad_id, tokenizer) —
    no RNG, no batch boundaries — so teacher and student streams are byte-identical.
    """
    buf: List[int] = []
    # (raw_line_index, cumulative_end_token_position) for rows touching the buffer.
    pending: List[Tuple[int, int]] = []
    emitted_tokens = 0
    cum = 0
    seq_index = 0
    last_consumed = -1

    def _consumed_through(emitted: int) -> int:
        nonlocal last_consumed
        c = last_consumed
        for line_idx, end_cum in pending:
            if end_cum <= emitted:
                c = max(c, line_idx)
        return c

    for line_idx, text in rows:
        text = (text or "").strip()
        if not text:
            continue
        ids = encode_row(tokenizer, text, max_seq_len)
        if not ids:
            continue
        piece = ids + [eos_id]

        # Oversized single row -> flush current buffer, then emit it truncated alone.
        if len(piece) > max_seq_len:
            if buf:
                first_row = pending[0][0] if pending else line_idx
                last_row = pending[-1][0] if pending else line_idx
                emitted_tokens += len(buf)
                yield _make_seq(buf, len(buf), (first_row, last_row), seq_index,
                                max_seq_len, pad_id, _consumed_through(emitted_tokens))
                seq_index += 1
                last_consumed = _consumed_through(emitted_tokens)
                buf = []
                pending = []
            trunc = piece[:max_seq_len]
            emitted_tokens += len(trunc)
            cum += len(piece)
            last_consumed = line_idx
            yield _make_seq(trunc, len(trunc), (line_idx, line_idx), seq_index,
                            max_seq_len, pad_id, line_idx)
            seq_index += 1
            continue

        buf.extend(piece)
        cum += len(piece)
        pending.append((line_idx, cum))

        while len(buf) >= max_seq_len:
            chunk = buf[:max_seq_len]
            emitted_tokens += max_seq_len
            consumed = _consumed_through(emitted_tokens)
            first_row = pending[0][0] if pending else line_idx
            last_row = pending[-1][0] if pending else line_idx
            yield _make_seq(chunk, max_seq_len, (first_row, last_row), seq_index,
                            max_seq_len, pad_id, consumed)
            seq_index += 1
            last_consumed = consumed
            buf = buf[max_seq_len:]
            # Drop fully-emitted rows from the pending window.
            pending = [(li, ec) for (li, ec) in pending if ec > emitted_tokens]

    if buf:
        emitted_tokens += len(buf)
        first_row = pending[0][0] if pending else last_consumed + 1
        last_row = pending[-1][0] if pending else first_row
        yield _make_seq(buf, len(buf), (first_row, last_row), seq_index,
                        max_seq_len, pad_id, last_row)
