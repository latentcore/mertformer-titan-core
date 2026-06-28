#!/usr/bin/env python3
"""
MertFormer Titan Phase-0 offline teacher logit precomputation.

This Phase-0 lane computes Top-K teacher logits once, stores them in compact
shards, and lets the training path reconstruct dense tensors lazily during KD.
The canonical offline_clean lane depends on these shards unless a separate
gated-teacher run is intentionally chosen.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("precompute_topk")

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent
for _ in range(4):
    if (_ROOT / "config").exists():
        break
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

try:
    from config.config import cfg  # type: ignore
except Exception as exc:  # pragma: no cover - fatal bootstrap path
    logger.error("Config import failed: %s", exc)
    sys.exit(1)

STAGE_FILES = {
    1: _ROOT / "datasets" / "stage1" / "stage1_data.jsonl",
    2: _ROOT / "datasets" / "stage2" / "stage2_data.jsonl",
    3: _ROOT / "datasets" / "stage3" / "stage3_data.jsonl",
    4: _ROOT / "datasets" / "stage4_soul" / "stage4_data.jsonl",
    5: _ROOT / "datasets" / "stage5_tools" / "stage5_data.jsonl",
}

DEFAULT_TOP_K = 256
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_SEQ = int(getattr(cfg, "max_seq_len", 512))
SUBSET = "train"


def _state_path(logits_dir: Path, stage_name: str) -> Path:
    return logits_dir / f"{stage_name}_{SUBSET}_topk_state.json"


def _stage_pattern(stage_name: str) -> str:
    return f"{stage_name}_{SUBSET}_part_*.pt"


def _shard_part_index(path: Path) -> int:
    """Integer part index parsed from a shard filename's ``..._part_<N>.pt`` tail."""
    try:
        return int(path.stem.rsplit("_part_", 1)[-1])
    except Exception:
        return -1


def _stage_shards(logits_dir: Path, stage_name: str) -> list[Path]:
    # Sort by INTEGER part index, mirroring the train reader
    # (orchestrator.distillation_manager._list_logits_files / _part_index).
    # A lexicographic sort would mis-order at >=11 single-process shards
    # (part_10 < part_2) and, for the parallel lane named by first-seq-index,
    # at >=6 blocks (part_10000 < part_2000) — which would make the alignment
    # validator false-FAIL a byte-correct shard set. The integer key keeps the
    # validator's read order identical to what training actually consumes.
    return sorted(logits_dir.glob(_stage_pattern(stage_name)), key=_shard_part_index)


def _unwrap_payload_for_count(payload) -> int:
    if isinstance(payload, dict) and "logits" in payload:
        payload = payload["logits"]
    if isinstance(payload, (list, tuple)):
        return len(payload)
    return 0


def _load_resume_state(logits_dir: Path, stage_name: str) -> dict:
    """TR: [B2] Resume durumu: HAM SATIR indexine gore (lines_consumed) + uretilen
    packed dizi sayisi. Eski 'done_samples' semasi geriye-uyumlu okunur.
    EN: [B2] Resume state keyed by RAW LINE INDEX (lines_consumed) + emitted packed
    sequence count. The old 'done_samples' schema is read back-compat."""
    state_path = _state_path(logits_dir, stage_name)
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            lines_consumed = int(data.get("lines_consumed", data.get("done_samples", 0)))
            sequences_emitted = int(data.get("sequences_emitted", 0))
            return {"lines_consumed": max(0, lines_consumed), "sequences_emitted": max(0, sequences_emitted)}
        except Exception:
            pass
    return {"lines_consumed": 0, "sequences_emitted": 0}


def _save_resume_state(logits_dir: Path, stage_name: str, lines_consumed: int, sequences_emitted: int) -> None:
    state_path = _state_path(logits_dir, stage_name)
    payload = {
        "lines_consumed": int(max(0, lines_consumed)),
        "sequences_emitted": int(max(0, sequences_emitted)),
        # back-compat mirror for older readers (titan_preflight)
        "done_samples": int(max(0, lines_consumed)),
    }
    state_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _load_done_samples(logits_dir: Path, stage_name: str) -> int:
    """Back-compat: raw lines consumed (was 'done_samples')."""
    state_path = _state_path(logits_dir, stage_name)
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            return int(data.get("lines_consumed", data.get("done_samples", 0)))
        except Exception as exc:
            logger.warning("resume state parse failed for %s, falling back to shard count: %s", state_path, exc)

    total = 0
    for shard in _stage_shards(logits_dir, stage_name):
        try:
            payload = torch.load(shard, map_location="cpu", weights_only=False)
            total += _unwrap_payload_for_count(payload)
        except Exception as exc:
            logger.warning("skipping unreadable/corrupt shard %s during count: %s", shard, exc)
            continue
    return total


def _next_chunk_idx(logits_dir: Path, stage_name: str) -> int:
    indices: list[int] = []
    for shard in _stage_shards(logits_dir, stage_name):
        try:
            indices.append(int(shard.stem.rsplit("_part_", 1)[-1]))
        except Exception as exc:
            logger.debug("skipping shard with unparseable index %s: %s", shard, exc)
            continue
    return (max(indices) + 1) if indices else 0


def _estimate_disk_mb(n_samples: int, seq_len: int, top_k: int) -> float:
    bytes_per_sample = seq_len * top_k * 6
    return (n_samples * bytes_per_sample) / (1024 ** 2)


def _count_jsonl(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except Exception as exc:
        logger.warning("failed to count jsonl lines for %s: %s; returning 0", path, exc)
        return 0


def _stage_complete(logits_dir: Path, stage_name: str, total_samples: int) -> bool:
    if total_samples <= 0:
        return False
    done_samples = _load_done_samples(logits_dir, stage_name)
    return done_samples >= total_samples and bool(_stage_shards(logits_dir, stage_name))


def load_teacher(model_id: str, hf_token: Optional[str], device_map: str = "auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading teacher model: %s", model_id)
    load_kwargs = {
        "device_map": device_map,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "trust_remote_code": True,
    }
    if hf_token:
        load_kwargs["token"] = hf_token

    try:
        import bitsandbytes  # noqa: F401

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Using 4-bit NF4 teacher load path.")
    except ImportError:
        logger.warning("bitsandbytes unavailable; falling back to BF16 teacher load.")

    teacher = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    teacher.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Teacher model loaded successfully.")
    return teacher, tokenizer


@torch.no_grad()
def extract_topk_logits(
    teacher,
    input_ids: torch.Tensor,
    top_k: int,
    teacher_device,
) -> list[dict[str, torch.Tensor]]:
    ids = input_ids.to(teacher_device)
    outputs = teacher(ids)
    logits = outputs.logits
    del outputs

    samples: list[dict[str, torch.Tensor]] = []
    for batch_idx in range(logits.size(0)):
        sample_logits = logits[batch_idx]
        topk_values, topk_indices = torch.topk(sample_logits.float(), k=top_k, dim=-1)
        samples.append(
            {
                "indices": topk_indices.to(torch.int32).cpu(),
                "values": topk_values.to(torch.bfloat16).cpu(),
            }
        )

    del logits
    return samples


# ---------------------------------------------------------------------------
# [PARALLEL] Multi-GPU data-parallel sharding (additive; default path untouched).
#
# TR: Cok-GPU paralel precompute. Her worker (GPU) TUM packed stream'i SIFIRDAN
#     deterministik paketler (boylece her seq_index için identity AYNI kalır) ama
#     teacher-forward'u yalniz KENDI bloklarina yapar. Blok atamasi BLOK-DONGUSEL:
#     block_index = seq_index // chunk_size; worker, (block_index % num_shards ==
#     shard_id) olan bloklari isler. Her shard TAM bir blok icerir ve ilk
#     seq_index'iyle adlandirilir (part_{block_index*chunk_size}); reader shard'lari
#     tam-sayi part-index'ine gore sıralı okudugu için GLOBAL seq sirasi korunur ve
#     train-side per-sequence identity assert'i AYNEN geçerli kalır. Resume STATELESS:
#     bir blogun shard dosyasi varsa o blok atlanir (idempotent).
# EN: Multi-GPU data-parallel precompute. Each worker (GPU) re-packs the WHOLE stream
#     from line 0 deterministically (so every seq_index keeps the SAME identity) but
#     only teacher-forwards its OWN blocks. Block assignment is BLOCK-CYCLIC:
#     block_index = seq_index // chunk_size; a worker owns blocks where
#     (block_index % num_shards == shard_id). Each shard holds EXACTLY one block and is
#     named by its first seq_index (part_{block_index*chunk_size}); since the reader
#     sorts shards by integer part index, GLOBAL sequence order is preserved and the
#     train-side per-sequence identity assertion still holds verbatim. Resume is
#     STATELESS: a block whose shard file already exists is skipped (idempotent).
# ---------------------------------------------------------------------------

def _worker_state_path(logits_dir: Path, stage_name: str, shard_id: int, num_shards: int) -> Path:
    return logits_dir / f"{stage_name}_{SUBSET}_shard{shard_id}of{num_shards}_state.json"


def _save_worker_state(
    logits_dir: Path,
    stage_name: str,
    shard_id: int,
    num_shards: int,
    *,
    emitted: int,
    blocks_done: int,
    total_sequences: int,
    done: bool,
) -> None:
    path = _worker_state_path(logits_dir, stage_name, shard_id, num_shards)
    payload = {
        "shard_id": int(shard_id),
        "num_shards": int(num_shards),
        "sequences_emitted": int(max(0, emitted)),
        "blocks_done": int(max(0, blocks_done)),
        "total_sequences": int(max(0, total_sequences)),
        "done": bool(done),
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best-effort progress file
        logger.warning("Failed to write worker state %s: %s", path, exc)


def _load_worker_state(logits_dir: Path, stage_name: str, shard_id: int, num_shards: int) -> dict:
    path = _worker_state_path(logits_dir, stage_name, shard_id, num_shards)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "shard_id": int(shard_id),
        "num_shards": int(num_shards),
        "sequences_emitted": 0,
        "blocks_done": 0,
        "total_sequences": 0,
        "done": False,
    }


def block_first_seq_indices(total_sequences: int, chunk_size: int) -> list[int]:
    """Canonical first-seq-index (== shard part index) of every block tiling [0, S).

    The full set of shard part indices a complete parallel run must produce.
    Pure/deterministic so the orchestrator can verify coverage without GPUs.
    """
    if total_sequences <= 0 or chunk_size <= 0:
        return []
    n_blocks = (int(total_sequences) + int(chunk_size) - 1) // int(chunk_size)
    return [b * int(chunk_size) for b in range(n_blocks)]


def precompute_stage(
    stage_num: int,
    teacher,
    tokenizer,
    logits_dir: Path,
    top_k: int,
    chunk_size: int,
    batch_size: int,
    max_seq: int,
) -> None:
    # TR: [B2] Packed precompute: student ile AYNI deterministik packer; her packed
    #     dizinin per-sequence kimligi (identity) + true_len shard'a yazilir, train
    #     load'da HARD-ASSERT edilir. Resume HAM SATIR indexine gore. Tokenizer,
    #     teacher MODELINE beslenecek ID'leri ureten student tokenizer'i ile AYNI
    #     olmali (resolve_tokenizer); main() bunu garanti eder.
    # EN: [B2] Packed precompute using the SAME deterministic packer as the student;
    #     each packed sequence's identity + true_len is written to the shard and
    #     HARD-ASSERTed at train load. Resume keyed by RAW LINE INDEX. Tokenizer must
    #     equal the student tokenizer that produces the ids fed to the teacher model.
    from train.packing import (
        TOPK_PACKED_FORMAT,
        extract_row_text,
        iter_packed_sequences,
    )
    from utils.tokenizer_resolver import tokenizer_identity

    stage_name = f"stage{stage_num}"
    jsonl_path = STAGE_FILES[stage_num]

    if not jsonl_path.exists():
        logger.warning("%s dataset missing: %s - skipping.", stage_name, jsonl_path)
        return

    total_lines = _count_jsonl(jsonl_path)
    resume = _load_resume_state(logits_dir, stage_name)
    lines_consumed = resume["lines_consumed"]
    sequences_emitted = resume["sequences_emitted"]
    chunk_idx = _next_chunk_idx(logits_dir, stage_name)

    if lines_consumed >= total_lines and total_lines > 0:
        logger.info("%s already complete (%s/%s lines) - skipping.", stage_name, lines_consumed, total_lines)
        return

    logger.info(
        "\n%s\n[%s] dataset=%s\n[%s] total=%s lines | resume from line %s (chunk %s, %s seqs done)\n%s",
        "=" * 60, stage_name, jsonl_path, stage_name, total_lines,
        lines_consumed, chunk_idx, sequences_emitted, "=" * 60,
    )

    try:
        teacher_device = next(teacher.parameters()).device
    except Exception:
        teacher_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    tok_identity = tokenizer_identity(tokenizer, cfg)

    shard_buffer: list[dict] = []
    started = time.time()

    def flush_chunk(consumed_through: int) -> None:
        nonlocal chunk_idx, sequences_emitted
        shard_path = logits_dir / f"{stage_name}_{SUBSET}_part_{chunk_idx}.pt"
        payload = {
            "format": TOPK_PACKED_FORMAT,
            "top_k": int(top_k),
            "vocab_size": vocab_size,
            "max_seq_len": int(max_seq),
            "pad_id": int(pad_id),
            "eos_id": int(eos_id),
            "tokenizer_identity": tok_identity,
            "packer_version": "packed_v1",
            "logits": list(shard_buffer),
        }
        torch.save(payload, shard_path)
        _save_resume_state(
            logits_dir, stage_name,
            lines_consumed=int(consumed_through) + 1,
            sequences_emitted=sequences_emitted,
        )
        elapsed = max(time.time() - started, 1e-6)
        logger.info(
            "Saved shard %s -> %s | %s seqs | %s seqs total | consumed<=line %s | %.1f seq/s",
            chunk_idx, shard_path.name, len(shard_buffer), sequences_emitted,
            consumed_through, sequences_emitted / elapsed,
        )
        chunk_idx += 1
        shard_buffer.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # TR: [tier-2 HIGH] iter_packed_sequences durumsal (greedy buffer carryover) bir
    #     packer; ortadan başlatmak (lines_consumed'dan) tek-geçiş packing'den sapıp
    #     resume seam'inde identity uyusmazligi uretebiliyordu. Cozum: HER ZAMAN satir
    #     0'dan tekrar-paketle (deterministik) ve sadece zaten uretilmis (seq_index <
    #     resume_seq) dizilerin TEACHER-FORWARD'ini atla. Packing ucuz; pahali olan
    #     teacher forward, o da atlaniyor.
    # EN: [tier-2 HIGH] iter_packed_sequences is a stateful greedy packer; resuming
    #     mid-file diverged from the single-pass packing and could produce a seam
    #     identity mismatch. Fix: ALWAYS re-pack from line 0 (deterministic) and only
    #     SKIP the teacher forward for already-emitted sequences (seq_index <
    #     resume_seq). Packing is cheap; the expensive teacher forward is skipped.
    resume_seq = sequences_emitted

    def _rows():
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for li, raw in enumerate(handle):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                yield li, extract_row_text(obj)

    def _process_batch(seqs: list[dict]) -> int:
        if not seqs:
            return -1
        input_ids = torch.tensor([s["input_ids"] for s in seqs], dtype=torch.long)
        topk_items = extract_topk_logits(teacher, input_ids, top_k, teacher_device)
        for seq, item in zip(seqs, topk_items):
            shard_buffer.append({
                # [B2] stamp the packed format on each item so the loader surfaces
                # 'topk_packed_v1' (not just the wrapper), keeping reader/writer aligned.
                "format": TOPK_PACKED_FORMAT,
                "indices": item["indices"],
                "values": item["values"],
                "true_len": int(seq["true_len"]),
                "seq_index": int(seq["seq_index"]),
                "row_span": seq["row_span"],
                "identity": seq["identity"],
            })
        return int(seqs[-1]["consumed_through"])

    batch_seqs: list[dict] = []
    last_consumed = lines_consumed - 1
    failed_batches = 0

    for seq in iter_packed_sequences(_rows(), tokenizer, max_seq, eos_id, pad_id):
        # Skip the teacher forward for sequences already emitted in a prior run.
        if int(seq["seq_index"]) < resume_seq:
            continue
        batch_seqs.append(seq)
        if len(batch_seqs) >= batch_size:
            try:
                last_consumed = _process_batch(batch_seqs)
                sequences_emitted += len(batch_seqs)
            except Exception as exc:
                # Batch failed: count it so the stage is NOT marked complete below.
                # Fallback (skip the batch and continue) is preserved; the resume
                # state will stay partial so this block is retried on the next run.
                failed_batches += 1
                logger.warning("Teacher forward failed for %s batch: %s", stage_name, exc)
            batch_seqs = []
            if len(shard_buffer) >= chunk_size:
                flush_chunk(last_consumed)

    if batch_seqs:
        try:
            last_consumed = _process_batch(batch_seqs)
            sequences_emitted += len(batch_seqs)
        except Exception as exc:
            failed_batches += 1
            logger.warning("Teacher forward failed for final %s batch: %s", stage_name, exc)
        batch_seqs = []

    if shard_buffer:
        flush_chunk(max(last_consumed, total_lines - 1))

    if failed_batches:
        # Coverage gap: one or more batches failed their teacher forward. Do NOT
        # claim the stage is fully consumed (that would silently skip the missing
        # sequences on resume). Persist a PARTIAL resume state instead so the next
        # run re-packs from line 0 and retries the not-yet-emitted sequences.
        logger.warning(
            "%s incomplete: %s batch(es) failed; leaving resume state partial for retry.",
            stage_name, failed_batches,
        )
        _save_resume_state(
            logits_dir, stage_name,
            lines_consumed=lines_consumed, sequences_emitted=sequences_emitted,
        )
    else:
        # Mark fully consumed even if the final partial buffer produced no shard.
        _save_resume_state(logits_dir, stage_name, lines_consumed=total_lines, sequences_emitted=sequences_emitted)

    elapsed = time.time() - started
    logger.info(
        "%s complete: %s packed sequences, %.1f minutes elapsed.",
        stage_name, sequences_emitted, elapsed / 60.0,
    )


def _precompute_stage_sharded(
    stage_num: int,
    teacher,
    tokenizer,
    logits_dir: Path,
    top_k: int,
    chunk_size: int,
    batch_size: int,
    max_seq: int,
    num_shards: int,
    shard_id: int,
) -> None:
    """[PARALLEL] One data-parallel worker of a block-cyclic precompute.

    See the module-level [PARALLEL] note. Invariants this function guarantees:
      * Re-packs the WHOLE stream from line 0 (identical to the single-process
        packer) so every sequence's identity is unchanged.
      * Teacher-forwards ONLY blocks it owns (block_index % num_shards == shard_id).
      * Writes each owned block as one shard named by its first seq_index, so the
        union of all workers' shards reads back in exact global seq_index order.
      * Resume is stateless: an owned block whose shard already exists is skipped.
    With num_shards == 1 / shard_id == 0 this is behaviourally equivalent to
    precompute_stage (same shards, same order); the canonical lane keeps using
    precompute_stage so its on-disk naming stays byte-identical.
    """
    from train.packing import (
        TOPK_PACKED_FORMAT,
        extract_row_text,
        iter_packed_sequences,
    )
    from utils.tokenizer_resolver import tokenizer_identity

    stage_name = f"stage{stage_num}"
    jsonl_path = STAGE_FILES[stage_num]

    if not jsonl_path.exists():
        logger.warning("%s dataset missing: %s - skipping.", stage_name, jsonl_path)
        _save_worker_state(
            logits_dir, stage_name, shard_id, num_shards,
            emitted=0, blocks_done=0, total_sequences=0, done=True,
        )
        return

    total_lines = _count_jsonl(jsonl_path)

    try:
        teacher_device = next(teacher.parameters()).device
    except Exception:
        teacher_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    tok_identity = tokenizer_identity(tokenizer, cfg)

    started = time.time()
    emitted = 0
    blocks_done = 0
    skipped_existing = 0
    last_seq_index = -1

    logger.info(
        "\n%s\n[%s] PARALLEL worker %d/%d | dataset=%s | %s lines | block(chunk)=%s\n%s",
        "=" * 60, stage_name, shard_id, num_shards, jsonl_path, total_lines, chunk_size, "=" * 60,
    )

    def _rows():
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for li, raw in enumerate(handle):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                yield li, extract_row_text(obj)

    def _process_batch_into(buf: list[dict], seqs: list[dict]) -> None:
        if not seqs:
            return
        input_ids = torch.tensor([s["input_ids"] for s in seqs], dtype=torch.long)
        items = extract_topk_logits(teacher, input_ids, top_k, teacher_device)
        for seq, item in zip(seqs, items):
            buf.append({
                "format": TOPK_PACKED_FORMAT,
                "indices": item["indices"],
                "values": item["values"],
                "true_len": int(seq["true_len"]),
                "seq_index": int(seq["seq_index"]),
                "row_span": seq["row_span"],
                "identity": seq["identity"],
            })

    def _flush_block(block_index: int, buf: list[dict]) -> None:
        nonlocal blocks_done
        first_seq = block_index * chunk_size
        shard_path = logits_dir / f"{stage_name}_{SUBSET}_part_{first_seq}.pt"
        payload = {
            "format": TOPK_PACKED_FORMAT,
            "top_k": int(top_k),
            "vocab_size": vocab_size,
            "max_seq_len": int(max_seq),
            "pad_id": int(pad_id),
            "eos_id": int(eos_id),
            "tokenizer_identity": tok_identity,
            "packer_version": "packed_v1",
            "shard_id": int(shard_id),
            "num_shards": int(num_shards),
            "block_index": int(block_index),
            "logits": list(buf),
        }
        torch.save(payload, shard_path)
        blocks_done += 1
        elapsed = max(time.time() - started, 1e-6)
        logger.info(
            "worker %d/%d %s: block %d -> %s | %d seqs | %d emitted | %.1f seq/s",
            shard_id, num_shards, stage_name, block_index, shard_path.name,
            len(buf), emitted + len(buf), (emitted + len(buf)) / elapsed,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    current_block: Optional[int] = None
    buf: list[dict] = []
    batch_seqs: list[dict] = []

    for seq in iter_packed_sequences(_rows(), tokenizer, max_seq, eos_id, pad_id):
        si = int(seq["seq_index"])
        last_seq_index = si
        block_index = si // chunk_size
        if (block_index % num_shards) != shard_id:
            continue  # not this worker's block
        shard_path = logits_dir / f"{stage_name}_{SUBSET}_part_{block_index * chunk_size}.pt"
        if shard_path.exists():
            skipped_existing += 1
            continue  # already produced by a prior run (idempotent resume)
        if current_block is None:
            current_block = block_index
        if block_index != current_block:
            # the previous owned block is now complete: drain + flush it
            _process_batch_into(buf, batch_seqs)
            batch_seqs = []
            if buf:
                _flush_block(current_block, buf)
                emitted += len(buf)
                buf = []
            current_block = block_index
        batch_seqs.append(seq)
        if len(batch_seqs) >= batch_size:
            _process_batch_into(buf, batch_seqs)
            batch_seqs = []

    # drain the final partial block
    _process_batch_into(buf, batch_seqs)
    batch_seqs = []
    if buf and current_block is not None:
        _flush_block(current_block, buf)
        emitted += len(buf)
        buf = []

    total_sequences = last_seq_index + 1
    _save_worker_state(
        logits_dir, stage_name, shard_id, num_shards,
        emitted=emitted, blocks_done=blocks_done,
        total_sequences=total_sequences, done=True,
    )
    elapsed = time.time() - started
    logger.info(
        "worker %d/%d %s done: emitted=%d, blocks=%d, skipped_existing=%d, S=%d, %.1f min",
        shard_id, num_shards, stage_name, emitted, blocks_done, skipped_existing,
        total_sequences, elapsed / 60.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MertFormer Titan Phase-0 Top-K teacher logit precomputation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all-stages", action="store_true", help="Process all 5 stages.")
    group.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5], help="Process a single stage.")

    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ)
    parser.add_argument(
        "--logits-dir",
        type=Path,
        default=Path(getattr(cfg, "precomputed_logits_path", "./datasets/logits/")),
        help="Target shard directory.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=getattr(cfg, "teacher_model_id", "meta-llama/Llama-3.3-70B-Instruct"),
    )
    parser.add_argument("--dry-run", action="store_true", help="Only verify dataset presence and estimate disk.")
    parser.add_argument(
        "--check-complete",
        action="store_true",
        help="Exit 0 only when the requested stages already have complete shard/state coverage.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="[PARALLEL] Total data-parallel workers (GPUs). 1 = single-process (default, "
             "byte-identical behaviour). Used by scripts/precompute_logits_parallel.py.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="[PARALLEL] This worker's id in [0, num-shards). Owns blocks where "
             "(seq_index // chunk-size) %% num-shards == shard-id.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stages = [1, 2, 3, 4, 5] if args.all_stages else [args.stage]
    logits_dir = args.logits_dir.resolve()

    if args.dry_run:
        logger.info("DRY-RUN: dataset presence and rough disk estimate")
        all_present = True
        for stage_num in stages:
            dataset_path = STAGE_FILES[stage_num]
            present = dataset_path.exists()
            all_present = all_present and present
            rows = _count_jsonl(dataset_path) if present else 0
            est_mb = _estimate_disk_mb(rows, args.max_seq, args.top_k) if present else 0.0
            if present:
                logger.info("stage%s: OK | %s rows | ~%d MB", stage_num, rows, int(est_mb))
            else:
                logger.warning("stage%s: missing | %s", stage_num, dataset_path)
        return 0 if all_present else 4

    logits_dir.mkdir(parents=True, exist_ok=True)

    if args.check_complete:
        all_complete = True
        for stage_num in stages:
            stage_name = f"stage{stage_num}"
            total_samples = _count_jsonl(STAGE_FILES[stage_num])
            complete = _stage_complete(logits_dir, stage_name, total_samples)
            logger.info(
                "%s: %s | done=%s/%s | shards=%s",
                stage_name,
                "complete" if complete else "incomplete",
                _load_done_samples(logits_dir, stage_name),
                total_samples,
                len(_stage_shards(logits_dir, stage_name)),
            )
            all_complete = all_complete and complete
        return 0 if all_complete else 5

    num_shards = int(getattr(args, "num_shards", 1) or 1)
    shard_id = int(getattr(args, "shard_id", 0) or 0)
    is_sharded = num_shards > 1
    if num_shards < 1 or not (0 <= shard_id < max(1, num_shards)):
        logger.error(
            "Invalid --num-shards/--shard-id: num_shards=%s shard_id=%s "
            "(require num_shards>=1 and 0<=shard_id<num_shards).",
            num_shards, shard_id,
        )
        return 1

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    if not hf_token:
        logger.error("HF_TOKEN is required for real teacher logit precomputation.")
        return 1

    logger.info("Logits dir: %s", logits_dir)
    logger.info("Format   : Top-%s sparse (indices int32 + values bf16)", args.top_k)
    logger.info("Seq len  : %s", args.max_seq)

    total_est = 0.0
    for stage_num in stages:
        dataset_path = STAGE_FILES[stage_num]
        if dataset_path.exists():
            total_est += _estimate_disk_mb(_count_jsonl(dataset_path), args.max_seq, args.top_k)
    logger.info("Estimated total disk: %.2f GB", total_est / 1024.0)

    # [P0 FIX] Disk pre-flight gate. Sparse Top-K logits scale ~linearly with --top-k and
    # can be enormous (top_k=256 over the full ~23.6B-token budget is tens of TB). Refuse to
    # start a multi-hour teacher run that would run out of disk mid-way. This process writes
    # only its own shard share (total_est / num_shards). Override with TITAN_SKIP_DISK_GATE=1.
    if os.environ.get("TITAN_SKIP_DISK_GATE", "").strip().lower() not in {"1", "true", "yes"}:
        try:
            free_bytes: Optional[int] = shutil.disk_usage(logits_dir).free
        except OSError:
            free_bytes = None
        if free_bytes is not None:
            est_bytes = (total_est / max(1, num_shards)) * 1024.0 * 1024.0
            required_bytes = est_bytes * 1.10  # +10% margin for shard headers / fs overhead
            gib = 1024.0 ** 3
            if required_bytes > free_bytes:
                logger.error(
                    "Insufficient disk for Top-%s logits at %s: need ~%.1f GB "
                    "(est %.1f GB + 10%% margin) but only %.1f GB free. Lower --top-k "
                    "(disk scales ~linearly with top_k), free space, or set "
                    "TITAN_SKIP_DISK_GATE=1 to override.",
                    args.top_k, logits_dir,
                    required_bytes / gib, est_bytes / gib, free_bytes / gib,
                )
                return 1
            logger.info(
                "Disk gate OK: need ~%.1f GB, %.1f GB free at %s",
                required_bytes / gib, free_bytes / gib, logits_dir,
            )

    time.sleep(1)
    teacher, teacher_tokenizer = load_teacher(args.model_id, hf_token)

    # TR: [B2] Tokenizer, training'in kullandigi student tokenizer'i ile AYNI olmali
    #     (resolve_tokenizer). Bu ID'ler teacher MODELINE besleniyor; sparse Top-K KD
    #     ancak teacher==student tokenizer ise anlamli. use_tr_tokenizer=True iken
    #     student TR-WordPiece olur ve Llama teacher'a verilemez -> acik hata.
    # EN: [B2] The tokenizer must equal the student tokenizer training uses
    #     (resolve_tokenizer). These ids are fed to the teacher MODEL; sparse Top-K KD
    #     only makes sense when teacher==student tokenizer. With use_tr_tokenizer=True
    #     the student is TR-WordPiece and cannot feed the Llama teacher -> hard error.
    from utils.tokenizer_resolver import resolve_tokenizer
    tokenizer = resolve_tokenizer(cfg)
    if bool(getattr(cfg, "use_tr_tokenizer", False)):
        logger.error(
            "use_tr_tokenizer=1 is incompatible with precomputed Top-K KD against the "
            "teacher model (the TR tokenizer's ids are not the teacher's vocabulary). "
            "Precompute requires teacher==student tokenizer (use_tr_tokenizer=0)."
        )
        return 1
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_sharded:
        logger.info("PARALLEL worker mode: shard %d of %d (block-cyclic).", shard_id, num_shards)

    for stage_num in stages:
        try:
            if is_sharded:
                _precompute_stage_sharded(
                    stage_num=stage_num,
                    teacher=teacher,
                    tokenizer=tokenizer,
                    logits_dir=logits_dir,
                    top_k=args.top_k,
                    chunk_size=args.chunk_size,
                    batch_size=args.batch_size,
                    max_seq=args.max_seq,
                    num_shards=num_shards,
                    shard_id=shard_id,
                )
            else:
                precompute_stage(
                    stage_num=stage_num,
                    teacher=teacher,
                    tokenizer=tokenizer,
                    logits_dir=logits_dir,
                    top_k=args.top_k,
                    chunk_size=args.chunk_size,
                    batch_size=args.batch_size,
                    max_seq=args.max_seq,
                )
        except KeyboardInterrupt:
            logger.warning("Interrupted by user; progress is already checkpointed in shard/state files.")
            return 130
        except Exception as exc:
            logger.error("stage%s failed: %s", stage_num, exc)
            raise

    logger.info("All requested Phase-0 precompute stages completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
