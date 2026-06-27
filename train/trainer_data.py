"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30 V2) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)

Module: Training datasets, collate function and teacher-logit payload
helpers (curriculum streaming, precomputed-logit alignment, sparse Top-K).
Split out of train/train.py as pure code motion; train/train.py re-exports
every symbol so all historical imports keep working unchanged.
==============================================================================
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert Yünlü"

import json
import random
from pathlib import Path
from typing import Any, List, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info

from config.config import cfg


# -----------------------------------------------------------------------------
# DATASETS (CURRICULUM + VALIDATION)
# -----------------------------------------------------------------------------


def _encode_with_eos_labels(
    tokenizer: Any, text: str, max_len: int, pad_id: int, eos_id: Optional[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """[H2] Tokenize one row, append EOS, pad to max_len; labels carry -100 at
    trailing pad so CE/KD ignore pad while EOS is supervised (pad==eos safe)."""
    from train.packing import encode_row
    ids = encode_row(tokenizer, text, max(1, int(max_len) - 1))
    if eos_id is not None:
        ids = ids + [int(eos_id)]
    ids = ids[: int(max_len)]
    true_len = len(ids)
    pad_n = int(max_len) - true_len
    input_list = ids + [int(pad_id)] * pad_n
    label_list = ids + [-100] * pad_n
    return (
        torch.tensor(input_list, dtype=torch.long),
        torch.tensor(label_list, dtype=torch.long),
    )


class ValidationJsonlDataset(IterableDataset):
    """
    Simple, sequential (deterministic) reader for validation.
    """
    def __init__(self, path: Path, max_len: int, tokenizer: Any) -> None:
        self.path = path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.eos_id = tokenizer.eos_token_id

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if not text:
                        continue
                    input_ids, labels = _encode_with_eos_labels(
                        self.tokenizer, text, self.max_len, self.pad_id, self.eos_id
                    )
                    yield input_ids, labels
                except Exception:
                    continue


class CurriculumDataset(IterableDataset):
    """
    Curriculum-aware dataset (RAM-friendly Streaming + Worker Safe).
    """

    def __init__(
        self, stage_paths: List[Path], max_len: int, tokenizer: Any, current_stage: int = 1
    ) -> None:
        self.stage_paths = stage_paths
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.current_stage = current_stage
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # [MED] Byte-seek over-sampled long lines (a longer line spans more byte
        #     space, so a random byte offset lands in it more often) and never
        #     yielded the first line, distorting intra-stage source ratios. Fix:
        #     index each non-empty line's byte offset once, then sample a UNIFORM
        #     line index (no length bias).
        self.file_offsets = {}
        self.line_offsets = {}
        for p in stage_paths:
            if p.exists():
                offsets = []
                with open(p, "rb") as f:
                    pos = f.tell()
                    line = f.readline()
                    while line:
                        if line.strip():
                            offsets.append(pos)
                        pos = f.tell()
                        line = f.readline()
                    self.file_offsets[str(p)] = pos
                self.line_offsets[str(p)] = offsets

    def set_stage(self, stage: int) -> None:
        """Update current curriculum stage."""
        self.current_stage = stage

    def __iter__(self):
        # 1. WORKER SEED SYNC (prevent multi-worker collisions)
        worker_info = get_worker_info()
        if worker_info is not None:
            # Each worker gets a different seed
            random.seed(cfg.seed + worker_info.id)

        skipped_count = 0
        total_attempts = 0

        while True:
            # [FIX] Fallback guard: If not enough stage files, use all available
            n = len(self.stage_paths)
            if n < 4:
                # Fallback dataset or missing stages: disable curriculum
                active_paths = self.stage_paths
            elif self.current_stage == 1:
                active_paths = [self.stage_paths[0]]
            elif self.current_stage == 2:
                active_paths = [self.stage_paths[1]]
            elif self.current_stage == 3:
                # Stage 3: Mix all previous stages (10% Stage1, 10% Stage2, 80% Stage3)
                active_paths = self.stage_paths[:3]
            elif self.current_stage == 4:
                # Stage 4: Soul + Knowledge + Identity Mix
                active_paths = self.stage_paths
            else:
                active_paths = self.stage_paths

            # Select path based on stage logic or random choice from actives
            if self.current_stage == 3 and len(active_paths) > 1:
                r = random.random()
                if r < 0.1: path = active_paths[0]
                elif r < 0.2: path = active_paths[1]
                else: path = active_paths[2]
            else:
                path = random.choice(active_paths)

            if not path.exists():
                continue

            offsets = self.line_offsets.get(str(path), [])
            if not offsets: continue

            total_attempts += 1

            # 2. DATA LOSS ALARM (5% threshold)
            if total_attempts % 5000 == 0 and total_attempts > 0:
                skip_rate = skipped_count / total_attempts
                if skip_rate > 0.05:
                    print(f"⚠️ DİKKAT: Veri atlama oranı yüksek: %{skip_rate*100:.1f} (JSON parse veya boş satır)")

            try:
                # [FIX] Binary mode seek is safe; text mode seek is fragile with UTF-8
                with open(path, "rb") as f:
                    # [MED] Pick a UNIFORM line index, seek to that line's offset.
                    idx = random.randint(0, len(offsets) - 1)
                    f.seek(offsets[idx])
                    line_bytes = f.readline()

                    if not line_bytes:
                        skipped_count += 1
                        continue

                    # [TITAN FIX] Decode with error handling (errors='ignore' for max stability)
                    try:
                        line = line_bytes.decode("utf-8", errors="ignore")
                    except Exception:
                        skipped_count += 1
                        continue

                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if not text:
                        skipped_count += 1
                        continue

                    # [H2] Append EOS + labels -100 at trailing pad (pad==eos safe).
                    input_ids, labels = _encode_with_eos_labels(
                        self.tokenizer, text, self.max_len, self.pad_id,
                        self.tokenizer.eos_token_id,
                    )
                    yield input_ids, labels
            except Exception:
                skipped_count += 1
                continue


class PrecomputedCurriculumDataset(IterableDataset):
    """
    Deterministic curriculum dataset paired with precomputed logits.
    """

    def __init__(
        self, stage_info: List[tuple[str, Path]], max_len: int, tokenizer: Any, distill_manager: Any
    ) -> None:
        self.stage_info = stage_info  # list of (stage_name, path)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.distill_manager = distill_manager
        self.current_stage = 1
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def set_stage(self, stage: int) -> None:
        self.current_stage = stage

    def _stage_entry(self, stage: int) -> tuple[str, Path]:
        if not self.stage_info:
            raise RuntimeError("PrecomputedCurriculumDataset has no stage paths.")
        idx = min(max(int(stage) - 1, 0), len(self.stage_info) - 1)
        return self.stage_info[idx]

    def _align_logits(self, logits: Any, target_len: int) -> Any:
        # Align logits length to token length
        if _is_sparse_topk_payload(logits):
            return _align_sparse_topk_payload(logits, target_len)

        # logits: [seq, vocab]
        if logits.dim() == 3 and logits.size(0) == 1:
            logits = logits.squeeze(0)
        if logits.dim() != 2:
            raise ValueError(f"Invalid logits shape: {tuple(logits.shape)}")
        seq_len = logits.size(0)
        if seq_len > target_len:
            logits = logits[:target_len]
        elif seq_len < target_len:
            pad = torch.zeros(target_len - seq_len, logits.size(1), dtype=logits.dtype)
            logits = torch.cat([logits, pad], dim=0)
        return logits

    def _iter_stage(self, stage_name: str, path: Path):
        # [B2] Build the student stream from the SAME packer as precompute and
        #     HARD-ASSERT each packed sequence's teacher-shard identity (no silent
        #     realign). Pad positions are -100 in labels. Legacy non-packed path is
        #     reachable only via TITAN_ALLOW_LEGACY_LOGIT_REALIGN=1.
        import os as _os
        from train.packing import (
            LogitAlignmentError,
            assert_sequence_identity,
            extract_row_text,
            iter_packed_sequences,
        )

        use_packing = bool(getattr(cfg, "sequence_packing", True))
        verify = bool(getattr(cfg, "verify_logit_alignment", True))
        legacy = _os.environ.get("TITAN_ALLOW_LEGACY_LOGIT_REALIGN", "0") == "1"
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = self.pad_id
        logits_iter = iter(self.distill_manager.get_precomputed_loader(stage_name))

        if use_packing and not legacy:
            def _rows():
                with open(path, "r", encoding="utf-8") as f:
                    for li, line in enumerate(f):
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        yield li, extract_row_text(obj)

            for seq in iter_packed_sequences(
                _rows(), self.tokenizer, self.max_len, eos_id, self.pad_id
            ):
                try:
                    t_item = next(logits_iter)
                except StopIteration:
                    return
                stored_identity = t_item.get("identity") if isinstance(t_item, dict) else None
                if verify:
                    assert_sequence_identity(seq["input_ids"], seq["true_len"], stored_identity)
                input_ids = torch.tensor(seq["input_ids"], dtype=torch.long)
                labels = input_ids.clone()
                tl = int(seq["true_len"])
                if tl < labels.numel():
                    labels[tl:] = -100  # ignore trailing pad (pad==eos safe)
                yield input_ids, labels, t_item
            return

        # Legacy non-packed path (escape hatch only).
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = extract_row_text(obj)
                    if not text:
                        continue
                    enc = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.max_len,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    input_ids = enc["input_ids"].squeeze(0)
                    try:
                        t_logits = next(logits_iter)
                    except StopIteration:
                        return
                    t_logits = self._align_logits(t_logits, input_ids.size(0))
                    labels = input_ids.clone()
                    labels[input_ids == self.pad_id] = -100
                    yield input_ids, labels, t_logits
                except LogitAlignmentError:
                    raise
                except Exception:
                    continue

    def __iter__(self):
        # Disable worker parallelism for offline logits alignment
        worker_info = get_worker_info()
        if worker_info is not None:
            raise RuntimeError("Precomputed logits require num_workers=0 for deterministic alignment.")

        current_stage = self.current_stage
        stage_name, path = self._stage_entry(current_stage)
        stage_iter = self._iter_stage(stage_name, path)

        while True:
            if self.current_stage != current_stage:
                current_stage = self.current_stage
                stage_name, path = self._stage_entry(current_stage)
                stage_iter = self._iter_stage(stage_name, path)

            try:
                yield next(stage_iter)
            except StopIteration:
                stage_iter = self._iter_stage(stage_name, path)


def collate_fn(batch: List[tuple]) -> tuple:
    # [FIX] Simplified to 2 elements (removed text)
    if len(batch[0]) == 3:
        x, y, t = zip(*batch)
        return torch.stack(x), torch.stack(y), _stack_teacher_payloads(t)
    x, y = zip(*batch)
    return torch.stack(x), torch.stack(y)


# -----------------------------------------------------------------------------
# TEACHER LOGIT PAYLOAD HELPERS (dense + sparse Top-K)
# -----------------------------------------------------------------------------


def _is_sparse_topk_payload(value: Any) -> bool:
    # A sparse Top-K teacher payload is a dict carrying matching indices/values
    # tensors. The 'format' string is intentionally OPTIONAL: the COLLATED payload
    # that crosses Accelerate's multi-GPU dispatch boundary is stripped to tensors
    # only - a str/int dict value makes accelerate.utils.concatenate() raise
    # TypeError on rank 0 while peers block at the next collective (NCCL hang).
    # Dense payloads are bare Tensors, so a dict with both tensors is unambiguously
    # the sparse case; if 'format' is present it must still be a known sparse tag.
    if not isinstance(value, dict):
        return False
    if not (isinstance(value.get("indices"), torch.Tensor)
            and isinstance(value.get("values"), torch.Tensor)):
        return False
    fmt = value.get("format")
    return fmt is None or fmt in ("topk_sparse_v1", "topk_packed_v1")


def _align_sparse_topk_payload(payload: dict, target_len: int, fill_value: float = -1e4) -> dict:
    indices = payload["indices"].long()
    values = payload["values"].float()
    if indices.dim() == 3 and indices.size(0) == 1:
        indices = indices.squeeze(0)
        values = values.squeeze(0)
    if indices.shape != values.shape or indices.dim() != 2:
        raise ValueError(
            "Sparse Top-K teacher payload must use matching [seq, top_k] indices/values tensors."
        )
    seq_len, top_k = indices.shape
    if seq_len > target_len:
        indices = indices[:target_len]
        values = values[:target_len]
    elif seq_len < target_len:
        pad_len = target_len - seq_len
        index_pad = torch.zeros(pad_len, top_k, dtype=indices.dtype)
        value_pad = torch.full((pad_len, top_k), fill_value, dtype=values.dtype)
        indices = torch.cat([indices, index_pad], dim=0)
        values = torch.cat([values, value_pad], dim=0)
    return {
        "format": "topk_sparse_v1",
        "indices": indices,
        "values": values,
        "vocab_size": int(payload.get("vocab_size", 0)),
        "top_k": int(payload.get("top_k", top_k)),
    }


def _stack_teacher_payloads(payloads: tuple[Any, ...]) -> Any:
    if all(isinstance(item, torch.Tensor) for item in payloads):
        return torch.stack(payloads)
    if all(_is_sparse_topk_payload(item) for item in payloads):
        vocab_sizes = {int(item.get("vocab_size", 0)) for item in payloads}
        top_ks = {int(item.get("top_k", item["indices"].size(-1))) for item in payloads}
        if len(vocab_sizes) != 1 or len(top_ks) != 1:
            raise ValueError("Sparse Top-K batch has inconsistent vocab_size or top_k metadata.")
        # TENSOR-ONLY stacked payload. The non-tensor metadata (format/vocab_size/
        # top_k) is deliberately dropped here: it is not read downstream (the KD loss
        # uses indices/values; top_k == indices.size(-1); sparse-ness is detected from
        # the tensors), and keeping it would crash accelerate's dispatch-time
        # concatenate() on multi-GPU (str/int values) -> NCCL hang on the canonical
        # offline_clean lane. The consistency check above still runs before we drop it.
        return {
            "indices": torch.stack([item["indices"].long() for item in payloads]),
            "values": torch.stack([item["values"].float() for item in payloads]),
        }
    raise ValueError("Cannot mix dense and sparse teacher logits in the same batch.")


def _teacher_payload_to_device(payload: Any, device: torch.device) -> Any:
    if _is_sparse_topk_payload(payload):
        return {
            **payload,
            "indices": payload["indices"].to(device=device, dtype=torch.long),
            "values": payload["values"].to(device=device, dtype=torch.float32),
        }
    return payload.to(device)


def _shift_teacher_payload(payload: Any) -> Any:
    if _is_sparse_topk_payload(payload):
        indices = payload["indices"]
        values = payload["values"]
        if indices.dim() == 3:
            indices = indices[:, :-1, :].contiguous()
            values = values[:, :-1, :].contiguous()
        elif indices.dim() == 2:
            indices = indices[:-1, :].contiguous()
            values = values[:-1, :].contiguous()
        else:
            raise ValueError(f"Invalid sparse teacher indices shape: {tuple(indices.shape)}")
        return {**payload, "indices": indices, "values": values}
    return payload[..., :-1, :].contiguous()
