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


def _stage_shards(logits_dir: Path, stage_name: str) -> list[Path]:
    return sorted(logits_dir.glob(_stage_pattern(stage_name)))


def _unwrap_payload_for_count(payload) -> int:
    if isinstance(payload, dict) and "logits" in payload:
        payload = payload["logits"]
    if isinstance(payload, (list, tuple)):
        return len(payload)
    return 0


def _load_done_samples(logits_dir: Path, stage_name: str) -> int:
    state_path = _state_path(logits_dir, stage_name)
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            return int(data.get("done_samples", 0))
        except Exception:
            pass

    total = 0
    for shard in _stage_shards(logits_dir, stage_name):
        try:
            payload = torch.load(shard, map_location="cpu", weights_only=False)
            total += _unwrap_payload_for_count(payload)
        except Exception:
            continue
    if total > 0:
        _save_done_samples(logits_dir, stage_name, total)
    return total


def _save_done_samples(logits_dir: Path, stage_name: str, done_samples: int) -> None:
    state_path = _state_path(logits_dir, stage_name)
    payload = {"done_samples": int(max(0, done_samples))}
    state_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _next_chunk_idx(logits_dir: Path, stage_name: str) -> int:
    indices: list[int] = []
    for shard in _stage_shards(logits_dir, stage_name):
        try:
            indices.append(int(shard.stem.rsplit("_part_", 1)[-1]))
        except Exception:
            continue
    return (max(indices) + 1) if indices else 0


def _estimate_disk_mb(n_samples: int, seq_len: int, top_k: int) -> float:
    bytes_per_sample = seq_len * top_k * 6
    return (n_samples * bytes_per_sample) / (1024 ** 2)


def _count_jsonl(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except Exception:
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
    stage_name = f"stage{stage_num}"
    jsonl_path = STAGE_FILES[stage_num]

    if not jsonl_path.exists():
        logger.warning("%s dataset missing: %s - skipping.", stage_name, jsonl_path)
        return

    total_samples = _count_jsonl(jsonl_path)
    done_samples = _load_done_samples(logits_dir, stage_name)
    chunk_idx = _next_chunk_idx(logits_dir, stage_name)

    if done_samples >= total_samples and total_samples > 0:
        logger.info("%s already complete (%s/%s) - skipping.", stage_name, done_samples, total_samples)
        return

    est_mb = _estimate_disk_mb(max(total_samples - done_samples, 0), max_seq, top_k)
    logger.info(
        "\n%s\n[%s] dataset=%s\n[%s] total=%s samples\n[%s] resume=%s samples (chunk %s)\n[%s] remaining disk ~= %.0f MB\n%s",
        "=" * 60,
        stage_name,
        jsonl_path,
        stage_name,
        total_samples,
        stage_name,
        done_samples,
        chunk_idx,
        stage_name,
        est_mb,
        "=" * 60,
    )

    try:
        teacher_device = next(teacher.parameters()).device
    except Exception:
        teacher_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shard_buffer: list[dict[str, torch.Tensor]] = []
    processed = done_samples
    skipped = 0
    started = time.time()
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)

    def flush_chunk() -> None:
        nonlocal chunk_idx
        shard_path = logits_dir / f"{stage_name}_{SUBSET}_part_{chunk_idx}.pt"
        payload = {
            "format": "topk_sparse_v1",
            "top_k": int(top_k),
            "vocab_size": vocab_size,
            "logits": list(shard_buffer),
        }
        torch.save(payload, shard_path)
        _save_done_samples(logits_dir, stage_name, processed)
        elapsed = max(time.time() - started, 1e-6)
        speed = processed / elapsed
        logger.info(
            "Saved shard %s -> %s | %s samples | total %s/%s | %.1f sample/s",
            chunk_idx,
            shard_path.name,
            len(shard_buffer),
            processed,
            total_samples,
            speed,
        )
        chunk_idx += 1
        shard_buffer.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    batch_texts: list[str] = []
    sample_idx = 0

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            if sample_idx < done_samples:
                sample_idx += 1
                continue

            try:
                obj = json.loads(raw_line)
                text = (
                    obj.get("text")
                    or obj.get("content")
                    or obj.get("instruction")
                    or ""
                ).strip()
            except Exception:
                sample_idx += 1
                skipped += 1
                continue

            if not text:
                sample_idx += 1
                skipped += 1
                continue

            batch_texts.append(text)
            sample_idx += 1
            if len(batch_texts) < batch_size:
                continue

            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=max_seq,
                truncation=True,
                padding=True,
            )
            input_ids = encoded["input_ids"]

            try:
                topk_items = extract_topk_logits(teacher, input_ids, top_k, teacher_device)
            except Exception as exc:
                failed_batch = len(batch_texts)
                logger.warning("Teacher forward failed for %s batch: %s", stage_name, exc)
                skipped += failed_batch
                batch_texts.clear()
                continue

            shard_buffer.extend(topk_items)
            processed += len(topk_items)
            batch_texts.clear()

            if len(shard_buffer) >= chunk_size:
                flush_chunk()

    if batch_texts:
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=max_seq,
            truncation=True,
            padding=True,
        )
        input_ids = encoded["input_ids"]
        try:
            topk_items = extract_topk_logits(teacher, input_ids, top_k, teacher_device)
            shard_buffer.extend(topk_items)
            processed += len(topk_items)
        except Exception as exc:
            logger.warning("Teacher forward failed for final %s batch: %s", stage_name, exc)
            skipped += len(batch_texts)

    if shard_buffer:
        flush_chunk()

    elapsed = time.time() - started
    logger.info(
        "%s complete: %s samples processed, %s skipped, %.1f minutes elapsed.",
        stage_name,
        processed,
        skipped,
        elapsed / 60.0,
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

    time.sleep(1)
    teacher, tokenizer = load_teacher(args.model_id, hf_token)

    for stage_num in stages:
        try:
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
