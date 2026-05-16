"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30 V2) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert"

import os
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import IterableDataset
import logging

# Set up logging
logger = logging.getLogger(__name__)

TOPK_DENSE_MAX_ELEMENTS = int(os.environ.get("TITAN_TOPK_DENSE_MAX_ELEMENTS", "8000000"))

class DistillationManager:
    """
    TR: Öğretmen model (Llama-3-70B) logitlerini önceden hesaplayıp diske yazan yönetici.
    EN: Manager that pre-computes teacher model (Llama-3-70B) logits and writes to disk.
    
    Objective:
    - Save VRAM during training by removing the Teacher model from memory.
    - Speed up training (Teacher forward pass done once, offline).
    """
    
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.logits_dir = Path(cfg.precomputed_logits_path)
        self.logits_dir.mkdir(parents=True, exist_ok=True)
        self.teacher_model = None
        self.device = cfg.device
        self.require_gated_teacher = bool(getattr(cfg, "require_gated_teacher", False))

    def _state_file(self, stage_name: str, subset: str) -> Path:
        return self.logits_dir / f"{stage_name}_{subset}_state.json"

    def _load_processed_samples(self, stage_name: str, subset: str) -> int:
        """
        Returns number of already-processed samples for robust resume.
        Prefers state file; falls back to counting existing shard payload sizes.
        """
        state_file = self._state_file(stage_name, subset)
        if state_file.exists():
            try:
                import json
                with state_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return int(max(0, data.get("processed_samples", 0)))
            except Exception:
                pass

        processed = 0
        for file in _list_logits_files(self.logits_dir, stage_name, subset=subset):
            try:
                chunk = torch.load(file, map_location="cpu")
                if isinstance(chunk, dict) and "logits" in chunk:
                    chunk = chunk["logits"]
                if isinstance(chunk, (list, tuple)):
                    processed += len(chunk)
            except Exception:
                logger.warning(f"⚠️ Could not count shard entries: {file}")
                continue
        if processed > 0:
            self._save_processed_samples(stage_name, subset, processed)
        return processed

    def _save_processed_samples(self, stage_name: str, subset: str, processed_samples: int) -> None:
        state_file = self._state_file(stage_name, subset)
        payload = {"processed_samples": int(max(0, processed_samples))}
        try:
            import json
            with state_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"⚠️ Failed to write distillation resume state: {e}")

    def load_teacher(self):
        """
        Loads the Teacher Model (Llama-3-70B) in 8-bit/4-bit if possible.
        """
        if self.teacher_model is not None:
            return

        hf_token = os.environ.get("HF_TOKEN")
        if self.require_gated_teacher and not hf_token:
            raise RuntimeError(
                "require_gated_teacher=true but HF_TOKEN is missing. "
                "Distillation cannot proceed."
            )

        logger.info(f"🎓 Loading Teacher Model: {self.cfg.teacher_model_id}")
        try:
            # Optimal loading for 70B on 8x A100 or minimal setup
            # Using 4-bit / 8-bit quantization for inference efficiency
            load_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                "trust_remote_code": True
            }
            
            # If bitsandbytes is available, load in 4-bit
            try:
                import bitsandbytes
                load_kwargs["load_in_4bit"] = True
                logger.info("   ✅ Using 4-bit quantization for Teacher")
            except ImportError:
                logger.warning("   ⚠️  BitsAndBytes not found, loading full precision (High VRAM)")
            
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.cfg.teacher_model_id,
                token=hf_token,
                **load_kwargs
            )
            self.teacher_model.eval()
            logger.info("🎓 Teacher Model Loaded Successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load teacher model: {e}")
            raise e

    def precompute_logits(self, dataset, stage_name, subset="train"):
        """
        Iterates over the dataset, computes logits, and saves them to shards.
        Safe against crashes: Saves every 5000 samples. 
        """
        self.load_teacher()
        self.logits_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"⚙️  Pre-computing logits for {stage_name}...")
        
        all_logits = []
        batch_size = 8 
        chunk_size = 5000 # Save every 5000 samples
        chunk_idx = 0
        
        # [RESTART LOGIC] Find next chunk index
        while (self.logits_dir / f"{stage_name}_{subset}_part_{chunk_idx}.pt").exists():
            chunk_idx += 1

        processed_total = self._load_processed_samples(stage_name, subset)

        print(f"🔄 Resuming from Chunk {chunk_idx}")
        if processed_total > 0:
            print(f"🔁 Resume offset: skipping first {processed_total} already-processed samples")

        dataset_iter = iter(dataset)
        buffer_inputs = []

        processed_in_current_chunk = 0
        seen_samples = 0

        with torch.no_grad():
            pbar = tqdm(desc=f"Distilling {stage_name} (Chunk {chunk_idx})")

            while True:
                try:
                    item = next(dataset_iter)
                    seen_samples += 1
                    if seen_samples <= processed_total:
                        continue

                    # Handle different dataset formats
                    text = item.get('text') or item.get('content') or item.get('instruction') or ""

                    if not text:
                        continue

                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        max_length=self.cfg.max_seq_len,
                        truncation=True
                    )
                    buffer_inputs.append(inputs.input_ids[0])

                    # Process Batch
                    if len(buffer_inputs) >= batch_size:
                        max_len = max([t.size(0) for t in buffer_inputs])
                        padded_batch = torch.stack([
                            F.pad(t, (0, max_len - t.size(0)), value=self.tokenizer.pad_token_id)
                            for t in buffer_inputs
                        ]).to(self.device)
                        
                        outputs = self.teacher_model(padded_batch)
                        logits = outputs.logits

                        for i in range(logits.size(0)):
                            all_logits.append(logits[i].cpu().clone()) # Move to CPU
                            processed_total += 1

                        buffer_inputs = []
                        pbar.update(batch_size)
                        processed_in_current_chunk += batch_size

                        # Chunk Save
                        if processed_in_current_chunk >= chunk_size:
                            save_path = self.logits_dir / f"{stage_name}_{subset}_part_{chunk_idx}.pt"
                            torch.save(all_logits, save_path)
                            print(f"📦 Saved Chunk {chunk_idx}: {save_path}")
                            self._save_processed_samples(stage_name, subset, processed_total)
                            
                            # [V27.5 FIX] Memory Cleanup Strategy
                            del all_logits
                            all_logits = []
                            processed_in_current_chunk = 0
                            chunk_idx += 1

                            # Force VRAM cleanup to prevent fragmentation over long runs
                            import gc
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            pbar.set_description(f"Distilling {stage_name} (Chunk {chunk_idx})")

                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"⚠️ Error processing sample: {e}")
                    continue
        
        # [FIX] Flush remaining buffer inputs
        if len(buffer_inputs) > 0:
            max_len = max([t.size(0) for t in buffer_inputs])
            padded_batch = torch.stack([
                F.pad(t, (0, max_len - t.size(0)), value=self.tokenizer.pad_token_id)
                for t in buffer_inputs
            ]).to(self.device)
            
            outputs = self.teacher_model(padded_batch)
            logits = outputs.logits

            for i in range(logits.size(0)):
                all_logits.append(logits[i].cpu().clone())
                processed_total += 1

        # Save final partial chunk
        if len(all_logits) > 0:
            save_path = self.logits_dir / f"{stage_name}_{subset}_part_{chunk_idx}.pt"
            torch.save(all_logits, save_path)
            logger.info(f"✅ Saved Final Chunk {chunk_idx}: {save_path}")
            self._save_processed_samples(stage_name, subset, processed_total)

        return self.logits_dir

    def get_precomputed_loader(self, stage_name):
        """
        Returns a dataloader that yields (input_ids, teacher_logits) from disk.
        """
        return PrecomputedLogitsIterable(self.logits_dir, stage_name, subset="train")

    def has_precomputed_logits(self, stage_names, subset="train"):
        """
        Quick sanity check to ensure offline logits exist for all stages.
        """
        # TR: Tüm stage'ler için shard var mı?
        # EN: Do shards exist for all stages?
        for stage in stage_names:
            files = _list_logits_files(self.logits_dir, stage, subset=subset)
            if not files:
                return False
        return True


def _list_logits_files(logits_dir: Path, stage_name: str, subset: str = "train"):
    pattern = f"{stage_name}_{subset}_part_*.pt"
    files = sorted(logits_dir.glob(pattern), key=lambda p: _part_index(p.name))
    return files


def _part_index(name: str) -> int:
    # Extract chunk index from "stage_subset_part_N.pt"
    try:
        base = name.rsplit("_part_", 1)[-1]
        return int(base.split(".")[0])
    except Exception:
        return 0


def _resolve_vocab_size(default: int = 128256) -> int:
    """
    Resolve vocab size lazily to avoid import-time config coupling.
    Falls back to a Llama-compatible default when config is unavailable.
    """
    try:
        from config.config import cfg as _cfg
        return int(getattr(_cfg, "vocab_size", default))
    except Exception:
        return int(default)


def _reconstruct_dense_from_topk(
    item: dict,
    vocab_size: int,
    fill_value: float = -1e4,
) -> torch.Tensor:
    """
    Reconstruct a dense [seq, vocab] tensor from Top-K sparse teacher logits.

    Non-top-k positions are filled with a large negative finite value so the
    downstream KD softmax behaves like zero probability without `-inf` NaNs.
    """
    indices = item["indices"].long()
    values = item["values"].float()
    if indices.shape != values.shape or indices.dim() != 2:
        raise ValueError(
            "Top-K sparse logits must use matching [seq, top_k] indices/values tensors."
        )
    seq_len, _ = indices.shape
    dense_elements = seq_len * int(vocab_size)
    allow_large = os.getenv("TITAN_ALLOW_DENSE_TOPK_RECONSTRUCT", "0").strip() == "1"
    if dense_elements > TOPK_DENSE_MAX_ELEMENTS and not allow_large:
        raise RuntimeError(
            "Refusing dense Top-K logits reconstruction for "
            f"{seq_len}x{int(vocab_size)}={dense_elements} elements. "
            "Use sparse KD or set TITAN_ALLOW_DENSE_TOPK_RECONSTRUCT=1 for an explicit debug run."
        )

    dense = torch.full(
        (seq_len, vocab_size),
        fill_value=fill_value,
        dtype=torch.float32,
    )
    dense.scatter_(dim=1, index=indices, src=values)
    return dense


def _as_sparse_topk_payload(item: dict, vocab_size: int) -> dict:
    indices = item["indices"].long()
    values = item["values"].float()
    if indices.shape != values.shape or indices.dim() != 2:
        raise ValueError(
            "Top-K sparse logits must use matching [seq, top_k] indices/values tensors."
        )
    return {
        "format": "topk_sparse_v1",
        "indices": indices,
        "values": values,
        "vocab_size": int(item.get("vocab_size", vocab_size)),
        "top_k": int(item.get("top_k", indices.size(-1))),
    }


class PrecomputedLogitsIterable(IterableDataset):
    """
    Sequential iterator over precomputed logits shards.
    Each yielded item is a single sample's logits tensor.
    """

    def __init__(self, logits_dir: Path, stage_name: str, subset: str = "train") -> None:
        super().__init__()
        self.logits_dir = Path(logits_dir)
        self.stage_name = stage_name
        self.subset = subset
        self.files = _list_logits_files(self.logits_dir, stage_name, subset=subset)

    def __iter__(self):
        # TR: Shard'ları sırayla oku ve sample başına logits üret
        # EN: Read shards sequentially and yield logits per sample
        if not self.files:
            raise RuntimeError(
                f"Precomputed logits not found for stage '{self.stage_name}' "
                f"(subset={self.subset}) in {self.logits_dir}"
            )
        for file in self.files:
            chunk = torch.load(file, map_location="cpu", weights_only=False)
            vocab_size = _resolve_vocab_size()
            # Support dict wrapper payloads with optional metadata
            if isinstance(chunk, dict):
                if "logits" not in chunk:
                    raise RuntimeError(f"Invalid logits shard wrapper: {file}")
                vocab_size = int(chunk.get("vocab_size", vocab_size))
                chunk = chunk["logits"]
            if not isinstance(chunk, (list, tuple)):
                raise RuntimeError(f"Invalid logits shard format: {file}")
            for item in chunk:
                if isinstance(item, dict) and "indices" in item and "values" in item:
                    yield _as_sparse_topk_payload(item, vocab_size)
                elif isinstance(item, torch.Tensor):
                    yield item
                else:
                    raise RuntimeError(
                        f"Unknown logits item type: {type(item)} in {file}"
                    )

if __name__ == "__main__":
    import argparse
    import sys
    
    # Setup Paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    sys.path.insert(0, str(project_root))
    
    from config.config import cfg
    from scripts.data_pipeline import STAGE_DIRS
    
    parser = argparse.ArgumentParser(description="Distillation Manager CLI")
    parser.add_argument("--stage", type=int, required=True, help="Stage number (1-5)")
    args = parser.parse_args()
    
    print(f"🔧 Distillation Manager: Launching for Stage {args.stage}")
    
    # Load Tokenizer
    hf_token = os.environ.get("HF_TOKEN")
    if bool(getattr(cfg, "require_gated_teacher", False)) and not hf_token:
        print("❌ HF_TOKEN missing under require_gated_teacher=true.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    manager = DistillationManager(cfg, tokenizer)
    
    # Locate dataset file
    stage_dir = STAGE_DIRS.get(args.stage)
    if not stage_dir:
        print(f"❌ Invalid stage: {args.stage}")
        sys.exit(1)
        
    dataset_path = project_root / stage_dir / f"stage{args.stage}_data.jsonl"
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)
        
    print(f"📂 Loading dataset from: {dataset_path}")
    
    # Streaming / Tailing Logic
    import time
    import json
    
    class TailingDataset:
        def __init__(self, path, done_signal_path):
            self.path = path
            self.done_signal_path = done_signal_path
            
        def __iter__(self):
            with open(self.path, "r") as f:
                while True:
                    line = f.readline()
                    if line:
                        if line.strip():
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                # Safe from half-written lines
                                continue
                    else:
                        # End of file reached
                        if self.done_signal_path.exists():
                            # Check if valid EOF
                            print("🏁 Done signal detected. Finishing Tailing.")
                            break
                        else:
                            # Wait for more data
                            time.sleep(0.1)
                            
    # Locate Done signal
    done_signal_path = project_root / stage_dir / f"stage{args.stage}_done.signal"
    dataset = TailingDataset(dataset_path, done_signal_path)

    # Run Pre-computation (Streaming)
    manager.precompute_logits(dataset, f"stage{args.stage}")

    print("✅ Distillation CLI Complete.")
