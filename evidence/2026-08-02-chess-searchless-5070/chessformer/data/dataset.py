"""Memmap-backed dataset and a transfer-minimal collate.

THROUGHPUT DESIGN
-----------------
The dense tensors the loss needs -- ``legal_mask`` and ``policy_target``, both
``[B, 4208]`` -- are never built on the CPU and never crossed over PCIe. At
batch 512 that would be ~17 MB of host-to-device traffic *per step*. Instead the
collate emits the compact ragged form (~150 KB) and
:func:`expand_batch_on_device` materializes the dense tensors directly on the
GPU with two scatters. That is a ~100x reduction in H2D bytes per step.

Value targets are a table lookup: HL-Gauss distributions are precomputed once
for all 1024 quantized win-probability levels, so the per-sample cost is a
gather rather than 128 ``erf`` evaluations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..board import VOCAB_SIZE, hl_gauss_target
from ..config import HL_GAUSS_SIGMA_RATIO
from .preprocess import MAX_PV_SLOTS, SHARD_PREFIX, VALUE_LEVELS

_ARRAYS = (
    "pieces", "meta", "pv_ids", "pv_w", "value_level", "wdl",
    "legal_offsets", "legal_ids",
)


@dataclass
class _Shard:
    path: Path
    positions: int
    arrays: Dict[str, np.ndarray]


class PackedChessDataset(Dataset):
    """Random access over the packed shards of one split."""

    def __init__(self, root: Path, split: str, limit: Optional[int] = None) -> None:
        self.root = Path(root) / split
        self.split = split
        if not self.root.exists():
            raise FileNotFoundError(f"no such split directory: {self.root}")

        shard_dirs = sorted(p for p in self.root.iterdir() if p.is_dir() and p.name.startswith(SHARD_PREFIX))
        if not shard_dirs:
            raise FileNotFoundError(f"no shards under {self.root}")

        self.shards: List[_Shard] = []
        counts: List[int] = []
        for d in shard_dirs:
            arrays = {name: np.load(d / f"{name}.npy", mmap_mode="r") for name in _ARRAYS}
            n = int(arrays["value_level"].shape[0])
            self.shards.append(_Shard(path=d, positions=n, arrays=arrays))
            counts.append(n)

        self.shard_starts = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(np.asarray(counts, dtype=np.int64), out=self.shard_starts[1:])
        self.total = int(self.shard_starts[-1])
        self.limit = int(min(limit, self.total)) if limit else self.total

    def __len__(self) -> int:
        return self.limit

    def _locate(self, index: int) -> Tuple[_Shard, int]:
        shard_idx = int(np.searchsorted(self.shard_starts, index, side="right") - 1)
        return self.shards[shard_idx], index - int(self.shard_starts[shard_idx])

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        shard, local = self._locate(int(index))
        a = shard.arrays
        start = int(a["legal_offsets"][local])
        end = int(a["legal_offsets"][local + 1])
        return {
            "pieces": np.asarray(a["pieces"][local], dtype=np.int8),
            "meta": np.asarray(a["meta"][local], dtype=np.int8),
            "pv_ids": np.asarray(a["pv_ids"][local], dtype=np.int16),
            "pv_w": np.asarray(a["pv_w"][local], dtype=np.float32),
            "value_level": np.int64(a["value_level"][local]),
            "wdl": np.int64(a["wdl"][local]),
            "legal_ids": np.asarray(a["legal_ids"][start:end], dtype=np.int16),
        }

    def stats(self) -> Dict[str, object]:
        return {
            "split": self.split,
            "positions": self.total,
            "used": self.limit,
            "shards": len(self.shards),
            "bytes": int(sum(
                sum(p.stat().st_size for p in s.path.glob("*.npy")) for s in self.shards
            )),
        }


def collate_packed(samples: Sequence[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Compact collate: nothing here is ``[B, vocab]``-shaped."""
    batch_size = len(samples)
    lengths = np.fromiter((s["legal_ids"].shape[0] for s in samples), dtype=np.int64, count=batch_size)

    pieces = np.stack([s["pieces"] for s in samples])
    meta = np.stack([s["meta"] for s in samples])
    pv_ids = np.stack([s["pv_ids"] for s in samples])
    pv_w = np.stack([s["pv_w"] for s in samples])
    value_level = np.fromiter((s["value_level"] for s in samples), dtype=np.int64, count=batch_size)
    wdl = np.fromiter((s["wdl"] for s in samples), dtype=np.int64, count=batch_size)

    legal_flat = np.concatenate([s["legal_ids"] for s in samples]) if batch_size else np.zeros(0, np.int16)
    legal_rows = np.repeat(np.arange(batch_size, dtype=np.int64), lengths)

    return {
        "pieces": torch.from_numpy(pieces),
        "meta": torch.from_numpy(meta),
        "pv_ids": torch.from_numpy(pv_ids),
        "pv_w": torch.from_numpy(pv_w),
        "value_level": torch.from_numpy(value_level),
        "wdl": torch.from_numpy(wdl),
        "legal_flat": torch.from_numpy(legal_flat.astype(np.int64)),
        "legal_rows": torch.from_numpy(legal_rows),
    }


def build_hl_gauss_table(num_bins: int, sigma_ratio: float = HL_GAUSS_SIGMA_RATIO) -> torch.Tensor:
    """[VALUE_LEVELS, num_bins] lookup so per-sample target building is a gather."""
    levels = (np.arange(VALUE_LEVELS, dtype=np.float64) / float(VALUE_LEVELS - 1))
    table = hl_gauss_target(levels, num_bins, sigma_ratio=sigma_ratio)
    return torch.from_numpy(np.ascontiguousarray(table, dtype=np.float32))


def level_to_win_prob(levels: torch.Tensor) -> torch.Tensor:
    return levels.float() / float(VALUE_LEVELS - 1)


def expand_batch_on_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    hl_gauss_table: torch.Tensor,
    vocab_size: int = VOCAB_SIZE,
    non_blocking: bool = True,
) -> Dict[str, torch.Tensor]:
    """Move the compact batch to ``device`` and build the dense targets there."""
    moved = {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}
    batch_size = moved["pieces"].shape[0]

    legal_mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=device)
    if moved["legal_flat"].numel():
        legal_mask[moved["legal_rows"], moved["legal_flat"]] = True

    policy_target = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)
    pv_ids = moved["pv_ids"].long()
    pv_w = moved["pv_w"].float()
    used = pv_ids >= 0
    safe_ids = torch.where(used, pv_ids, torch.zeros_like(pv_ids))
    policy_target.scatter_add_(1, safe_ids, torch.where(used, pv_w, torch.zeros_like(pv_w)))
    # Rows already sum to 1 from preprocessing; renormalize defensively so a
    # truncated slot list can never produce an improper target.
    policy_target = policy_target / policy_target.sum(dim=1, keepdim=True).clamp(min=1e-8)

    value_target = hl_gauss_table.to(device, non_blocking=non_blocking).index_select(
        0, moved["value_level"].long()
    )

    return {
        "piece_ids": moved["pieces"].long(),
        "meta_ids": moved["meta"].long(),
        "legal_mask": legal_mask,
        "policy_target": policy_target,
        "value_target": value_target,
        "wdl_target": moved["wdl"].long(),
        "value_level": moved["value_level"].long(),
    }


def make_dataloader(
    dataset: PackedChessDataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    seed: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    drop_last: bool = True,
) -> torch.utils.data.DataLoader:
    """DataLoader wired for throughput.

    The upstream onefile ran ``num_workers=0`` with no ``pin_memory``, no
    prefetch and no persistent workers, so every batch was built on the main
    process between GPU steps. ``profile.py`` measures the difference.
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs: Dict[str, object] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "collate_fn": collate_packed,
        "pin_memory": bool(pin_memory),
        "drop_last": bool(drop_last),
        "generator": generator,
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return torch.utils.data.DataLoader(dataset, **kwargs)


def load_manifest(root: Path) -> Dict[str, object]:
    path = Path(root) / "dataset_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
