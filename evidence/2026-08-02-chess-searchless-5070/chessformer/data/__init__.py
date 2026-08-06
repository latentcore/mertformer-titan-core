"""Data pipeline: download -> packed shards -> memmap dataset."""
from .download import DownloadResult, download_prefix, estimate_bytes_for_positions, head, sha256_file
from .preprocess import build_dataset, iter_records, parse_record, split_for_key
from .dataset import (
    PackedChessDataset,
    build_hl_gauss_table,
    collate_packed,
    expand_batch_on_device,
    level_to_win_prob,
    load_manifest,
    make_dataloader,
)

__all__ = [
    "DownloadResult",
    "download_prefix",
    "estimate_bytes_for_positions",
    "head",
    "sha256_file",
    "build_dataset",
    "iter_records",
    "parse_record",
    "split_for_key",
    "PackedChessDataset",
    "build_hl_gauss_table",
    "collate_packed",
    "expand_batch_on_device",
    "level_to_win_prob",
    "load_manifest",
    "make_dataloader",
]
