#!/usr/bin/env python3
"""
Teacher-logit top-k shard disk-provisioning calculator.

[2026-07-12] BACKLOG I.3 #26: "top_k'yi sec ve karsilik gelen diski provision
et". Precomputed teacher-logit shards (scripts/precompute_logits_topk.py)
store, per token, top_k (index, logit) pairs. This is real arithmetic against
the repo's own FACTS.json token budget and dtype choices, not a guess -- it
lets a human pick top_k with the actual disk cost in front of them before
provisioning a machine.

Usage:
    python scripts/estimate_shard_disk.py --top-k 20
    python scripts/estimate_shard_disk.py --top-k 8 16 20 32 64   # compare several
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACTS_PATH = PROJECT_ROOT / "reports" / "FACTS.json"

INDEX_DTYPE_BYTES = 4   # int32 vocab index (vocab_size=128256 needs >16 bits, so int32)
LOGIT_DTYPE_BYTES = 2   # float16/bfloat16 logit value
BYTES_PER_TOKEN_PER_K = INDEX_DTYPE_BYTES + LOGIT_DTYPE_BYTES  # 6 bytes per (index, logit) pair
OVERHEAD_FACTOR = 1.15  # shard framing/manifest/padding overhead, conservative


def estimate(top_k: int, target_tokens: int) -> dict:
    raw_bytes = target_tokens * top_k * BYTES_PER_TOKEN_PER_K
    with_overhead = raw_bytes * OVERHEAD_FACTOR
    return {
        "top_k": top_k,
        "target_tokens": target_tokens,
        "raw_bytes": int(raw_bytes),
        "raw_gb": round(raw_bytes / 1e9, 2),
        "with_overhead_gb": round(with_overhead / 1e9, 2),
        "with_overhead_tb": round(with_overhead / 1e12, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate on-disk size of precomputed teacher-logit top-k shards.")
    parser.add_argument("--top-k", type=int, nargs="+", default=[8, 16, 20, 32, 64], help="one or more top_k values to compare")
    parser.add_argument("--target-tokens", type=int, default=None, help="override FACTS.json target_tokens_min")
    args = parser.parse_args(argv)

    facts = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
    target_tokens = args.target_tokens or int(facts["target_tokens_min"])

    rows = [estimate(k, target_tokens) for k in sorted(set(args.top_k))]

    print(f"Target tokens (from FACTS.json unless --target-tokens given): {target_tokens:,}")
    print(f"Bytes/token/k = {BYTES_PER_TOKEN_PER_K} (int32 index + fp16 logit), overhead factor = {OVERHEAD_FACTOR}")
    print()
    print(f"{'top_k':>6} | {'raw GB':>10} | {'with overhead GB':>18} | {'with overhead TB':>18}")
    print("-" * 62)
    for row in rows:
        print(f"{row['top_k']:>6} | {row['raw_gb']:>10} | {row['with_overhead_gb']:>18} | {row['with_overhead_tb']:>18}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
