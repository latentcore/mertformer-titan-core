"""Hard preflight gate: teacher-logit shards must align to the student token stream.

TR: Mevcut preflight kontrolleri yalnız SAYI/VARLIK bakıyordu; per-sample KİMLİK
    bakmıyordu. Bu betik her stage için JSONL'i training ile AYNI şekilde yeniden
    paketleyip, saklanan shard kimliğini (identity) ilk-N + rastgele-K dizide HARD
    doğrular. Hizasızlık = sessiz KD bozulması; bu kapı onu gürültülü bir ön-uçuş
    hatasına çevirir.
EN: Existing preflight checks only counted/existence-checked shards; they never
    verified per-sample IDENTITY. For each stage this script re-packs the JSONL
    exactly as training does and HARD-asserts the stored shard identity for the
    first-N + random-K sequences. Misalignment = silent KD corruption; this gate
    turns it into a loud preflight failure.

Exit codes (repo convention):
  0   all requested stages aligned        (LOGIT_ALIGNMENT:PASS reason_code=ALIGNED)
  1   mismatch / drift / runtime error    (LOGIT_ALIGNMENT:FAIL reason_code=...)
  4   dataset or shards missing
  5   shards present but incomplete coverage
  130 interrupted
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from config.config import cfg  # noqa: E402
from scripts.precompute_logits_topk import STAGE_FILES, _stage_shards  # noqa: E402
from train.packing import (  # noqa: E402
    TOPK_PACKED_FORMAT,
    LogitAlignmentError,
    assert_sequence_identity,
    extract_row_text,
    iter_packed_sequences,
)
from utils.tokenizer_resolver import resolve_tokenizer, tokenizer_identity  # noqa: E402

DEFAULT_FIRST_N = 64
DEFAULT_RANDOM_K = 64
DEFAULT_SEED = 1453


def _stage_rows(jsonl_path: Path):
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


def _load_stored_identities(shards):
    """Sequential per-item identities across all shards + first wrapper metadata."""
    meta = None
    identities = []
    for sh in shards:
        chunk = torch.load(sh, map_location="cpu", weights_only=False)
        if isinstance(chunk, dict):
            if meta is None:
                meta = {k: chunk.get(k) for k in (
                    "format", "max_seq_len", "pad_id", "eos_id",
                    "tokenizer_identity", "packer_version",
                )}
            items = chunk.get("logits", [])
        else:
            items = chunk
        for it in items:
            identities.append(it.get("identity") if isinstance(it, dict) else None)
    return meta, identities


def validate_stage(
    stage_num: int,
    logits_dir: Path,
    tokenizer,
    first_n: int = DEFAULT_FIRST_N,
    random_k: int = DEFAULT_RANDOM_K,
    seed: int = DEFAULT_SEED,
) -> dict:
    stage_name = f"stage{stage_num}"
    jsonl_path = STAGE_FILES[stage_num]
    shards = _stage_shards(logits_dir, stage_name)

    if not jsonl_path.exists() or not shards:
        return {"stage": stage_name, "status": "MISSING", "reason_code": "SHARDS_OR_DATASET_MISSING"}

    meta, stored = _load_stored_identities(shards)
    if not meta or meta.get("format") != TOPK_PACKED_FORMAT:
        return {"stage": stage_name, "status": "FAIL", "reason_code": "LEGACY_OR_UNPACKED_FORMAT"}

    cur_tok = tokenizer_identity(tokenizer, cfg)
    stored_tok = meta.get("tokenizer_identity") or {}
    if (stored_tok.get("name_or_path") != cur_tok.get("name_or_path")
            or int(stored_tok.get("vocab_size", -1)) != int(cur_tok.get("vocab_size", -2))):
        return {
            "stage": stage_name, "status": "FAIL", "reason_code": "TOKENIZER_IDENTITY_DRIFT",
            "stored_tokenizer": stored_tok, "current_tokenizer": cur_tok,
        }

    max_seq = int(meta.get("max_seq_len") or getattr(cfg, "max_seq_len", 512))
    pad_id = int(meta.get("pad_id") if meta.get("pad_id") is not None else (tokenizer.pad_token_id or 0))
    eos_id = int(meta.get("eos_id") if meta.get("eos_id") is not None else (tokenizer.eos_token_id or pad_id))

    total_stored = len(stored)
    targets = set(range(min(first_n, total_stored)))
    if total_stored > first_n:
        rng = random.Random(seed)
        targets |= set(rng.sample(range(total_stored), min(random_k, total_stored)))

    count = 0
    for i, seq in enumerate(_iter_packed(jsonl_path, tokenizer, max_seq, eos_id, pad_id)):
        if i in targets:
            if i >= total_stored:
                return {"stage": stage_name, "status": "FAIL", "reason_code": "INCOMPLETE_COVERAGE",
                        "seq_index": i, "stored": total_stored}
            try:
                assert_sequence_identity(seq["input_ids"], seq["true_len"], stored[i])
            except LogitAlignmentError as exc:
                return {"stage": stage_name, "status": "FAIL", "reason_code": "IDENTITY_MISMATCH",
                        "seq_index": i, "detail": str(exc)}
        count += 1

    if count != total_stored:
        return {"stage": stage_name, "status": "FAIL", "reason_code": "INCOMPLETE_COVERAGE",
                "regenerated": count, "stored": total_stored}

    return {"stage": stage_name, "status": "PASS", "reason_code": "ALIGNED",
            "checked": len(targets), "sequences": total_stored}


def _iter_packed(jsonl_path, tokenizer, max_seq, eos_id, pad_id):
    return iter_packed_sequences(_stage_rows(jsonl_path), tokenizer, max_seq, eos_id, pad_id)


def validate_all_stages(logits_dir: Path, stages=None, **kwargs) -> dict:
    """Programmatic entry used by titan_preflight / has_precomputed_logits."""
    stages = stages or [1, 2, 3, 4, 5]
    tokenizer = resolve_tokenizer(cfg)
    results = {}
    overall = "PASS"
    for s in stages:
        res = validate_stage(s, logits_dir, tokenizer, **kwargs)
        results[res["stage"]] = res
        if res["status"] == "MISSING" and overall == "PASS":
            overall = "MISSING"
        elif res["status"] == "FAIL":
            overall = "FAIL"
    return {"status": overall, "checks": results}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate teacher-logit shard alignment to the student stream.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all-stages", action="store_true")
    g.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5])
    p.add_argument("--logits-dir", type=Path,
                   default=Path(getattr(cfg, "precomputed_logits_path", "datasets/logits")))
    p.add_argument("--first-n", type=int, default=DEFAULT_FIRST_N)
    p.add_argument("--random-k", type=int, default=DEFAULT_RANDOM_K)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--json-report", type=Path, default=Path("logs/preflight/logit_alignment.json"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stages = [1, 2, 3, 4, 5] if args.all_stages else [args.stage]
    logits_dir = args.logits_dir.resolve()
    started = time.time()

    result = validate_all_stages(
        logits_dir, stages=stages,
        first_n=args.first_n, random_k=args.random_k, seed=args.seed,
    )
    result["elapsed_s"] = round(time.time() - started, 3)

    try:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass

    statuses = {c["status"] for c in result["checks"].values()}
    if "FAIL" in statuses:
        bad = [c for c in result["checks"].values() if c["status"] == "FAIL"]
        code = bad[0].get("reason_code", "FAIL")
        print(f"LOGIT_ALIGNMENT:FAIL reason_code={code} detail={bad[0]}")
        return 1
    if statuses == {"MISSING"} or (statuses and statuses <= {"MISSING"}):
        print("LOGIT_ALIGNMENT:SKIP reason_code=SHARDS_OR_DATASET_MISSING")
        return 4
    if "MISSING" in statuses:
        # some stages missing shards -> incomplete coverage for the requested set
        print("LOGIT_ALIGNMENT:FAIL reason_code=INCOMPLETE_COVERAGE")
        return 5
    print("LOGIT_ALIGNMENT:PASS reason_code=ALIGNED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
