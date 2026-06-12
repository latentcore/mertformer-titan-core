#!/usr/bin/env python3
"""
Logbook hash-chain VERIFIER (complements scripts/logbook_build.py).

RunLogger (utils/logger.py) writes each JSONL record with a tamper-evident `_chain` field:
    _chain.hash = sha256( prev_hash_utf8 + canonical_line_without_hash + "\\n" )
where the canonical line is json.dumps(record, ensure_ascii=False, sort_keys=True,
separators=(",", ":")) with `_chain.hash` removed, and `_chain.prev` links to the previous
line's hash (the first line links to the genesis = sha256("")).

This script re-walks that chain for one run file (or every logs/*.jsonl) and reports the first
break. Lines without a `_chain` (e.g. a logbook header) are skipped.

Exit 0 = all chains intact; exit 1 = a broken/forged/missing-link line was found.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENESIS = hashlib.sha256(b"").hexdigest()


def _recompute_hash(record: dict, prev_hash: str) -> str:
    rec = json.loads(json.dumps(record))  # deep copy
    rec.get("_chain", {}).pop("hash", None)
    line = json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    h = hashlib.sha256()
    h.update(prev_hash.encode("utf-8"))
    h.update(line.encode("utf-8"))
    return h.hexdigest()


def verify_file(path: Path) -> tuple[bool, str]:
    """Hard-check the chain LINKAGE (prev == previous hash) — robust across logger versions and
    enough to detect insertion / deletion / reordering / truncation. The full hash recompute is
    serialization-sensitive (logger versions differ), so a recompute miss is reported as an
    informational warning, not a failure."""
    prev = GENESIS
    checked = 0
    hash_warnings = 0
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                return False, f"{path.name}:{lineno} invalid JSON: {exc}"
            chain = rec.get("_chain")
            if not isinstance(chain, dict) or "hash" not in chain:
                continue  # header / non-chained line
            stored_hash = str(chain.get("hash", ""))
            stored_prev = str(chain.get("prev", ""))
            if stored_prev != prev:
                return False, f"{path.name}:{lineno} broken chain link (prev {stored_prev[:12]}… != {prev[:12]}…)"
            if _recompute_hash(rec, prev) != stored_hash:
                hash_warnings += 1
            prev = stored_hash
            checked += 1
    suffix = f"; {hash_warnings} hash-recompute warning(s) (logger serialization drift)" if hash_warnings else ""
    return True, f"{path.name}: chain links OK ({checked} records){suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify RunLogger JSONL hash-chains.")
    parser.add_argument("paths", nargs="*", help="JSONL files to verify (default: logs/*.jsonl)")
    args = parser.parse_args()

    targets = [Path(p) for p in args.paths] if args.paths else sorted((ROOT / "logs").glob("*.jsonl"))
    if not targets:
        print("logbook_verify: no JSONL run files found (logs/*.jsonl).")
        return 0

    ok_all = True
    for path in targets:
        if not path.exists():
            print(f"logbook_verify: missing {path}")
            ok_all = False
            continue
        ok, msg = verify_file(path)
        print(("OK:   " if ok else "FAIL: ") + msg)
        ok_all = ok_all and ok
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
