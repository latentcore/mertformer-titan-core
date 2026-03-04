#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail if non-ASCII path is found")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="reports/unicode_path_guard_report.json")
    ap.add_argument("--fail-on-hit", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    ignore = {".git", ".titan-venv", ".lint-venv", ".venv", ".idea", "__pycache__"}
    hits = []

    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if any(part in ignore for part in rel.parts):
            continue
        if not is_ascii(str(rel)):
            hits.append(str(rel))

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": ".",
        "non_ascii_count": len(hits),
        "non_ascii_paths": sorted(hits),
        "ok": len(hits) == 0,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.fail_on_hit and hits:
        print(f"FAIL: non-ASCII paths found ({len(hits)})")
        return 1

    print(f"OK: unicode path guard; hits={len(hits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
