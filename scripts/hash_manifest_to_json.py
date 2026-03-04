#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_shasum_line(line: str) -> tuple[str, str] | None:
    line = line.rstrip("\n")
    if not line.strip():
        return None
    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        return None
    digest, path = parts[0].strip(), parts[1].strip()
    if path.startswith("*"):
        path = path[1:]
    return digest, path


def file_meta(path: Path) -> dict:
    try:
        st = path.stat()
        return {
            "size_bytes": int(st.st_size),
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            "exists": True,
        }
    except FileNotFoundError:
        return {"size_bytes": None, "mtime_utc": None, "exists": False}


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert shasum output to JSON manifest")
    ap.add_argument("--base", default=os.getcwd(), help="Base directory for resolving relative paths")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    entries = []

    for raw in sys.stdin:
        parsed = parse_shasum_line(raw)
        if parsed is None:
            continue
        digest, raw_path = parsed
        p = Path(raw_path)
        resolved = (base / p).resolve() if not p.is_absolute() else p.resolve()
        meta = file_meta(resolved)
        try:
            rel_path = str(resolved.relative_to(base))
        except Exception:
            rel_path = str(p)
        entries.append(
            {
                "sha256": digest,
                "path": rel_path,
                **meta,
            }
        )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(entries),
        "entries": entries,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
