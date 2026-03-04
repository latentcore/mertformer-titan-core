#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect duplicate zip files by sha256")
    ap.add_argument("--root", action="append", default=["packages", "artifacts", str(Path.home() / "Documents")])
    ap.add_argument("--out", default="reports/duplicate_zip_guard_report.json")
    ap.add_argument("--fail-on-duplicates", action="store_true")
    args = ap.parse_args()

    zips = []
    for r in args.root:
        rp = Path(r)
        if not rp.exists():
            continue
        for z in rp.rglob("*.zip"):
            zips.append(z)

    digest_map: dict[str, list[str]] = {}
    for z in sorted(set(zips)):
        d = sha256(z)
        digest_map.setdefault(d, []).append(str(z))

    duplicates = {d: ps for d, ps in digest_map.items() if len(ps) > 1}
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "zip_count": len(zips),
        "unique_sha_count": len(digest_map),
        "duplicate_group_count": len(duplicates),
        "duplicate_groups": duplicates,
        "ok": len(duplicates) == 0,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.fail_on_duplicates and duplicates:
        print(f"FAIL: duplicate zip groups={len(duplicates)}")
        return 1

    print(f"OK: duplicate zip guard; groups={len(duplicates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
