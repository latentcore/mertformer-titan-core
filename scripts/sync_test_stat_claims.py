#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
STAT_RE = re.compile(r"\d+\s+passed,\s*\d+\s+skipped")

TARGETS = [
    ROOT / "README.md",
    ROOT / "README_TR.md",
    ROOT / "reports" / "release_snapshot.md",
    ROOT / "reports" / "release_snapshot_TR.md",
    ROOT / "reports" / "one_command_full_sop_summary.md",
]


def sync_file(path: Path, stat: str) -> bool:
    text = path.read_text(encoding="utf-8")
    updated = STAT_RE.sub(stat, text)
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize current pytest pass/skipped claim markers.")
    parser.add_argument("--test-stat", default=os.environ.get("MERTFORMER_EXPECTED_TEST_STAT", ""))
    args = parser.parse_args()
    stat = args.test_stat.strip()
    if not re.fullmatch(r"\d+\s+passed,\s*\d+\s+skipped", stat):
        raise SystemExit(f"invalid or missing --test-stat: {stat!r}")

    changed = []
    missing = []
    for path in TARGETS:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        if sync_file(path, stat):
            changed.append(str(path.relative_to(ROOT)))

    print(
        {
            "status": "ok",
            "test_stat": stat,
            "changed": changed,
            "missing": missing,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
