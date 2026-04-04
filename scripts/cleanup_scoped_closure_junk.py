#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = ROOT / "reports" / "cleanup_scoped_closure_junk_report.json"
DEFAULT_INTAKE = ROOT / "reports" / "scoped_external_intake_matrix.json"

JUNK_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
JUNK_SUFFIXES = {".pyc", ".pyo"}
JUNK_FILES = {".DS_Store"}


def delete_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def collect_roots(intake_path: Path) -> List[Path]:
    roots = [ROOT]
    if intake_path.exists():
        payload = json.loads(intake_path.read_text(encoding="utf-8"))
        for entry in payload.get("entries", []):
            path = Path(str(entry.get("path", "")))
            if path.exists() and path.is_dir():
                roots.append(path)
    deduped = []
    seen = set()
    for path in roots:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean scoped closure junk from repo + scoped external dirs")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--delete-stale-zips", action="store_true")
    parser.add_argument("--intake", default=str(DEFAULT_INTAKE))
    parser.add_argument("--out", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    intake_path = Path(args.intake)
    report_path = Path(args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    roots = collect_roots(intake_path)
    removed: List[str] = []
    found: List[str] = []

    for root in roots:
        for path in root.rglob("*"):
            if path.name in JUNK_DIRS or path.name in JUNK_FILES or path.suffix in JUNK_SUFFIXES:
                found.append(str(path))
                if args.apply:
                    delete_path(path)
                    removed.append(str(path))

    stale_deleted: List[str] = []
    if args.delete_stale_zips and intake_path.exists():
        payload = json.loads(intake_path.read_text(encoding="utf-8"))
        for entry in payload.get("entries", []):
            if entry.get("disposition") != "delete_as_stale_generated":
                continue
            path = Path(str(entry.get("path", "")))
            if path.exists() and path.is_file():
                found.append(str(path))
                if args.apply:
                    delete_path(path)
                    removed.append(str(path))
                    stale_deleted.append(str(path))

    report = {
        "roots": [str(path) for path in roots],
        "found_count": len(found),
        "removed_count": len(removed),
        "stale_deleted_count": len(stale_deleted),
        "found": found,
        "removed": removed,
        "stale_deleted": stale_deleted,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "found": len(found), "removed": len(removed)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
