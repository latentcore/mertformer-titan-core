#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CANONICAL_ONEFILE = ROOT / "scripts" / "chess_5080_onefile.py"
DEFAULT_GUI_DIR = Path("/Users/mertyunlu/Documents/MertFormer_Chess_GUI")
REPORT_JSON = ROOT / "reports" / "chess_gui_onefile_sync_report.json"
REPORT_MD = ROOT / "reports" / "chess_gui_onefile_sync_report.md"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_report(gui_dir: Path, target_file: Path, status: str, reason: str, copied: bool) -> dict:
    canonical_sha = sha256_file(CANONICAL_ONEFILE) if CANONICAL_ONEFILE.exists() else ""
    target_exists = target_file.exists()
    target_sha = sha256_file(target_file) if target_exists else ""
    return {
        "schema": "chess_gui_onefile_sync_report_v1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "canonical_onefile": str(CANONICAL_ONEFILE),
        "gui_dir": str(gui_dir),
        "gui_onefile": str(target_file),
        "status": status,
        "reason": reason,
        "copied": copied,
        "canonical_sha256": canonical_sha,
        "gui_sha256": target_sha,
        "hashes_match": bool(canonical_sha and target_sha and canonical_sha == target_sha),
    }


def build_report_md(report: dict) -> str:
    return "\n".join(
        [
            "# Chess GUI Onefile Sync Report",
            "",
            f"- status: `{report['status']}`",
            f"- reason: `{report['reason']}`",
            f"- copied: `{report['copied']}`",
            f"- hashes_match: `{report['hashes_match']}`",
            f"- canonical_onefile: `{report['canonical_onefile']}`",
            f"- gui_onefile: `{report['gui_onefile']}`",
            f"- canonical_sha256: `{report['canonical_sha256'] or 'missing'}`",
            f"- gui_sha256: `{report['gui_sha256'] or 'missing'}`",
        ]
    )


def sync_onefile(gui_dir: Path, check_only: bool) -> dict:
    target_file = gui_dir / "chess_5080_onefile.py"
    if not CANONICAL_ONEFILE.exists():
        return build_report(gui_dir, target_file, "error", "canonical_onefile_missing", copied=False)
    if not gui_dir.exists():
        return build_report(gui_dir, target_file, "skipped", "gui_dir_missing", copied=False)
    if target_file.exists() and sha256_file(target_file) == sha256_file(CANONICAL_ONEFILE):
        return build_report(gui_dir, target_file, "up_to_date", "hashes_match", copied=False)
    if check_only:
        return build_report(gui_dir, target_file, "drift_detected", "hash_mismatch_or_missing", copied=False)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CANONICAL_ONEFILE, target_file)
    return build_report(gui_dir, target_file, "synced", "canonical_copy_written", copied=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync the canonical chess onefile into the local GUI workspace.")
    parser.add_argument("--gui-dir", default=str(DEFAULT_GUI_DIR))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    report = sync_onefile(Path(args.gui_dir).expanduser().resolve(), check_only=args.check_only)
    write_text(REPORT_JSON, json.dumps(report, indent=2, ensure_ascii=False))
    write_text(REPORT_MD, build_report_md(report))
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] in {"up_to_date", "synced", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
