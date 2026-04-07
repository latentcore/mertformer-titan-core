#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CANONICAL_ONEFILE = ROOT / "scripts" / "chess_5080_onefile.py"
DEFAULT_GUI_DIR = ROOT / "apps" / "chess_gui"
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


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(ROOT)
        rel_text = rel.as_posix()
        return "<REPO_ROOT>" if not rel_text else f"<REPO_ROOT>/{rel_text}"
    except ValueError:
        return str(resolved)


def build_report(gui_dir: Path, target_file: Path, status: str, reason: str, copied: bool) -> dict:
    canonical_sha = sha256_file(CANONICAL_ONEFILE) if CANONICAL_ONEFILE.exists() else ""
    target_exists = target_file.exists()
    target_sha = sha256_file(target_file) if target_exists else ""
    return {
        "schema": "chess_gui_onefile_sync_report_v1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "canonical_onefile": display_path(CANONICAL_ONEFILE),
        "gui_dir": display_path(gui_dir),
        "gui_onefile": display_path(target_file),
        "status": status,
        "reason": reason,
        "copied": copied,
        "local_copy_present": target_exists,
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
            f"- local_copy_present: `{report['local_copy_present']}`",
            f"- hashes_match: `{report['hashes_match']}`",
            f"- canonical_onefile: `{report['canonical_onefile']}`",
            f"- gui_onefile: `{report['gui_onefile']}`",
            f"- canonical_sha256: `{report['canonical_sha256'] or 'missing'}`",
            f"- gui_sha256: `{report['gui_sha256'] or 'missing'}`",
        ]
    )


def supports_repo_canonical_fallback(gui_dir: Path) -> bool:
    try:
        if gui_dir.resolve() != DEFAULT_GUI_DIR.resolve():
            return False
    except FileNotFoundError:
        return False
    return (gui_dir / "play_mertformer_chess_web.py").exists()


def sync_onefile(gui_dir: Path, check_only: bool) -> dict:
    target_file = gui_dir / "chess_5080_onefile.py"
    if not CANONICAL_ONEFILE.exists():
        return build_report(gui_dir, target_file, "error", "canonical_onefile_missing", copied=False)
    if not gui_dir.exists():
        return build_report(gui_dir, target_file, "skipped", "gui_dir_missing", copied=False)
    if target_file.exists() and sha256_file(target_file) == sha256_file(CANONICAL_ONEFILE):
        return build_report(gui_dir, target_file, "up_to_date", "hashes_match", copied=False)
    if check_only:
        if not target_file.exists() and supports_repo_canonical_fallback(gui_dir):
            return build_report(gui_dir, target_file, "canonical_fallback_ready", "local_copy_missing_repo_canonical_fallback_available", copied=False)
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
    return 0 if report["status"] in {"up_to_date", "synced", "skipped", "canonical_fallback_ready"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
