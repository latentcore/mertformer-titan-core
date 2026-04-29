#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

HOME = Path.home()
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = ROOT / "reports" / "cleanup_scoped_closure_junk_report.json"
DEFAULT_INTAKE = ROOT / "reports" / "scoped_external_intake_matrix.json"

JUNK_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
JUNK_SUFFIXES = {".pyc", ".pyo"}
JUNK_FILES = {".DS_Store"}


def unlock_path(path: Path) -> None:
    if not path.exists():
        return
    for args in (
        ["chflags", "nouchg", str(path)],
        ["chflags", "noschg", str(path)],
        ["chflags", "nouchg,noschg", str(path)],
    ):
        subprocess.run(args, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        current_mode = path.stat().st_mode
        os.chmod(path, current_mode | 0o200)
    except Exception:
        pass


def delete_path(path: Path) -> str | None:
    try:
        unlock_path(path)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def sanitize_path(path: Path) -> str:
    resolved = path.resolve()
    root_resolved = ROOT.resolve()
    home_resolved = HOME.resolve()
    resolved_str = str(resolved)
    root_str = str(root_resolved)
    home_str = str(home_resolved)
    if resolved_str == root_str:
        return "<REPO_ROOT>"
    if resolved_str.startswith(root_str + "/"):
        return resolved_str.replace(root_str, "<REPO_ROOT>", 1)
    if resolved_str == home_str:
        return "<HOME>"
    if resolved_str.startswith(home_str + "/"):
        return resolved_str.replace(home_str, "<HOME>", 1)
    return resolved_str


def resolve_sanitized_path(path_str: str) -> Path:
    if path_str == "<REPO_ROOT>":
        return ROOT
    if path_str.startswith("<REPO_ROOT>/"):
        return ROOT / path_str.removeprefix("<REPO_ROOT>/")
    if path_str == "<HOME>":
        return HOME
    if path_str.startswith("<HOME>/"):
        return HOME / path_str.removeprefix("<HOME>/")
    return Path(path_str)


def collect_roots(intake_path: Path) -> List[Path]:
    roots = [ROOT]
    if intake_path.exists():
        payload = json.loads(intake_path.read_text(encoding="utf-8"))
        for entry in payload.get("entries", []):
            path = resolve_sanitized_path(str(entry.get("path", "")))
            mutation_policy = str(entry.get("mutation_policy") or "")
            if path.exists() and path.is_dir() and mutation_policy in {"project_safe_cleanup", "project_sync"}:
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
    errors: List[Dict[str, str]] = []

    for root in roots:
        for path in root.rglob("*"):
            if path.name in JUNK_DIRS or path.name in JUNK_FILES or path.suffix in JUNK_SUFFIXES:
                found.append(str(path))
                if args.apply:
                    error = delete_path(path)
                    if error is None:
                        removed.append(str(path))
                    else:
                        errors.append({"path": str(path), "error": error})

    stale_deleted: List[str] = []
    if args.delete_stale_zips and intake_path.exists():
        payload = json.loads(intake_path.read_text(encoding="utf-8"))
        for entry in payload.get("entries", []):
            if entry.get("disposition") != "delete_as_stale_generated":
                continue
            if entry.get("mutation_policy") not in {"project_safe_cleanup", "project_sync"}:
                continue
            path = resolve_sanitized_path(str(entry.get("path", "")))
            if path.exists() and path.is_file():
                found.append(str(path))
                if args.apply:
                    error = delete_path(path)
                    if error is None:
                        removed.append(str(path))
                        stale_deleted.append(str(path))
                    else:
                        errors.append({"path": str(path), "error": error})

    report = {
        "roots": [sanitize_path(path) for path in roots],
        "found_count": len(found),
        "removed_count": len(removed),
        "stale_deleted_count": len(stale_deleted),
        "error_count": len(errors),
        "found": [sanitize_path(Path(path)) for path in found],
        "removed": [sanitize_path(Path(path)) for path in removed],
        "stale_deleted": [sanitize_path(Path(path)) for path in stale_deleted],
        "errors": [
            {"path": sanitize_path(Path(item["path"])), "error": item["error"]}
            for item in errors
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": sanitize_path(report_path),
                "found": len(found),
                "removed": len(removed),
                "errors": len(errors),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
