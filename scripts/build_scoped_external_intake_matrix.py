#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

HOME = Path.home()
ROOT = Path(__file__).resolve().parent.parent
REPORT_JSON = ROOT / "reports" / "scoped_external_intake_matrix.json"
REPORT_MD = ROOT / "reports" / "scoped_external_intake_matrix.md"

SCOPED_PATTERNS = [
    HOME / "Desktop" / "MertFormer_45K_Launch_Bundle_20260401_2130",
    HOME / "Desktop" / "MertFormer_45K_Launch_Bundle_20260401_2130.zip",
    HOME / "Desktop" / "MertFormer_45K_Launch_Bundle_20260401_2130.zip.sha256",
    HOME / "Desktop" / "MertFormerStream",
    HOME / "Documents" / "mertformer_outputs_LINKEDIN_run_20260220_175540.zip",
    HOME / "Documents" / "mertformer_outputs_LINKEDIN_run_20260220_175540.zip.sha256",
    HOME / "Documents" / "mertformer-titan-core.zip",
    HOME / "Documents" / "mertformer_outputs",
    HOME / "Downloads" / "MertOS_Core" / "mertformer_cleanup_20260110_015202.log",
    HOME / "Downloads" / "MertOS_Core" / "mertformer_cleanup_20260110_015332.log",
    HOME / "Downloads" / "content" / "mertformer_outputs",
    Path("/Applications") / "mertformer-titan-core.zip",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def classify(path: Path, duplicate_rank: int) -> str:
    name = path.name.lower()
    if path.is_dir():
        if "bundle" in name:
            return "archive_into_closure_pack"
        if "outputs" in name:
            return "keep_as_external_artifact"
        return "keep_as_external_artifact"
    if duplicate_rank > 1 and path.suffix == ".zip":
        return "delete_as_stale_generated"
    if name.endswith(".sha256"):
        return "archive_into_closure_pack"
    if name.endswith(".log"):
        return "archive_into_closure_pack"
    if "bundle" in name:
        return "archive_into_closure_pack"
    if "mertformer" in name or "proje" in name:
        return "keep_as_external_artifact"
    return "promote_into_repo"


def collect_entries() -> List[Dict[str, object]]:
    present = [path for path in SCOPED_PATTERNS if path.exists()]
    zip_hash_groups: Dict[str, List[Path]] = defaultdict(list)
    for path in present:
        if path.is_file() and path.suffix == ".zip":
            zip_hash_groups[sha256_file(path)].append(path)

    entries: List[Dict[str, object]] = []
    for path in present:
        sha = None
        size = None
        duplicate_rank = 1
        if path.is_file():
            size = path.stat().st_size
            sha = sha256_file(path)
            if path.suffix == ".zip":
                duplicate_rank = len(zip_hash_groups[sha])
        entries.append(
            {
                "path": sanitize_path(path),
                "kind": "dir" if path.is_dir() else "file",
                "exists": True,
                "size_bytes": size,
                "sha256": sha,
                "duplicate_rank": duplicate_rank,
                "disposition": classify(path, duplicate_rank),
            }
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Build scoped external intake matrix")
    parser.add_argument("--json-out", default=str(REPORT_JSON))
    parser.add_argument("--md-out", default=str(REPORT_MD))
    args = parser.parse_args()

    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)

    entries = collect_entries()
    payload = {
        "scope": "Desktop/Documents/Downloads/Applications project-related artifacts only",
        "repo_root": sanitize_path(ROOT),
        "entry_count": len(entries),
        "entries": entries,
    }
    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Scoped External Intake Matrix",
        "",
        "| Path | Kind | Disposition | Size | SHA256 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        lines.append(
            f"| `{entry['path']}` | `{entry['kind']}` | `{entry['disposition']}` | `{entry.get('size_bytes') or ''}` | `{entry.get('sha256') or ''}` |"
        )
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json": sanitize_path(json_out),
                "md": sanitize_path(md_out),
                "entries": len(entries),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
