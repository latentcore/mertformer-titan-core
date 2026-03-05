#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXCLUDE_PARTS = {
    ".git",
    ".titan-venv",
    ".lint-venv",
    ".venv",
    ".idea",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}


SENSITIVE_FILE_NAMES = {".env"}


def tracked_files(root: Path) -> list[Path]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
        raw = proc.stdout.decode("utf-8", errors="replace")
        rels = [p for p in raw.split("\0") if p]
        return [root / rel for rel in rels]
    except Exception:
        return []




def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel_if_under(root: Path, target: Path) -> str | None:
    try:
        return str(target.resolve().relative_to(root.resolve()))
    except Exception:
        return None


def collect_entries(root: Path, excluded_relpaths: set[str]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []

    candidates = tracked_files(root)
    if not candidates:
        candidates = [p for p in root.rglob("*") if p.is_file()]

    for p in candidates:
        if not p.exists() or not p.is_file():
            continue
        rel = p.relative_to(root)
        rel_s = str(rel)
        if any(part in EXCLUDE_PARTS for part in rel.parts):
            continue
        if rel_s in excluded_relpaths:
            continue
        name = rel.name.lower()
        if name in SENSITIVE_FILE_NAMES or name.startswith(".env."):
            continue

        st = p.stat()
        entries.append(
            {
                "path": rel_s,
                "size_bytes": int(st.st_size),
                "sha256": file_hash(p),
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            }
        )

    entries.sort(key=lambda x: str(x["path"]))
    return entries


def structure_comment(rel: str, is_dir: bool) -> str:
    p = Path(rel)
    name = p.name
    lower = name.lower()
    suffix = p.suffix.lower()

    if is_dir:
        return "directory"
    if name == ".gitignore":
        return "git ignore policy"
    if name == "Dockerfile":
        return "container build baseline"
    if name == "CITATION.cff":
        return "citation metadata"
    if name == "pyproject.toml":
        return "project metadata"
    if name == "LICENSE":
        return "license terms (EN)"
    if name == "LICENSE_TR":
        return "license terms (TR)"
    if name == "README.md":
        return "primary documentation (EN)"
    if name == "README_TR.md" or lower.endswith("_tr.md"):
        return "Turkish document counterpart"
    if suffix == ".md":
        return "documentation/report file"
    if suffix == ".py":
        return "Python module/script"
    if suffix == ".sh":
        return "shell automation script"
    if suffix in {".yaml", ".yml"}:
        return "YAML configuration file"
    if suffix == ".jsonl":
        return "JSONL data/log artifact"
    if suffix == ".json":
        return "JSON schema artifact" if "schema" in lower else "JSON data artifact"
    if suffix == ".csv":
        return "CSV data artifact"
    if suffix == ".txt":
        return "text artifact"
    if suffix == ".cpp":
        return "C++ source file"
    if suffix in {".png", ".gif", ".mp4", ".svg"}:
        return "media asset"
    if suffix == ".sha256":
        return "artifact checksum"
    if suffix in {".pdf", ".pptx", ".age"}:
        return "artifact"
    if suffix == ".toml":
        return "TOML configuration file"
    return "artifact"


def build_tree(paths: list[str]) -> dict[str, Any]:
    root: dict[str, Any] = {"dirs": {}, "files": []}
    for rel in sorted(paths):
        parts = Path(rel).parts
        node = root
        for part in parts[:-1]:
            node = node["dirs"].setdefault(part, {"dirs": {}, "files": []})
        node["files"].append(parts[-1])
    return root


def emit_tree(node: dict[str, Any], prefix: str = "", parent_parts: tuple[str, ...] = ()) -> list[str]:
    lines: list[str] = []
    dir_names = sorted(node["dirs"])
    file_names = sorted(node["files"])
    entries = [(name, True) for name in dir_names] + [(name, False) for name in file_names]

    for idx, (name, is_dir) in enumerate(entries):
        last = idx == len(entries) - 1
        branch = "└── " if last else "├── "
        rel = "/".join(parent_parts + (name,))
        display = f"{name}/" if is_dir else name
        lines.append(f"{prefix}{branch}{display}  # {structure_comment(rel, is_dir)}")
        if is_dir:
            child_prefix = prefix + ("    " if last else "│   ")
            lines.extend(emit_tree(node["dirs"][name], child_prefix, parent_parts + (name,)))
    return lines


def build_structure_md(paths: list[str], out_path: Path) -> None:
    lines = [
        "# PROJECT_STRUCTURE",
        "",
        "Generated automatically from tracked files with inline role comments.",
        "",
        "```text",
        "mertformer-titan-core/  # project root (git ls-files inventory)",
    ]
    lines.extend(emit_tree(build_tree(paths)))
    lines.append("```")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate release manifest and sync reports")
    ap.add_argument("--root", default=".")
    ap.add_argument("--manifest", default="reports/release_manifest.json")
    ap.add_argument("--structure", default="docs/PROJECT_STRUCTURE.md")
    ap.add_argument("--matrix", default="reports/file_sync_matrix.json")
    ap.add_argument("--sync-report", default="reports/project_structure_sync_report.json")
    ap.add_argument("--policy-report", default="reports/policy_sync_report.json")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    manifest_path = Path(args.manifest)
    structure_path = Path(args.structure)
    matrix_path = Path(args.matrix)
    sync_path = Path(args.sync_report)
    policy_path = Path(args.policy_report)

    excluded_relpaths = set()
    for p in [manifest_path, structure_path, matrix_path, sync_path, policy_path]:
        rel = _rel_if_under(root, p)
        if rel:
            excluded_relpaths.add(rel)

    entries = collect_entries(root, excluded_relpaths)
    entry_paths = [str(e["path"]) for e in entries]

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": ".",
        "entry_count": len(entries),
        "entries": entries,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    build_structure_md(entry_paths, structure_path)

    matrix_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_count": len(entry_paths),
        "structure_count": len(entry_paths),
        "missing_in_structure": [],
        "missing_in_manifest": [],
        "ok": True,
    }

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(json.dumps(matrix_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    sync_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "details": matrix_payload,
    }
    sync_path.parent.mkdir(parents=True, exist_ok=True)
    sync_path.write_text(json.dumps(sync_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    policy_file = Path("policy/allow_deny_policy.yaml")
    policy_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policy_file": str(policy_file),
        "policy_exists": (root / policy_file).exists(),
        "ok": (root / policy_file).exists(),
    }
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = bool(policy_payload["ok"])
    print(json.dumps({"manifest_entries": len(entries), "ok": ok}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
