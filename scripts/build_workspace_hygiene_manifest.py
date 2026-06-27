#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE_ROOT = ROOT.parent
DEFAULT_QUARANTINE_ROOT = DEFAULT_WORKSPACE_ROOT / "workspace_quarantine"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def display_path(path: Path, *, workspace_root: Path, repo_root: Path, quarantine_root: Path) -> str:
    resolved = path.resolve()
    repo_resolved = repo_root.resolve()
    workspace_resolved = workspace_root.resolve()
    quarantine_resolved = quarantine_root.resolve()
    if resolved == repo_resolved:
        return "<REPO_ROOT>"
    try:
        rel = resolved.relative_to(repo_resolved)
        rel_text = rel.as_posix()
        return f"<REPO_ROOT>/{rel_text}" if rel_text else "<REPO_ROOT>"
    except ValueError:
        pass
    if resolved == quarantine_resolved:
        return "<QUARANTINE_ROOT>"
    try:
        rel = resolved.relative_to(quarantine_resolved)
        rel_text = rel.as_posix()
        return f"<QUARANTINE_ROOT>/{rel_text}" if rel_text else "<QUARANTINE_ROOT>"
    except ValueError:
        pass
    if resolved == workspace_resolved:
        return "<WORKSPACE_ROOT>"
    try:
        rel = resolved.relative_to(workspace_resolved)
        rel_text = rel.as_posix()
        return f"<WORKSPACE_ROOT>/{rel_text}" if rel_text else "<WORKSPACE_ROOT>"
    except ValueError:
        return str(resolved)


def tracked_repo_paths(repo_root: Path) -> set[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z"],
            capture_output=True,
            check=True,
        )
    except Exception as exc:
        print(
            f"[warn] git ls-files failed for {repo_root}: {exc}; "
            "treating all paths as untracked",
            file=sys.stderr,
        )
        return set()
    raw = proc.stdout.decode("utf-8", errors="ignore")
    return {path for path in raw.split("\0") if path}


def classify_workspace_entry(
    path: Path,
    *,
    workspace_root: Path,
    repo_root: Path,
    tracked_paths: set[str],
) -> dict[str, str]:
    del workspace_root
    restore_target = str(path.resolve())
    if path.resolve() == repo_root.resolve():
        return {
            "path": str(path.resolve()),
            "classification": "project_repo",
            "reason": "Active main repository root; never quarantine or delete directly from workspace hygiene.",
            "restore_target": restore_target,
            "decision_state": "keep",
        }

    if path.name == ".idea":
        return {
            "path": str(path.resolve()),
            "classification": "workspace_metadata",
            "reason": "IDE workspace metadata outside the repo root. Keep unless a human explicitly retires the workspace.",
            "restore_target": restore_target,
            "decision_state": "keep",
        }

    if path.name == ".ruff_cache":
        return {
            "path": str(path.resolve()),
            "classification": "workspace_cache",
            "reason": "Rebuildable lint cache outside the repo root. Safe candidate for quarantine-first handling.",
            "restore_target": restore_target,
            "decision_state": "quarantine_first",
        }

    if path.name == ".DS_Store":
        return {
            "path": str(path.resolve()),
            "classification": "desktop_metadata",
            "reason": "macOS metadata file. Ignore by default unless a human wants cosmetic cleanup.",
            "restore_target": restore_target,
            "decision_state": "ignore",
        }

    if repo_root.resolve() in path.resolve().parents:
        rel = str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
        decision_state = "keep" if rel in tracked_paths else "ignore"
        reason = (
            "Tracked repo path; workspace hygiene must not move repo-owned files."
            if decision_state == "keep"
            else "Untracked repo-internal path; leave repo-local cleanup to repo-specific tooling."
        )
        return {
            "path": str(path.resolve()),
            "classification": "repo_internal",
            "reason": reason,
            "restore_target": restore_target,
            "decision_state": decision_state,
        }

    if path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"}:
        return {
            "path": str(path.resolve()),
            "classification": "archive_or_dump",
            "reason": "Top-level archive or dump outside the main repo. Review and quarantine before any permanent cleanup.",
            "restore_target": restore_target,
            "decision_state": "quarantine_first",
        }

    if path.is_dir():
        return {
            "path": str(path.resolve()),
            "classification": "external_workspace_entry",
            "reason": "Top-level directory outside the active repo root. Keep only if still actively used; otherwise quarantine first.",
            "restore_target": restore_target,
            "decision_state": "quarantine_first",
        }

    return {
        "path": str(path.resolve()),
        "classification": "external_workspace_file",
        "reason": "Top-level file outside the active repo root. Quarantine first instead of deleting in place.",
        "restore_target": restore_target,
        "decision_state": "quarantine_first",
    }


def build_manifest(
    *,
    workspace_root: Path,
    repo_root: Path,
    quarantine_root: Path,
    mode: str,
) -> dict[str, Any]:
    tracked_paths = tracked_repo_paths(repo_root)
    items: list[dict[str, str]] = []
    for path in sorted(workspace_root.iterdir(), key=lambda item: item.name.lower()):
        items.append(
            classify_workspace_entry(
                path,
                workspace_root=workspace_root,
                repo_root=repo_root,
                tracked_paths=tracked_paths,
            )
        )

    return {
        "schema": "workspace_hygiene_manifest_v1",
        "generated_utc": utc_now(),
        "mode": mode,
        "workspace_root": str(workspace_root.resolve()),
        "repo_root": str(repo_root.resolve()),
        "quarantine_root": str(quarantine_root.resolve()),
        "items": items,
    }


def sanitize_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    workspace_root = Path(payload["workspace_root"])
    repo_root = Path(payload["repo_root"])
    quarantine_root = Path(payload["quarantine_root"])
    sanitized_items: list[dict[str, str]] = []
    for item in payload["items"]:
        sanitized_items.append(
            {
                **item,
                "path": display_path(
                    Path(item["path"]),
                    workspace_root=workspace_root,
                    repo_root=repo_root,
                    quarantine_root=quarantine_root,
                ),
                "restore_target": display_path(
                    Path(item["restore_target"]),
                    workspace_root=workspace_root,
                    repo_root=repo_root,
                    quarantine_root=quarantine_root,
                ),
            }
        )
    sanitized_payload = {
        **payload,
        "workspace_root": "<WORKSPACE_ROOT>",
        "repo_root": "<REPO_ROOT>",
        "quarantine_root": "<QUARANTINE_ROOT>",
        "items": sanitized_items,
    }
    if "moves" in payload:
        sanitized_payload["moves"] = [
            {
                "source": display_path(
                    Path(entry["source"]),
                    workspace_root=workspace_root,
                    repo_root=repo_root,
                    quarantine_root=quarantine_root,
                ),
                "quarantine_path": display_path(
                    Path(entry["quarantine_path"]),
                    workspace_root=workspace_root,
                    repo_root=repo_root,
                    quarantine_root=quarantine_root,
                ),
                "restore_target": display_path(
                    Path(entry["restore_target"]),
                    workspace_root=workspace_root,
                    repo_root=repo_root,
                    quarantine_root=quarantine_root,
                ),
            }
            for entry in payload["moves"]
        ]
    return sanitized_payload


def apply_quarantine(manifest: dict[str, Any], quarantine_root: Path) -> list[dict[str, str]]:
    quarantine_root.mkdir(parents=True, exist_ok=True)
    moved: list[dict[str, str]] = []
    for item in manifest["items"]:
        if item["decision_state"] != "quarantine_first":
            continue
        source = Path(item["path"])
        if not source.exists():
            continue
        target = quarantine_root / source.name
        suffix = 1
        while target.exists():
            target = quarantine_root / f"{source.stem}_{suffix}{source.suffix}"
            suffix += 1
        shutil.move(str(source), str(target))
        moved.append(
            {
                "source": str(source),
                "quarantine_path": str(target),
                "restore_target": item["restore_target"],
            }
        )
    return moved


def build_markdown(payload: dict[str, Any], moved: list[dict[str, str]]) -> str:
    lines = [
        "# Workspace Hygiene Manifest",
        "",
        "Quarantine-first workspace hygiene report. No repo-tracked path should be moved or deleted by this lane.",
        "",
        f"- mode: `{payload['mode']}`",
        f"- workspace_root: `{payload['workspace_root']}`",
        f"- repo_root: `{payload['repo_root']}`",
        f"- quarantine_root: `{payload['quarantine_root']}`",
        "",
        "## Decisions",
        "",
        "| Path | Classification | Decision | Reason | Restore Target |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in payload["items"]:
        lines.append(
            "| `{path}` | `{classification}` | `{decision}` | {reason} | `{restore}` |".format(
                path=item["path"],
                classification=item["classification"],
                decision=item["decision_state"],
                reason=item["reason"],
                restore=item["restore_target"],
            )
        )

    lines.extend(["", "## Quarantine Moves"])
    if moved:
        for entry in moved:
            lines.append(
                "- `{source}` -> `{target}` (restore to `{restore}`)".format(
                    source=entry["source"],
                    target=entry["quarantine_path"],
                    restore=entry["restore_target"],
                )
            )
    else:
        lines.append("- none")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a quarantine-first workspace hygiene manifest.")
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--quarantine-root", default=str(DEFAULT_QUARANTINE_ROOT))
    parser.add_argument("--out-json", default="reports/workspace_hygiene_manifest.json")
    parser.add_argument("--out-md", default="reports/workspace_hygiene_manifest.md")
    parser.add_argument("--apply-quarantine", action="store_true")
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    quarantine_root = Path(args.quarantine_root).expanduser().resolve()
    mode = "apply_quarantine" if args.apply_quarantine else "audit_only"

    payload = build_manifest(
        workspace_root=workspace_root,
        repo_root=repo_root,
        quarantine_root=quarantine_root,
        mode=mode,
    )
    moved = apply_quarantine(payload, quarantine_root) if args.apply_quarantine else []
    if moved:
        payload["moves"] = moved
    serialized_payload = sanitize_manifest(payload)

    out_json = ROOT / args.out_json
    out_md = ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(serialized_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(build_markdown(serialized_payload, serialized_payload.get("moves", [])).rstrip() + "\n", encoding="utf-8")

    print("OK: workspace hygiene manifest refreshed")
    print(f" - {out_md.relative_to(ROOT)}")
    print(f" - {out_json.relative_to(ROOT)}")
    if moved:
        print(f" - quarantine_moves={len(moved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
