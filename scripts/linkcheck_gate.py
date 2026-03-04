#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def is_external(link: str) -> bool:
    return link.startswith("http://") or link.startswith("https://") or link.startswith("mailto:")


def _release_core_files(root: Path) -> list[Path]:
    targets = [
        root / "README.md",
        root / "README_TR.md",
        root / "reports" / "demo_script.md",
        root / "reports" / "final_repo_audit.md",
        root / "reports" / "folder_structure_policy.md",
        root / "reports" / "docs_dedup_canonical_list.md",
    ]
    targets.extend(sorted((root / "docs").rglob("*.md")))
    return sorted({p for p in targets if p.exists()})


def _all_md_files(root: Path) -> list[Path]:
    ignore_parts = {".git", ".titan-venv", ".lint-venv", "__pycache__"}
    files = []
    for p in root.rglob("*.md"):
        rel = p.relative_to(root)
        if any(part in ignore_parts for part in rel.parts):
            continue
        files.append(p)
    return sorted(files)


def scan(path: Path) -> list[dict]:
    issues = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for i, line in enumerate(text.splitlines(), start=1):
        for m in LINK_RE.finditer(line):
            target = m.group(1).strip()
            if target.startswith("#") or is_external(target):
                continue
            rel = target.split("#", 1)[0]
            p = (path.parent / rel).resolve()
            if not p.exists():
                issues.append({"line": i, "missing": target})
    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline markdown link checker")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="reports/linkcheck_report.json")
    ap.add_argument("--scope", choices=["release_core", "all"], default="release_core")
    ap.add_argument("--fail-on-missing", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if args.scope == "all":
        md_files = _all_md_files(root)
    else:
        md_files = _release_core_files(root)

    findings = []
    missing_total = 0
    for f in md_files:
        issues = scan(f)
        missing_total += len(issues)
        if issues:
            findings.append({"path": str(f), "missing_links": issues})

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "scope": args.scope,
        "md_file_count": len(md_files),
        "missing_link_count": missing_total,
        "findings": findings,
        "ok": missing_total == 0,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"linkcheck: scope={args.scope} files={len(md_files)} missing={missing_total}")
    if args.fail_on_missing and missing_total > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
