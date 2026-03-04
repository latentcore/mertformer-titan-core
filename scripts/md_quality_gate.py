#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Small TR/EN typo dictionary for deterministic offline checks.
TYPO_MAP = {
    "yapiliyacak": "yapılacak",
    "calisiyor": "çalışıyor",
    "dont": "don't",
    "tehcnical": "technical",
    "definately": "definitely",
}


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


def scan_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    issues = []

    last_level = 0
    for i, line in enumerate(lines, start=1):
        if line.rstrip(" ") != line:
            issues.append({"line": i, "type": "trailing_spaces", "severity": "error", "detail": "line has trailing spaces"})

        hm = HEADING_RE.match(line)
        if hm:
            level = len(hm.group(1))
            if last_level and level > last_level + 1:
                issues.append(
                    {
                        "line": i,
                        "type": "heading_jump",
                        "severity": "warning",
                        "detail": f"jump from h{last_level} to h{level}",
                    }
                )
            last_level = level

        low = line.lower()
        for bad, good in TYPO_MAP.items():
            if bad in low:
                issues.append({"line": i, "type": "typo", "severity": "error", "detail": f"{bad} -> {good}"})

    return {"path": str(path), "issue_count": len(issues), "issues": issues}


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline markdown quality gate")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="reports/md_lint_report.json")
    ap.add_argument("--scope", choices=["release_core", "all"], default="release_core")
    ap.add_argument("--fail-on-issues", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if args.scope == "all":
        md_files = _all_md_files(root)
    else:
        md_files = _release_core_files(root)

    per_file = [scan_file(p) for p in md_files]

    severity_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    for f in per_file:
        for issue in f["issues"]:
            severity_counter[issue["severity"]] += 1
            type_counter[issue["type"]] += 1

    error_count = severity_counter.get("error", 0)
    warning_count = severity_counter.get("warning", 0)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "scope": args.scope,
        "file_count": len(md_files),
        "total_issues": int(sum(severity_counter.values())),
        "error_count": int(error_count),
        "warning_count": int(warning_count),
        "issue_type_counts": dict(type_counter),
        "files": per_file,
        "ok": error_count == 0,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"md_quality: scope={args.scope} files={len(md_files)} "
        f"errors={error_count} warnings={warning_count}"
    )
    if args.fail_on_issues and error_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
