"""Markdown integrity checks: utf-8, mojibake, and local link validity."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

MOJIBAKE_PATTERNS = ["Ã", "Ä", "Å", "�"]
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
IGNORE_PARTS = {".git", ".titan-venv", ".lint-venv", "node_modules"}
# Link-integrity is only meaningful for live, tracked, non-frozen docs. The following are
# scoped OUT (the gate stays strict for everything else):
#   - pitch_kit: a gitignored, regenerated commercial kit (not part of the tracked repo);
#   - archive/: frozen / superseded content (e.g. the pre-cut full README) whose old
#     relative links intentionally no longer resolve from the new location;
#   - *_before_final_simplification.md: frozen pre-simplification README snapshots that
#     intentionally preserve the OLD layout/links as a historical record.
IGNORE_PARTS_SCOPED = {"pitch_kit", "archive"}
FROZEN_SNAPSHOT_SUFFIX = "_before_final_simplification.md"


def should_skip(path: Path) -> bool:
    if path.name.endswith(FROZEN_SNAPSHOT_SUFFIX):
        return True
    return any(
        part in IGNORE_PARTS or part in IGNORE_PARTS_SCOPED or part.startswith(".titan-venv")
        for part in path.parts
    )


def check_file(path: Path, root: Path) -> list[str]:
    issues: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [f"utf8_decode_error:{exc}"]

    for bad in MOJIBAKE_PATTERNS:
        if bad in text:
            issues.append(f"mojibake_pattern:{bad}")
            break

    for match in LINK_RE.finditer(text):
        target = match.group(1).strip()
        if not target:
            continue
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        local = target.split("#", 1)[0].strip()
        if not local:
            continue
        candidates = []
        if local.startswith("/"):
            candidates.append(Path(local))
        else:
            candidates.append((path.parent / local).resolve())
            candidates.append((root / local).resolve())
        if not any(c.exists() for c in candidates):
            issues.append(f"broken_link:{target}")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings: list[tuple[Path, list[str]]] = []
    scanned = 0

    for path in root.rglob("*.md"):
        if should_skip(path):
            continue
        scanned += 1
        issues = check_file(path, root)
        if issues:
            findings.append((path, issues))

    print(f"md_integrity_check scanned={scanned} findings={len(findings)}")
    for path, issues in findings[:200]:
        for issue in issues:
            print(f"{path}:{issue}")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
