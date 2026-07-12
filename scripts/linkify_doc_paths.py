#!/usr/bin/env python3
"""
Linkify bare repo-relative path mentions in markdown docs into clickable
`[path](path)` links.

Read-link VALIDITY (does an existing `[text](path)` actually resolve on
disk?) is already covered by scripts/md_integrity_check.py, wired into
verify_all.sh -- this tool does not duplicate that. This is the other half:
turning a bare mention like `scripts/foo.py` (plain prose text, not already a
link) into a clickable `[scripts/foo.py](scripts/foo.py)`, so a reader
browsing the doc on GitHub can jump straight to the file.

Idempotent (safe to re-run): skips a path token already inside a markdown
link `(...)`  or an inline code span, and skips tokens that don't resolve to
a real tracked file. Manually-triggered maintenance tool -- NOT wired into
CI/verify_all.sh (mass-editing docs on every CI run would be its own risk;
a human reviews the diff via --apply).

Default is dry-run (prints what WOULD change). Pass --apply to write.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent

# Repo-relative path tokens: one or more dir segments + a recognized extension.
# Deliberately conservative: word boundary on both sides, no leading "./" or "/".
PATH_TOKEN_RE = re.compile(
    r"(?<![\w/\[\(`])"
    r"([A-Za-z0-9_][A-Za-z0-9_\-./]*\.(?:py|md|yaml|yml|json|sh|txt))"
    r"(?![\w/\)\]`])"
)

CODE_SPAN_RE = re.compile(r"`[^`]*`")
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
EXISTING_LINK_TARGET_RE = re.compile(r"\]\(([^)]+)\)")


def _tracked_files() -> set:
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, encoding="utf-8")
    return {line.strip() for line in out.splitlines() if line.strip()}


def _protected_spans(text: str) -> List[Tuple[int, int]]:
    """Byte ranges to leave untouched: fenced code blocks, inline code spans,
    and existing markdown link targets/labels."""
    spans = []
    for m in CODE_FENCE_RE.finditer(text):
        spans.append(m.span())
    for m in CODE_SPAN_RE.finditer(text):
        spans.append(m.span())
    for m in re.finditer(r"\[[^\]]*\]\([^)]*\)", text):
        spans.append(m.span())
    return spans


def _in_protected(pos: int, spans: List[Tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def linkify_text(text: str, tracked: set) -> Tuple[str, int]:
    protected = _protected_spans(text)
    out = []
    last = 0
    count = 0
    for m in PATH_TOKEN_RE.finditer(text):
        token = m.group(1)
        start, end = m.span(1)
        if _in_protected(start, protected):
            continue
        if token not in tracked:
            continue
        out.append(text[last:start])
        out.append(f"[{token}]({token})")
        last = end
        count += 1
    out.append(text[last:])
    return "".join(out), count


def main() -> int:
    parser = argparse.ArgumentParser(description="Linkify bare repo-relative path mentions in markdown docs.")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run report only).")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    tracked = _tracked_files()
    total_links = 0
    changed_files = []

    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        if any(part in (".git", ".titan-venv", ".lint-venv", "node_modules", "archive") for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        new_text, count = linkify_text(text, tracked)
        if count > 0:
            total_links += count
            changed_files.append((rel, count))
            if args.apply:
                path.write_text(new_text, encoding="utf-8")

    verb = "Linkified" if args.apply else "Would linkify"
    print(f"{verb} {total_links} bare path mention(s) across {len(changed_files)} file(s):")
    for rel, count in changed_files:
        print(f"  {rel}: {count}")
    if not args.apply and changed_files:
        print("\nDry-run only. Re-run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
