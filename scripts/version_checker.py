"""Version consistency checker.

Scans the repository for deprecated version markers (legacy labels). Exits non-zero if any are found.
"""
from __future__ import annotations

import argparse
from pathlib import Path

IGNORE_DIRS = {".git", ".titan-venv", ".lint-venv", "logs", "checkpoints", "datasets"}
TEXT_EXTS = {".md", ".py", ".yaml", ".yml", ".toml", ".txt", ".sh", ".cff"}
IGNORE_FILES = {"CITATION.cff", "version_checker.py"}

BANNED_TOKENS = [
    "v27",
    "v27.0",
    "FINAL",
    "Production Ready",
    "PRODUCTION READY",
    "Locked & Sealed",
    "LOCKED",
]


def should_scan(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name in IGNORE_FILES:
        return False
    if any(part in IGNORE_DIRS for part in path.parts):
        return False
    return path.suffix in TEXT_EXTS


def scan(root: Path) -> list[tuple[Path, int, str, str]]:
    hits: list[tuple[Path, int, str, str]] = []
    for path in root.rglob("*"):
        if not should_scan(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            for token in BANNED_TOKENS:
                if token in line:
                    hits.append((path, i, token, line.strip()))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    hits = scan(root)
    if not hits:
        print("OK: version markers are clean.")
        return 0

    print("ERROR: deprecated version markers found:")
    for path, line_no, token, line in hits:
        print(f"- {path}:{line_no} ({token}) :: {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())