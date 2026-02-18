"""Bulk Build30 sync for markdown files."""
from __future__ import annotations

import argparse
from pathlib import Path

REPLACEMENTS = [
    ("v1.0-TITAN-BUILD30", "v1.0-TITAN-BUILD30"),
    ("1.0-BUILD30", "1.0-BUILD30"),
    ("v1.0_B30", "v1.0_B30"),
    ("Build 30", "Build 30"),
    ("BUILD30", "BUILD30"),
    ("_B30_", "_B30_"),
    ("_B30.", "_B30."),
    ("_B30-", "_B30-"),
    (
        "MertFormer_Titan_OnyxStorm_v1.0_B30_Hamdi_Package_Release.zip",
        "MertFormer_Titan_OnyxStorm_v1.0_B30_Hamdi_Package_Release.passphrase.age",
    ),
    (
        "MertFormer_Titan_OnyxStorm_v1.0_B30_Hamdi_Package_Release.passphrase.age",
        "MertFormer_Titan_OnyxStorm_v1.0_B30_Hamdi_Package_Release.passphrase.age",
    ),
    (
        "MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age",
        "MertFormer_Titan_OnyxStorm_v1.0_B30_Locked.secure.age",
    ),
    (
        "MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip",
        "MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip",
    ),
]

IGNORE_PARTS = {".git", ".titan-venv", ".lint-venv", "node_modules"}


def should_skip(path: Path) -> bool:
    return any(part in IGNORE_PARTS or part.startswith(".titan-venv") for part in path.parts)


def sweep(root: Path, apply: bool) -> tuple[int, int]:
    changed = 0
    scanned = 0
    for path in root.rglob("*.md"):
        if should_skip(path):
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        updated = text
        for old, new in REPLACEMENTS:
            updated = updated.replace(old, new)
        if updated != text:
            changed += 1
            if apply:
                path.write_text(updated, encoding="utf-8")
    return scanned, changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--check", action="store_true", help="Check only, do not write")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    scanned, changed = sweep(root, apply=not args.check)
    mode = "check" if args.check else "apply"
    print(f"md_build30_sweep mode={mode} scanned={scanned} changed={changed}")
    if args.check and changed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
