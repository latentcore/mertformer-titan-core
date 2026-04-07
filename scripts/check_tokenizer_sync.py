#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CANONICAL = ROOT / "interfaces/tokenizer_spec.json"
DEFAULT_MIRROR = ROOT / "tokenizer/tokenizer.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    root_resolved = ROOT.resolve()
    if resolved == root_resolved:
        return "<REPO_ROOT>"
    try:
        rel = resolved.relative_to(root_resolved)
        rel_text = rel.as_posix()
        return f"<REPO_ROOT>/{rel_text}" if rel_text else "<REPO_ROOT>"
    except ValueError:
        return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify tokenizer spec mirror stays byte-synced.")
    parser.add_argument("--canonical", default=str(DEFAULT_CANONICAL))
    parser.add_argument("--mirror", default=str(DEFAULT_MIRROR))
    parser.add_argument("--fix", action="store_true", help="Copy canonical file to mirror if mismatch")
    args = parser.parse_args()

    canonical = Path(args.canonical)
    mirror = Path(args.mirror)

    if not canonical.exists():
        print(f"FAIL: canonical file missing: {canonical}")
        return 2
    if not mirror.exists():
        print(f"FAIL: mirror file missing: {mirror}")
        return 2

    c_data = canonical.read_bytes()
    m_data = mirror.read_bytes()
    c_hash = sha256_bytes(c_data)
    m_hash = sha256_bytes(m_data)

    print(f"canonical={display_path(canonical)}")
    print(f"mirror={display_path(mirror)}")
    print(f"canonical_sha256={c_hash}")
    print(f"mirror_sha256={m_hash}")

    if c_data == m_data:
        print("OK: tokenizer spec files are byte-identical")
        return 0

    if args.fix:
        mirror.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(canonical, mirror)
        print("FIXED: mirror overwritten from canonical")
        return 0

    print("FAIL: tokenizer spec files diverged")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
