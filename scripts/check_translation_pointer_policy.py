#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CANONICAL = ROOT / "reports/codex_deep_audit_TR.md"
POINTERS = [
    ROOT / "reports/codex_deep_audit_EN_TR.md",
    ROOT / "reports/codex_deep_audit_DE_TR.md",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce TR audit pointer-file policy.")
    parser.add_argument("--max-lines", type=int, default=80)
    args = parser.parse_args()

    errors: list[str] = []
    if not CANONICAL.exists():
        errors.append(f"missing canonical file: {CANONICAL}")
    else:
        canonical_sha = sha(CANONICAL)

    for p in POINTERS:
        if not p.exists():
            errors.append(f"missing pointer file: {p}")
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        if len(lines) > args.max_lines:
            errors.append(f"pointer too long ({len(lines)} lines): {p}")

        lower = text.lower()
        if "pointer" not in lower and "yonlendirme" not in lower and "yönlendirme" not in lower:
            errors.append(f"pointer keyword missing: {p}")
        if "reports/codex_deep_audit_tr.md" not in lower:
            errors.append(f"canonical path reference missing: {p}")

        if CANONICAL.exists() and sha(p) == canonical_sha:
            errors.append(f"pointer must not be byte-identical to canonical: {p}")

    if errors:
        print("FAIL: translation pointer policy violations")
        for e in errors:
            print(f" - {e}")
        return 1

    print("OK: translation pointer policy satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
