#!/usr/bin/env python3
"""
Repo hygiene guard — prevents two recurring closure-audit regressions.

Motivation: across many closure passes the same two classes of defect kept
re-appearing in hand-edited code and had to be re-found by manual audits:

  1. bare ``except:`` clauses (silently swallow KeyboardInterrupt/SystemExit
     and hide real errors), and
  2. stale build/version fossils (old "BUILD27"/"Build 28"/"V27.0" stamps left
     behind after the project moved to Build 30 V2).

This gate makes both impossible to reintroduce silently: it scans tracked
source files and fails (non-zero exit) if either pattern appears outside the
explicit allowlists. It is wired into scripts/verify_all.sh so every closure
re-checks it automatically. It is intentionally narrow and zero-false-positive
on the current tree — it is a regression backstop, not a style linter.

Run:
    python scripts/repo_hygiene_guard.py            # human summary + exit code
    python scripts/repo_hygiene_guard.py --json      # machine-readable summary
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Check 1: bare except --------------------------------------------------
# Matches `except:` with no exception type. `except Exception:` is allowed.
_BARE_EXCEPT = re.compile(r"^[ \t]*except[ \t]*:")

# --- Check 2: stale build/version fossils ----------------------------------
# Matches old build numbers (BUILD20..BUILD29 / "Build 2x") and old V2x.0 tags.
# The current stamp is BUILD30 / Build 30 V2, which does NOT match BUILD2[0-9].
_BUILD_FOSSIL = re.compile(r"\bBUILD ?2[0-9]\b|\bBuild ?2[0-9]\b|\bV2[0-9]\.0\b")

# Files/dirs allowed to mention old build numbers because they intentionally
# document the project's *history* (the user requires evolution traces, e.g.
# the b27->b30 and MLA->GQA transitions, to be preserved).
_FOSSIL_ALLOW_SUBSTR = (
    "CHANGELOG",
    "DECISIONS",
    "/docs/",
    "/reports/",
    "/adr/",
    "/archive/",
    "/snapshots/",
    "BACKLOG",
    "dev_report",
    "DevReport",
    "_audit",
    "closure",
    "postmortems/",
    "scripts/repo_hygiene_guard.py",  # this file (contains the patterns)
    "scripts/md_build30_sweep.py",    # documented historical-sweep helper
)

# Only scan textual source we actually author. Skip binaries/data/vendored.
_TEXT_SUFFIXES = {
    ".py", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".md", ".txt", ".json", ".jsonl", ".cpp", ".c", ".h",
}
# Directories never worth scanning even if somehow tracked.
_SKIP_DIR_PARTS = {".git", ".titan-venv", ".lint-venv", "__pycache__"}


def tracked_files() -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "ls-files"],
        check=True, capture_output=True, text=True,
    )
    return [p for p in proc.stdout.splitlines() if p]


def _is_allowlisted_for_fossil(rel: str) -> bool:
    return any(sub in rel for sub in _FOSSIL_ALLOW_SUBSTR)


def scan() -> dict:
    bare_except: list[str] = []
    build_fossils: list[str] = []

    for rel in tracked_files():
        p = Path(rel)
        if any(part in _SKIP_DIR_PARTS for part in p.parts):
            continue
        if p.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        abs_p = PROJECT_ROOT / rel
        try:
            text = abs_p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # unreadable/binary-ish: not our concern here

        is_py = p.suffix.lower() == ".py"
        allow_fossil = _is_allowlisted_for_fossil(rel)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if is_py and _BARE_EXCEPT.match(line):
                bare_except.append(f"{rel}:{lineno}: {line.strip()}")
            if not allow_fossil and _BUILD_FOSSIL.search(line):
                build_fossils.append(f"{rel}:{lineno}: {line.strip()}")

    ok = not bare_except and not build_fossils
    return {
        "ok": ok,
        "bare_except_count": len(bare_except),
        "build_fossil_count": len(build_fossils),
        "bare_except": bare_except,
        "build_fossils": build_fossils,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Repo hygiene regression guard")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    result = scan()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result["ok"]:
            print(
                "OK: repo hygiene guard; "
                f"bare_except=0, build_fossils=0 (scanned tracked source)."
            )
        else:
            print("FAIL: repo hygiene guard found regressions:", file=sys.stderr)
            for hit in result["bare_except"]:
                print(f"  [bare-except] {hit}", file=sys.stderr)
            for hit in result["build_fossils"]:
                print(f"  [build-fossil] {hit}", file=sys.stderr)
            print(
                "\nFix: use `except Exception:` (or a specific type) instead of bare "
                "`except:`; update stale build/version stamps to the current Build 30 "
                "V2 (or move historical references into an allowlisted history doc).",
                file=sys.stderr,
            )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
