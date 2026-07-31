#!/usr/bin/env python3
"""
bandit (Python SAST) — local pre-push mirror of the existing CI gate.

[2026-07-12] BACKLOG I.7 #89: "bandit/semgrep guvenlik taramasi CI'da" was
ALREADY partly done: .github/workflows/ci.yml already runs
`bandit -r . -lll -iii -q -x ./.titan-venv,./.lint-venv,./build,./tests`
(HIGH severity + HIGH confidence). Found by reading BACKLOG.md's own history
(I.7.4, 2026-07-09 pass) -- should have been checked before building this,
not after. What was actually missing: that gate only runs in GitHub Actions,
AFTER a push -- verify_all.sh (the local pre-push gate) never called bandit
at all, so there was no local, pre-push signal matching what CI would say.
This script closes that gap, deliberately mirroring the EXISTING CI policy
(HIGH severity AND HIGH confidence, via --fail-on/--min-confidence both
defaulting to "high") rather than inventing a different one, and using the
same `./`-relative exclude form the CI step already uses (confirmed, after
two broken attempts of my own, to be the only form bandit's -x actually
matches against its own `-r .` walk).

A full repo scan currently returns ~2587 findings (2476 LOW, 111 MEDIUM, 0
HIGH) -- overwhelmingly expected patterns for this codebase (75x B615
huggingface_unsafe_download: this repo deliberately loads HF models/
tokenizers; 28x B614 pytorch_load: already-justified `weights_only=False` on
trusted own-checkpoint loads, see eval/gsm8k.py's own comment; a sandboxed
eval() in orchestrator/tool_executor.py already carries a `# noqa: S307`
acknowledging the tradeoff; an exec() in
scripts/mertformer_5080_final_onefile.py is the onefile bundler re-executing
the repo's OWN embedded source, not arbitrary code). Auditing and resolving
2587 historical findings is a separate, much larger initiative than "add a
local mirror of the existing CI gate" -- out of scope here.

Usage:
    python scripts/check_bandit_security_scan.py
    python scripts/check_bandit_security_scan.py --fail-on medium --min-confidence medium
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "reports" / "bandit_security_scan.json"
# Matches .github/workflows/ci.yml's own `-x ./.titan-venv,./.lint-venv,./build,./tests`
# exactly (both the set of excluded dirs and the `./`-relative form -- two OTHER forms
# were tried here and silently excluded nothing, found by inspecting this script's own
# live runs: a bare ".titan-venv" with no prefix, and an absolute path rooted at ROOT;
# both let bandit scan third-party site-packages, e.g. aiohttp/dill/fsspec/paramiko/pip's
# vendored deps, and surface THEIR pre-existing HIGH findings as if they were this repo's
# own code). Only the `./`-relative form bandit's own `-r .` walk actually matches.
_EXCLUDE_NAMES = (".titan-venv", ".lint-venv", "build", "tests")
EXCLUDE_DIRS = ",".join(f"./{name}" for name in _EXCLUDE_NAMES)

SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def _bandit_python() -> str:
    """bandit is installed in .titan-venv, NOT necessarily on sys.executable --
    a caller invoked under system/pytest Python has no bandit module, and
    `python -m bandit` then fails near-instantly with ModuleNotFoundError.
    That failure was previously swallowed silently (subprocess check=False,
    then unconditionally re-reading whatever REPORT_PATH already held on
    disk from an earlier, differently-configured run) -- found by adding the
    real end-to-end test below, which caught a stale report being re-read as
    if it were fresh. Prefer .titan-venv's own interpreter explicitly; fall
    back to sys.executable only if that venv genuinely doesn't exist."""
    venv_python = ROOT / ".titan-venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def run_bandit() -> dict:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists():
        REPORT_PATH.unlink()  # never trust a leftover file from a prior run as "fresh"

    proc = subprocess.run(
        [
            _bandit_python(), "-m", "bandit", "-r", ".",
            "-x", EXCLUDE_DIRS,
            "-f", "json",
            "-o", str(REPORT_PATH),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        check=False,  # bandit exits non-zero when it finds ANY issue -- we read the JSON ourselves
    )
    if not REPORT_PATH.exists():
        raise RuntimeError(
            f"bandit did not produce a report at {REPORT_PATH} (exit={proc.returncode}); "
            f"stderr tail: {proc.stderr[-1000:]}"
        )
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="bandit SAST scan, HIGH-severity+HIGH-confidence hard-fail policy.")
    parser.add_argument("--fail-on", choices=["high", "medium", "low"], default="high")
    parser.add_argument(
        "--min-confidence",
        choices=["high", "medium", "low"],
        default="high",
        help="Only findings at or above BOTH --fail-on severity AND this confidence block. "
        "Matches .github/workflows/ci.yml's existing `bandit -lll -iii` policy (HIGH severity "
        "AND HIGH confidence) -- a local run should not be stricter than the CI gate it mirrors.",
    )
    args = parser.parse_args(argv)

    report = run_bandit()
    results = report.get("results", [])

    by_severity: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for r in results:
        sev = r.get("issue_severity", "LOW")
        by_severity[sev] = by_severity.get(sev, 0) + 1

    sev_threshold = SEVERITY_ORDER[args.fail_on.upper()]
    conf_threshold = SEVERITY_ORDER[args.min_confidence.upper()]
    blocking = [
        r for r in results
        if SEVERITY_ORDER.get(r.get("issue_severity", "LOW"), 0) >= sev_threshold
        and SEVERITY_ORDER.get(r.get("issue_confidence", "LOW"), 0) >= conf_threshold
    ]

    print(f"bandit scan: {len(results)} total findings -- {by_severity}")
    print(f"Full report: {REPORT_PATH.relative_to(ROOT)}")

    policy = f"severity>={args.fail_on}, confidence>={args.min_confidence}"
    if blocking:
        print(f"\nFAIL: {len(blocking)} finding(s) at or above {policy}:")
        for r in blocking[:20]:
            print(
                f"  - {r['filename']}:{r['line_number']} "
                f"[{r['issue_severity']}/{r['issue_confidence']}] {r['test_id']} {r['issue_text']}"
            )
        if len(blocking) > 20:
            print(f"  ... and {len(blocking) - 20} more (see {REPORT_PATH.relative_to(ROOT)})")
        return 1

    print(f"\nOK: no findings at or above {policy} (currently: HIGH={by_severity['HIGH']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
