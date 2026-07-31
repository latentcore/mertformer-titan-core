"""
scripts/pre45k_gate.py -- chains the existing zero-GPU-spend pre-flight checks with a
new short, real 2-GPU DDP smoke test, so a broken launch (including a B8-class silent
DDP no-op) fails cheaply before real training budget is spent.

Chain (see BACKLOG.md, B8, "PROPOSED (2026-07-25)" for the original design note; "Gate 1"
and "Gate 2" there are informal shorthand for the two mechanisms below, not literal script
names):
  1. Offline preflight -- `scripts/titan_preflight.py`'s default profile (disk/data-stage/
     architecture/mini-forward pipeline chain). Already built, already tested, already the
     canonical offline-safe entrypoint (see README.md's verified matrix).
  2. Dry-run preview -- `zero_touch_start.sh --dry-run` (resolves the plan + real train
     command, does not launch training).
  3. DDP smoke -- `scripts/ddp_smoke.py::run_ddp_smoke_test()` (new). Skips cleanly on
     anything other than exactly 2 GPUs; on 2 GPUs, launches a short real
     `accelerate launch --num_processes 2` job and only confirms DDP if genuine dual-GPU
     activity was observed while it ran.

Claim boundary: step 3 can only be exercised for real on a genuine 2-GPU machine. On a
single-GPU or CPU machine (like a laptop) it is a correct, non-blocking skip -- not a
pass for DDP correctness, and this script never claims otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# [2026-07-29] sys.path bootstrap must run BEFORE the `scripts.*` import below.
# The canonical entrypoint (scripts/pre45k_gate.sh) invokes this as
# `python -m scripts.pre45k_gate`, and tests import it as `from scripts import
# pre45k_gate` -- both put the repo root on sys.path for us. But a plain
# `python scripts/pre45k_gate.py` sets sys.path[0] to scripts/ instead, so
# `import scripts.ddp_smoke` raised ModuleNotFoundError. This is the pre-spend
# gate: the one script most likely to be run by hand, directly, the first time.
# Every other script in this directory already does this bootstrap.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ddp_smoke import run_ddp_smoke_test  # noqa: E402  (needs the path above)

DEFAULT_REPORT_JSON = ROOT / "reports" / "pre45k_gate_report.json"
DEFAULT_REPORT_MD = ROOT / "reports" / "pre45k_gate_report.md"


def sanitize_text(text: str) -> str:
    """Same convention as scripts/start_gate.py: redact the absolute repo root and any
    home-relative Desktop/Downloads/Documents path before anything is written to disk."""
    out = text.replace(str(ROOT), "<REPO_ROOT>")
    out = re.sub(r"/Users/[^/]+/(?:Desktop|Downloads|Documents)/[^\s\"']+", "<HOME_PATH>", out)
    return out


def _run_subprocess_step(cmd: list, *, cwd: Path, env: dict = None) -> dict:
    # [2026-07-31] Explicit encoding/errors, not the `text=True` default of
    # locale.getpreferredencoding(). On Windows that default is the system codepage
    # (e.g. cp1254), not UTF-8 -- but child scripts here (titan_preflight.py etc.) now
    # reconfigure their own stdout/stderr to UTF-8, so a parent still decoding as cp1254
    # crashes trying to decode valid UTF-8 bytes it wasn't expecting (confirmed: a
    # background reader thread died mid-read, leaving p.stderr as None). Pin both sides
    # to UTF-8 explicitly so the encoding contract is consistent regardless of the
    # invoking process's locale.
    p = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
    )
    return {
        "cmd": sanitize_text(" ".join(cmd)),
        "return_code": p.returncode,
        "stdout_tail": sanitize_text(p.stdout[-4000:]),
        "stderr_tail": sanitize_text(p.stderr[-4000:]),
        "ok": p.returncode == 0,
    }


def run_offline_preflight(python_bin: str) -> dict:
    """The existing, already-tested, zero-GPU-spend offline preflight chain."""
    env = dict(os.environ)
    env.setdefault("TITAN_OFFLINE", "1")
    return _run_subprocess_step([python_bin, "scripts/titan_preflight.py"], cwd=ROOT, env=env)


def run_dry_run_preview() -> dict:
    """The existing plan + resolved-command preview. Zero GPU spend, no training launched."""
    return _run_subprocess_step(["bash", "zero_touch_start.sh", "--dry-run"], cwd=ROOT)


def combine_verdict(*, offline_preflight: dict, dry_run_preview: dict, ddp: dict, strict_ddp: bool) -> tuple:
    """Pure decision function (no I/O) so the verdict logic is directly unit-testable."""
    structural_ok = bool(offline_preflight["ok"]) and bool(dry_run_preview["ok"])
    ddp_blocking = strict_ddp and (not ddp["skipped"]) and (not ddp["ok"])
    overall_ok = structural_ok and not ddp_blocking

    if not structural_ok:
        verdict = "BLOCKED"
    elif ddp_blocking:
        verdict = "BLOCKED_DDP"
    elif ddp["skipped"]:
        verdict = "PASS_DDP_NOT_APPLICABLE"
    elif ddp["ok"]:
        verdict = "PASS_DDP_CONFIRMED"
    else:
        verdict = "PASS_DDP_UNCONFIRMED"

    return overall_ok, verdict


def build_report(
    *,
    python_bin: str,
    strict_ddp: bool,
    offline_preflight: dict = None,
    dry_run_preview: dict = None,
    ddp: dict = None,
    ddp_kwargs: dict = None,
) -> dict:
    """Builds the full report dict. Each of offline_preflight/dry_run_preview/ddp can be
    injected directly (used by tests to avoid real subprocess/GPU/network calls); when
    omitted, the real step is executed."""
    if offline_preflight is None:
        offline_preflight = run_offline_preflight(python_bin)
    if dry_run_preview is None:
        dry_run_preview = run_dry_run_preview()
    if ddp is None:
        ddp = run_ddp_smoke_test(**(ddp_kwargs or {}))

    overall_ok, verdict = combine_verdict(
        offline_preflight=offline_preflight,
        dry_run_preview=dry_run_preview,
        ddp=ddp,
        strict_ddp=strict_ddp,
    )

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ok": overall_ok,
        "verdict": verdict,
        "strict_ddp": strict_ddp,
        "steps": {
            "offline_preflight": offline_preflight,
            "dry_run_preview": dry_run_preview,
            "ddp_smoke": ddp,
        },
    }


def build_report_md(report: dict) -> str:
    steps = report["steps"]
    ddp = steps["ddp_smoke"]
    lines = [
        "# Pre-45K Gate Report",
        "",
        f"Generated: {report['generated_utc']}",
        f"Verdict: **{report['verdict']}**",
        f"strict_ddp: {report['strict_ddp']}",
        "",
        "| Step | OK |",
        "|---|---|",
        f"| Offline preflight (`titan_preflight.py`) | {steps['offline_preflight']['ok']} |",
        f"| Dry-run preview (`zero_touch_start.sh --dry-run`) | {steps['dry_run_preview']['ok']} |",
        f"| DDP smoke (2-GPU) | status={ddp['status']} ok={ddp['ok']} skipped={ddp['skipped']} |",
        "",
        "Claim boundary: the DDP smoke step only confirms genuine dual-GPU activity when "
        "exactly 2 GPUs are present and `accelerate`/CUDA are available. On a single-GPU "
        "or CPU machine it is a clean, non-blocking skip (`PASS_DDP_NOT_APPLICABLE`) -- "
        "not a pass for DDP correctness itself.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-45K gate: offline preflight + dry-run preview + a short real 2-GPU DDP "
            "smoke test, all before real training spend."
        )
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--strict-ddp",
        action="store_true",
        help="Fail (exit 1) if 2+ GPUs are present but the DDP smoke test doesn't confirm genuine dual-GPU activity.",
    )
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    args = parser.parse_args()

    report = build_report(python_bin=args.python, strict_ddp=args.strict_ddp)

    out_json = Path(args.report_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_md = Path(args.report_md)
    out_md.write_text(build_report_md(report), encoding="utf-8")

    print(json.dumps({"ok": report["ok"], "verdict": report["verdict"]}, ensure_ascii=False))

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
