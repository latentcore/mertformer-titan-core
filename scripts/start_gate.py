#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sanitize_text(text: str) -> str:
    out = text
    out = out.replace(str(ROOT), "<REPO_ROOT>")
    out = re.sub(r"/Users/[^/]+/Desktop/[^\s\"']+", "<DESKTOP_PATH>", out)
    return out


def sanitize_value(value):
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {sanitize_value(key): sanitize_value(item) for key, item in value.items()}
    return value


def run(cmd: list[str], py: str | None = None) -> dict:
    resolved = [py if token == "<PY>" and py else token for token in cmd]
    p = subprocess.run(resolved, cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "cmd": sanitize_text(" ".join(resolved)),
        "return_code": p.returncode,
        "stdout_tail": sanitize_text(p.stdout[-4000:]),
        "stderr_tail": sanitize_text(p.stderr[-4000:]),
        "ok": p.returncode == 0,
    }


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Closure start gate with exact readiness reporting.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--report-out", default=str(ROOT / "reports" / "start_gate_report.json"))
    parser.add_argument("--skip-verify-all", action="store_true")
    parser.add_argument("--allow-not-ready", action="store_true")
    parser.add_argument("--require-train-allowed", action="store_true")
    args = parser.parse_args()

    steps = {
        "secret_scan": run(["<PY>", "scripts/secret_scan.py"], py=args.python),
        "check_57_matrix": run(["<PY>", "scripts/check_57_matrix.py"], py=args.python),
        "train_readiness_contract": run(
            ["<PY>", "scripts/build_train_readiness_contract.py", "--allow-not-ready"],
            py=args.python,
        ),
        "git_status": run(["git", "status", "--short", "--branch"]),
        "git_remote": run(["git", "remote", "-v"]),
    }
    if not args.skip_verify_all:
        steps["verify_all"] = run(["bash", "scripts/verify_all.sh"])

    readiness = load_json(ROOT / "reports" / "train_readiness_decision.json")
    structural_ok = all(step["ok"] for step in steps.values())
    train_allowed = readiness.get("final_status") == "TRAIN_ALLOWED"
    start_allowed = structural_ok and train_allowed

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ok": structural_ok,
        "start_allowed": start_allowed,
        "train_allowed": train_allowed,
        "train_readiness_status": readiness.get("final_status"),
        "decision_reason_code": readiness.get("decision_reason_code"),
        "blockers": readiness.get("blockers", []),
        "steps": steps,
    }

    out = Path(args.report_out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sanitize_value(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": structural_ok,
                "train_allowed": train_allowed,
                "decision_reason_code": readiness.get("decision_reason_code"),
            },
            ensure_ascii=False,
        )
    )

    if not structural_ok:
        return 1
    if args.require_train_allowed and not train_allowed and not args.allow_not_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
