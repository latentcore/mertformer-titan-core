#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sanitize_text(text: str) -> str:
    out = text
    out = out.replace(str(ROOT), "<REPO_ROOT>")
    out = re.sub(r"/Users/[^/]+/Desktop/[^\s\"']+", "<DESKTOP_PATH>", out)
    return out


def run(cmd: list[str]) -> dict:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return {
        "cmd": " ".join(cmd),
        "return_code": p.returncode,
        "stdout_tail": sanitize_text(p.stdout[-4000:]),
        "stderr_tail": sanitize_text(p.stderr[-4000:]),
        "ok": p.returncode == 0,
    }


def main() -> int:
    steps = {
        "secret_scan": run([".titan-venv/bin/python", "scripts/secret_scan.py"]),
        "check_57_matrix": run([".titan-venv/bin/python", "scripts/check_57_matrix.py"]),
        "verify_all": run(["bash", "scripts/verify_all.sh"]),
        "git_status": run(["git", "status", "--short", "--branch"]),
        "git_remote": run(["git", "remote", "-v"]),
    }
    ok = all(s["ok"] for s in steps.values())
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "steps": steps,
    }
    out = ROOT / "reports" / "start_gate_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
