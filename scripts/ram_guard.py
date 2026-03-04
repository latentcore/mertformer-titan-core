#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a command with system RAM guard")
    ap.add_argument("--warn", type=float, default=10.5)
    ap.add_argument("--slow", type=float, default=12.0)
    ap.add_argument("--hard", type=float, default=13.0)
    ap.add_argument("--poll-sec", type=float, default=0.5)
    ap.add_argument("--out", default="reports/ram_guard_report.json")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    if not args.cmd:
        raise SystemExit("No command provided")

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    start = time.time()
    proc = subprocess.Popen(cmd)

    peak_used_gb = 0.0
    events = []
    terminated_by_guard = False

    while proc.poll() is None:
        vm = psutil.virtual_memory()
        used_gb = (vm.total - vm.available) / (1024 ** 3)
        peak_used_gb = max(peak_used_gb, used_gb)

        if used_gb >= args.hard:
            events.append({"event": "hard_stop", "used_gb": used_gb, "ts": time.time()})
            proc.terminate()
            terminated_by_guard = True
            break
        elif used_gb >= args.slow:
            events.append({"event": "slow_mode", "used_gb": used_gb, "ts": time.time()})
        elif used_gb >= args.warn:
            events.append({"event": "warn", "used_gb": used_gb, "ts": time.time()})

        time.sleep(args.poll_sec)

    if terminated_by_guard:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    rc = proc.returncode if proc.returncode is not None else (130 if terminated_by_guard else 1)
    duration = time.time() - start

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(x) for x in cmd),
        "warn_threshold_gb": args.warn,
        "slow_threshold_gb": args.slow,
        "hard_threshold_gb": args.hard,
        "peak_used_gb": round(peak_used_gb, 3),
        "terminated_by_guard": terminated_by_guard,
        "events": events,
        "return_code": int(rc),
        "duration_sec": round(duration, 3),
        "ok": (rc == 0 and not terminated_by_guard),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
