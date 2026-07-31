from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_drone_sitl_demo_generates_artifacts(tmp_path: Path):
    out_root = tmp_path / "pilots"

    cmd = [
        sys.executable,
        "scripts/drone_sitl_demo.py",
        "--pilot-id",
        "pilot_test",
        "--runs",
        "2",
        "--steps",
        "40",
        "--fault-start",
        "10",
        "--fault-duration",
        "5",
        "--seed",
        "27",
        "--out-root",
        str(out_root),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    payload = json.loads(result.stdout)
    summary_path = Path(payload["summary_path"])
    events_path = Path(payload["events_path"])
    report_path = Path(payload["report_path"])

    assert summary_path.exists()
    assert events_path.exists()
    assert report_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["all_green"] is True
    assert len(summary["runs"]) == 2
    assert all(run["fallback_triggered"] for run in summary["runs"])
