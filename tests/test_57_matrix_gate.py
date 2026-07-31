from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_closure_57_gate_returns_all_green_and_writes_reports(tmp_path: Path):
    out = tmp_path / "closure_57.json"
    md = tmp_path / "closure_57.md"
    md_tr = tmp_path / "closure_57_TR.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_57_matrix.py",
            "--out",
            str(out),
            "--md-out",
            str(md),
            "--md-tr-out",
            str(md_tr),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "closure_57_matrix_v1"
    assert payload["total_items"] == 57
    assert payload["green_items"] == 57
    assert payload["all_green"] is True
    assert md.exists()
    assert md_tr.exists()

