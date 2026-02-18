from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mertformer_sdk.kpi import collect_kpis


def test_collect_kpis_has_12_checks_without_verify():
    payload = collect_kpis(project_root=Path('.'), run_verify=False, run_onnx_check=False)
    assert payload["schema"] == "kpi_report_v1"
    assert len(payload["checks"]) == 12


def test_cli_kpi_report_writes_file(tmp_path: Path):
    out_path = tmp_path / "kpi_report.json"
    result = subprocess.run(
        [
            "python3",
            "-m",
            "mertformer_sdk.cli",
            "kpi-report",
            "--out",
            str(out_path),
            "--skip-verify",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "kpi_report_v1"
