from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_kaggle_compare_script_generates_reports(tmp_path: Path):
    out_dir = tmp_path / "bench"
    result = subprocess.run(
        [
            "python3",
            "scripts/kaggle_train_compare_build30.py",
            "--quick",
            "--steps",
            "2",
            "--batch-size",
            "1",
            "--seq-len",
            "16",
            "--vocab-size",
            "256",
            "--hidden",
            "64",
            "--layers",
            "2",
            "--heads",
            "4",
            "--out-dir",
            str(out_dir),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    json_path = out_dir / "kaggle_compare_build30.json"
    csv_path = out_dir / "kaggle_compare_build30.csv"
    md_path = out_dir / "kaggle_compare_build30.md"

    assert json_path.exists()
    assert csv_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "kaggle_compare_build30_v1"
    assert "mertformer" in payload and "vanilla" in payload
