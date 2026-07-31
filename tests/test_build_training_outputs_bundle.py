from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_build_training_outputs_bundle_collects_outputs_and_excludes_sensitive_files(tmp_path: Path):
    project_root = tmp_path / "repo"
    (project_root / "logs").mkdir(parents=True)
    (project_root / "checkpoints").mkdir(parents=True)
    (project_root / "reports").mkdir(parents=True)
    (project_root / "artifacts").mkdir(parents=True)
    (project_root / "packages").mkdir(parents=True)
    (project_root / "datasets" / "logits").mkdir(parents=True)
    (project_root / "repro").mkdir(parents=True)

    (project_root / "logs" / "run.log").write_text("log\n", encoding="utf-8")
    (project_root / "logs" / "__pycache__").mkdir()
    (project_root / "logs" / "__pycache__" / "ignored.pyc").write_bytes(b"x")
    (project_root / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
    (project_root / "reports" / "summary.md").write_text("# report\n", encoding="utf-8")
    (project_root / "artifacts" / "mertformer_release.zip").write_bytes(b"zip")
    (project_root / "artifacts" / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (project_root / "packages" / "bundle.zip").write_bytes(b"pkg")
    (project_root / "datasets" / "logits" / "stage1_train_part_0.pt").write_bytes(b"logits")
    (project_root / "repro" / "cuda.lock").write_text("gpu=ok\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_training_outputs_bundle.py"),
            "--project-root",
            str(project_root),
            "--reports-dir",
            str(project_root / "reports"),
            "--artifacts-dir",
            str(project_root / "artifacts"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    bundle_zip = project_root / "artifacts" / "mertformer_training_outputs_bundle.zip"
    bundle_sha = project_root / "artifacts" / "mertformer_training_outputs_bundle.zip.sha256"
    manifest_json = project_root / "reports" / "training_outputs_bundle_manifest.json"
    manifest_md = project_root / "reports" / "training_outputs_bundle_manifest.md"

    assert bundle_zip.exists()
    assert bundle_sha.exists()
    assert manifest_json.exists()
    assert manifest_md.exists()

    payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert payload["included_files"] >= 6
    assert payload["integrity"]["zipfile_crc_ok"] is True

    with zipfile.ZipFile(bundle_zip, "r") as archive:
        names = set(archive.namelist())

    assert "logs/run.log" in names
    assert "checkpoints/latest.pt" in names
    assert "reports/summary.md" in names
    assert "artifacts/mertformer_release.zip" in names
    assert "packages/bundle.zip" in names
    assert "datasets/logits/stage1_train_part_0.pt" in names
    assert "repro/cuda.lock" in names
    assert "artifacts/mertformer_training_outputs_bundle.zip" not in names
    assert "artifacts/mertformer_training_outputs_bundle.zip.sha256" not in names
    assert "reports/training_outputs_bundle_manifest.json" not in names
    assert "reports/training_outputs_bundle_manifest.md" not in names
    assert "artifacts/.env" not in names
    assert "logs/__pycache__/ignored.pyc" not in names
