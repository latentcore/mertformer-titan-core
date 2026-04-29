from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.build_scoped_external_intake_matrix as intake
import scripts.cleanup_scoped_closure_junk as cleanup


def test_collect_entries_can_run_with_temp_scope(monkeypatch, tmp_path: Path) -> None:
    file_a = tmp_path / "mertformer-a.zip"
    file_b = tmp_path / "mertformer-b.zip"
    file_a.write_bytes(b"same")
    file_b.write_bytes(b"same")
    monkeypatch.setattr(intake, "SCOPED_PATTERNS", [file_a, file_b])
    monkeypatch.setattr(intake, "SCAN_ROOTS", [])
    entries = intake.collect_entries()
    assert len(entries) == 2
    assert all(entry["disposition"] == "delete_as_stale_generated" for entry in entries)


def test_collect_entries_marks_immutable_and_canonical_sources(monkeypatch, tmp_path: Path) -> None:
    immutable_zip = tmp_path / "mertformer_outputs_history.zip"
    immutable_zip.write_bytes(b"immutable")
    release_copy = tmp_path / "mertformer-titan-core.zip"
    release_copy.write_bytes(b"release")
    monkeypatch.setattr(intake, "SCOPED_PATTERNS", [immutable_zip, release_copy])
    monkeypatch.setattr(intake, "SCAN_ROOTS", [])
    monkeypatch.setattr(intake, "ARTIFACTS", tmp_path)
    entries = intake.collect_entries()
    by_name = {Path(entry["path"]).name: entry for entry in entries}
    assert by_name["mertformer_outputs_history.zip"]["immutable_evidence"] is True
    assert by_name["mertformer_outputs_history.zip"]["disposition"] == "preserve_immutable_evidence"
    assert by_name["mertformer-titan-core.zip"]["canonical_source"] is not None


def test_cleanup_scoped_junk_removes_pycache(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    junk_dir = repo_root / "__pycache__"
    junk_dir.mkdir()
    (junk_dir / "demo.pyc").write_bytes(b"x")
    intake_path = tmp_path / "intake.json"
    intake_path.write_text(
        json.dumps({"entries": [{"path": str(repo_root), "kind": "dir", "mutation_policy": "project_safe_cleanup"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cleanup, "ROOT", repo_root)
    out = tmp_path / "cleanup.json"
    monkeypatch.setattr(sys, "argv", ["cleanup", "--apply", "--intake", str(intake_path), "--out", str(out)])
    cleanup.main()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["removed_count"] >= 1
    assert not junk_dir.exists()
