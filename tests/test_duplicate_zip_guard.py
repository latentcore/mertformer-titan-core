from __future__ import annotations

from pathlib import Path

from scripts import duplicate_zip_guard


def test_sanitize_path_redacts_repo_and_documents_roots(tmp_path: Path):
    repo_root = tmp_path / "repo"
    docs_root = tmp_path / "Documents"
    target = docs_root / "nested" / "artifact.zip"
    sanitized = duplicate_zip_guard.sanitize_path(target, repo_root, docs_root)
    assert "<DOCUMENTS_PATH>" in sanitized
    assert str(docs_root) not in sanitized
