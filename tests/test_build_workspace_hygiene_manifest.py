from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.build_workspace_hygiene_manifest as hygiene


def test_build_manifest_classifies_workspace_entries(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    repo_root = workspace_root / "repo"
    repo_root.mkdir(parents=True)
    (workspace_root / ".idea").mkdir()
    (workspace_root / ".ruff_cache").mkdir()
    (workspace_root / ".DS_Store").write_text("", encoding="utf-8")
    (workspace_root / "old_bundle.zip").write_text("zip", encoding="utf-8")

    payload = hygiene.build_manifest(
        workspace_root=workspace_root,
        repo_root=repo_root,
        quarantine_root=workspace_root / "workspace_quarantine",
        mode="audit_only",
    )

    by_path = {Path(item["path"]).name: item for item in payload["items"]}
    assert payload["schema"] == "workspace_hygiene_manifest_v1"
    assert by_path["repo"]["decision_state"] == "keep"
    assert by_path[".idea"]["decision_state"] == "keep"
    assert by_path[".ruff_cache"]["decision_state"] == "quarantine_first"
    assert by_path[".DS_Store"]["decision_state"] == "ignore"
    assert by_path["old_bundle.zip"]["classification"] == "archive_or_dump"


def test_apply_quarantine_moves_candidates_only(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    repo_root = workspace_root / "repo"
    repo_root.mkdir(parents=True)
    candidate = workspace_root / "old_bundle.zip"
    candidate.write_text("zip", encoding="utf-8")
    quarantine_root = workspace_root / "workspace_quarantine"

    payload = hygiene.build_manifest(
        workspace_root=workspace_root,
        repo_root=repo_root,
        quarantine_root=quarantine_root,
        mode="apply_quarantine",
    )
    moved = hygiene.apply_quarantine(payload, quarantine_root)

    assert moved
    assert not candidate.exists()
    assert quarantine_root.exists()
    assert (quarantine_root / "old_bundle.zip").exists()


def test_markdown_renders_decisions_table(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    repo_root = workspace_root / "repo"
    repo_root.mkdir(parents=True)
    (workspace_root / ".ruff_cache").mkdir(parents=True)
    payload = hygiene.build_manifest(
        workspace_root=workspace_root,
        repo_root=repo_root,
        quarantine_root=workspace_root / "workspace_quarantine",
        mode="audit_only",
    )
    md = hygiene.build_markdown(payload, [])
    assert "Workspace Hygiene Manifest" in md
    assert "quarantine_first" in md
