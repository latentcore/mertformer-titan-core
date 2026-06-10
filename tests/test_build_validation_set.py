"""H5: validation set must exclude rows that leak from the training stage JSONLs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.build_validation_set as BVS  # noqa: E402


def test_fingerprint_matches_rolling_deduper():
    from scripts.data_pipeline import RollingDeduper
    f_val = BVS._training_fingerprinter()
    f_train = RollingDeduper(enabled=True)._fingerprint
    assert f_val("Hello World") == f_train("Hello World")
    # normalized (case + whitespace)
    assert f_val("hello   world") == f_val("Hello World")


def test_load_training_fingerprints_reads_stage_jsonls(tmp_path):
    stage = tmp_path / "stage1_data.jsonl"
    with stage.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "leaked training row"}) + "\n")
        f.write(json.dumps({"text": "another row"}) + "\n")
    fps, used = BVS._load_training_fingerprints([stage])
    assert len(fps) == 2
    assert str(stage) in used
    fp = BVS._training_fingerprinter()
    assert fp("leaked training row") in fps


def test_offline_rebuild_excludes_training_leak(monkeypatch, tmp_path):
    # No network: offline_rebuild reads current val + golden only.
    monkeypatch.setattr(BVS, "_load_local_golden", lambda *_a, **_k: [])

    stage = tmp_path / "stage1_data.jsonl"
    with stage.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "this row leaks from training set abc"}) + "\n")

    val = tmp_path / "validation.jsonl"
    with val.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "this row leaks from training set abc"}) + "\n")
        f.write(json.dumps({"text": "a perfectly clean held-out validation row"}) + "\n")

    rows, prov = BVS.build_validation_set(
        target_size=10, seed=1, min_chars=5, max_chars=400,
        exclude_training=True, stage_paths=[stage],
        offline_rebuild=True, current_val_path=val,
    )
    texts = [r["text"] for r in rows]
    assert "a perfectly clean held-out validation row" in texts
    assert "this row leaks from training set abc" not in texts  # excluded
    assert prov["network_used"] is False
    assert prov["excluded_leak"].get("offline_rebuild", 0) >= 1


def test_strict_offline_without_stage_files_raises(tmp_path):
    with pytest.raises(RuntimeError):
        BVS.build_validation_set(
            target_size=5, seed=1, min_chars=5, max_chars=400,
            exclude_training=True, stage_paths=[tmp_path / "missing.jsonl"],
            strict_offline=True, offline_rebuild=True,
            current_val_path=tmp_path / "none.jsonl",
        )
