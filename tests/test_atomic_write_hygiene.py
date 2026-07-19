"""Regression tests for the 2026-07-19 atomic-write hygiene pass.

Five previously non-atomic writes (data_pipeline.py, precompute_logits_topk.py x2,
chess_5080_onefile.py, record_dataset_hashes.py, build_validation_set.py) all shared
one bug class: a direct write to a path that some OTHER piece of code treats as
"done/complete" via a bare `.exists()` check, with no temp-file + os.replace() step —
so a crash mid-write could leave a corrupt/truncated file that is silently accepted
as valid on the next run. Each test below verifies the fixed function (a) produces
correct final content on success, (b) leaves no stray `.tmp` file behind, and (c)
never disturbs a pre-existing final file unless the new write actually completes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# precompute_logits_topk.py::_atomic_torch_save
# ---------------------------------------------------------------------------
def test_atomic_torch_save_success_leaves_no_tmp(tmp_path):
    import scripts.precompute_logits_topk as P

    shard_path = tmp_path / "stage1_train_part_0.pt"
    P._atomic_torch_save({"logits": [1, 2, 3]}, shard_path)

    assert shard_path.exists()
    assert not shard_path.with_suffix(shard_path.suffix + ".tmp").exists()
    import torch
    assert torch.load(shard_path, weights_only=False) == {"logits": [1, 2, 3]}


def test_atomic_torch_save_failure_preserves_old_shard(tmp_path, monkeypatch):
    import scripts.precompute_logits_topk as P

    shard_path = tmp_path / "stage1_train_part_0.pt"
    # Pre-existing, previously-completed shard.
    import torch
    torch.save({"logits": ["old", "good", "shard"]}, shard_path)

    def _boom(*_a, **_k):
        raise RuntimeError("simulated crash mid torch.save")

    monkeypatch.setattr(P.torch, "save", _boom)
    with pytest.raises(RuntimeError):
        P._atomic_torch_save({"logits": ["new", "partial"]}, shard_path)

    # The old, complete shard must survive untouched -- this is the exact bug: a
    # downstream .exists()-only resume/coverage check must never see a corrupt file.
    assert torch.load(shard_path, weights_only=False) == {"logits": ["old", "good", "shard"]}


# ---------------------------------------------------------------------------
# chess_5080_onefile.py::atomic_torch_save
# ---------------------------------------------------------------------------
def test_chess_atomic_torch_save_success_and_failure(tmp_path, monkeypatch):
    import scripts.chess_5080_onefile as C
    import torch

    ckpt_path = tmp_path / "nested" / "latest.pt"
    C.atomic_torch_save(ckpt_path, {"step": 1})
    assert ckpt_path.exists()
    assert torch.load(ckpt_path, weights_only=False) == {"step": 1}
    assert not ckpt_path.with_suffix(ckpt_path.suffix + ".tmp").exists()

    # Now simulate a crash on a re-save and confirm the old checkpoint survives.
    def _boom(*_a, **_k):
        raise RuntimeError("simulated crash mid torch.save")

    monkeypatch.setattr(C.torch, "save", _boom)
    with pytest.raises(RuntimeError):
        C.atomic_torch_save(ckpt_path, {"step": 2})
    assert torch.load(ckpt_path, weights_only=False) == {"step": 1}


# ---------------------------------------------------------------------------
# record_dataset_hashes.py::_atomic_write_json
# ---------------------------------------------------------------------------
def test_record_dataset_hashes_atomic_write_success_and_failure(tmp_path, monkeypatch):
    import scripts.record_dataset_hashes as R

    out = tmp_path / "hashes.json"
    out.write_text(json.dumps({"schema_version": 1, "sources": {"old": True}}), encoding="utf-8")

    R._atomic_write_json(out, {"schema_version": 1, "sources": {"new": True}})
    assert json.loads(out.read_text(encoding="utf-8"))["sources"] == {"new": True}
    assert not out.with_suffix(out.suffix + ".tmp").exists()

    # Simulate the write failing partway (e.g. disk full while serializing) -- the
    # offline-readiness gate this file feeds only checks .exists(), so a truncated
    # file here would previously have silently PASSed.
    real_write_text = Path.write_text

    def _boom(self, *a, **k):
        if self.name.endswith(".tmp"):
            raise OSError("simulated disk-full mid-write")
        return real_write_text(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", _boom)
    with pytest.raises(OSError):
        R._atomic_write_json(out, {"schema_version": 1, "sources": {"corrupt": True}})
    monkeypatch.undo()
    assert json.loads(out.read_text(encoding="utf-8"))["sources"] == {"new": True}


# ---------------------------------------------------------------------------
# build_validation_set.py::_atomic_write_jsonl
# ---------------------------------------------------------------------------
def test_build_validation_set_atomic_write_success_and_failure(tmp_path, monkeypatch):
    import scripts.build_validation_set as BVS

    out = tmp_path / "validation.jsonl"
    out.write_text('{"text": "old row 1"}\n{"text": "old row 2"}\n', encoding="utf-8")

    BVS._atomic_write_jsonl(out, [{"text": "new row"}])
    lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == "new row"
    assert not out.with_suffix(out.suffix + ".tmp").exists()

    def _boom_open(self, mode="r", *a, **k):
        if self.name.endswith(".tmp") and "w" in mode:
            raise OSError("simulated crash mid-write")
        return real_open(self, mode, *a, **k)

    real_open = Path.open
    monkeypatch.setattr(Path, "open", _boom_open)
    with pytest.raises(OSError):
        BVS._atomic_write_jsonl(out, [{"text": "partial row"}])
    monkeypatch.undo()

    lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == "new row"


# ---------------------------------------------------------------------------
# data_pipeline.py::download_stage — full end-to-end with a mocked HF source
# ---------------------------------------------------------------------------
class _FakeStreamedDataset:
    def __init__(self, samples):
        self._samples = list(samples)

    def shuffle(self, seed=42, buffer_size=10_000):
        return self

    def __iter__(self):
        return iter(self._samples)


def test_download_stage_replaces_old_file_atomically(tmp_path, monkeypatch):
    import scripts.data_pipeline as DP

    stage_dir = tmp_path / "stage1"
    stage_dir.mkdir()
    monkeypatch.setitem(DP.STAGE_DIRS, 1, stage_dir)

    stage_output = stage_dir / "stage1_data.jsonl"
    stage_output.write_text('{"text": "stale data from a previous, now-superseded run"}\n', encoding="utf-8")

    samples = [{"text": f"fresh sample number {i} with enough length to pass filters"} for i in range(5)]
    monkeypatch.setattr(DP, "load_dataset", lambda *a, **k: _FakeStreamedDataset(samples))
    monkeypatch.setattr(DP, "get_hf_revision", lambda *a, **k: None)

    sources = [
        {
            "dataset": "fake/dataset",
            "subset": None,
            "split": "train",
            "ratio": 1.0,
            "field": "text",
            "min_length": 1,
            "max_length": 1000,
            "filters": None,
            "optional": False,
        }
    ]

    result = DP.download_stage(1, sources, target_samples_per_source=5)

    assert result["collected"] == 5
    assert not stage_output.with_suffix(stage_output.suffix + ".tmp").exists()
    final_lines = [l for l in stage_output.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(final_lines) == 5
    assert "stale data" not in stage_output.read_text(encoding="utf-8")
    for line in final_lines:
        assert "fresh sample number" in json.loads(line)["text"]


def test_download_stage_no_active_sources_leaves_old_file_untouched(tmp_path, monkeypatch):
    """Pre-existing behavior (already correct) -- kept as a regression guard next to
    the new atomic-replace test above, so both halves of the "don't corrupt old data"
    contract are covered in one file."""
    import scripts.data_pipeline as DP

    stage_dir = tmp_path / "stage2"
    stage_dir.mkdir()
    monkeypatch.setitem(DP.STAGE_DIRS, 2, stage_dir)

    stage_output = stage_dir / "stage2_data.jsonl"
    stage_output.write_text('{"text": "must survive: no source connects this run"}\n', encoding="utf-8")

    def _raise_connect(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(DP, "load_dataset", _raise_connect)

    sources = [
        {
            "dataset": "fake/dataset",
            "subset": None,
            "split": "train",
            "ratio": 1.0,
            "field": "text",
            "min_length": 1,
            "max_length": 1000,
            "filters": None,
            "optional": False,
        }
    ]

    result = DP.download_stage(2, sources, target_samples_per_source=5)
    assert result["collected"] == 0
    assert "must survive" in stage_output.read_text(encoding="utf-8")
    assert not stage_output.with_suffix(stage_output.suffix + ".tmp").exists()
