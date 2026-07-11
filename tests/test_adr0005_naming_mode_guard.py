"""ADR-0005 "one naming mode per logits_dir" guard.

[2026-07-11] ``scripts/validate_logit_alignment.py`` used to carry a module-level
``assert getattr(cfg, "adr_0005_single_naming_mode", True) == True`` — ``cfg`` never
sets that attribute anywhere in the repo, so the ``getattr`` always fell through to
``True``, and ``True == True`` can never fail. The assertion was mathematically
incapable of catching anything; the actual invariant it claimed to protect (ADR-0005
§"One-mode-per-logits_dir": parallel shards are named by first-seq-index, single-process
shards by chunk-index, and the two must never be mixed in one directory) was never
checked at all. These tests pin the real replacement: ``detect_shard_naming_mode`` /
``assert_single_naming_mode`` in ``scripts/precompute_logits_topk.py``, wired into both
``validate_logit_alignment.validate_stage`` and the parallel orchestrator's
``finalize_stage`` writer path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.precompute_logits_topk as P  # noqa: E402
import scripts.validate_logit_alignment as VLA  # noqa: E402


# ---------------------------------------------------------------------------
# Pure classifier
# ---------------------------------------------------------------------------

def test_chunk_index_mode_detected():
    assert P.detect_shard_naming_mode([0, 1, 2, 3]) == "chunk_index"


def test_first_seq_index_mode_detected():
    # chunk_size=2000, 4 blocks merged from a parallel run
    assert P.detect_shard_naming_mode([0, 2000, 4000, 6000]) == "first_seq_index"


def test_single_shard_is_not_mixed():
    assert P.detect_shard_naming_mode([0]) == "chunk_index"
    assert P.detect_shard_naming_mode([]) == "chunk_index"


def test_mixed_naming_detected():
    # A single-process run's 0, 1, 2 plus a stray parallel-worker shard at 2000.
    assert P.detect_shard_naming_mode([0, 1, 2, 2000]) == "mixed"


def test_inconsistent_step_detected_as_mixed():
    assert P.detect_shard_naming_mode([0, 3, 4, 9]) == "mixed"


# ---------------------------------------------------------------------------
# assert_single_naming_mode: genuinely raises on disk, genuinely silent when clean
# ---------------------------------------------------------------------------

def test_assert_single_naming_mode_raises_on_mixed_shards(tmp_path):
    logits_dir = tmp_path / "logits"
    logits_dir.mkdir()
    for part in (0, 1, 2, 2000):
        (logits_dir / f"stage1_train_part_{part}.pt").write_bytes(b"")

    with pytest.raises(ValueError, match="ADR-0005"):
        P.assert_single_naming_mode(logits_dir, "stage1")


def test_assert_single_naming_mode_silent_on_clean_chunk_index_shards(tmp_path):
    logits_dir = tmp_path / "logits"
    logits_dir.mkdir()
    for part in (0, 1, 2):
        (logits_dir / f"stage1_train_part_{part}.pt").write_bytes(b"")

    P.assert_single_naming_mode(logits_dir, "stage1")  # must not raise


def test_assert_single_naming_mode_silent_on_clean_first_seq_index_shards(tmp_path):
    logits_dir = tmp_path / "logits"
    logits_dir.mkdir()
    for part in (0, 2000, 4000):
        (logits_dir / f"stage1_train_part_{part}.pt").write_bytes(b"")

    P.assert_single_naming_mode(logits_dir, "stage1")  # must not raise


# ---------------------------------------------------------------------------
# End-to-end through validate_stage(): the actual former call site.
# ---------------------------------------------------------------------------

def test_validate_stage_fails_closed_on_mixed_shard_naming(monkeypatch, tmp_path):
    jsonl = tmp_path / "stage1.jsonl"
    jsonl.write_text('{"text": "alpha"}\n', encoding="utf-8")
    logits_dir = tmp_path / "logits"
    logits_dir.mkdir()
    for part in (0, 1, 2000):  # deliberately mixed: chunk-index + first-seq-index
        (logits_dir / f"stage1_train_part_{part}.pt").write_bytes(b"")

    monkeypatch.setattr(VLA, "STAGE_FILES", {1: jsonl})

    class _StubTok:
        name_or_path = "stub-tok"
        pad_token_id = 0
        eos_token_id = 2

        def __len__(self):
            return 64

    result = VLA.validate_stage(1, logits_dir, _StubTok())
    assert result["status"] == "FAIL"
    assert result["reason_code"] == "MIXED_SHARD_NAMING_MODE"
