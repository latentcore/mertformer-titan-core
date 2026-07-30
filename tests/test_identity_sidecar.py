"""Tests for the K-6 teacher-logit identity sidecar (2026-07-29).

``validate_logit_alignment._load_stored_identities`` used to ``torch.load`` every shard in
full just to read each item's small ``identity`` dict, materialising all the Top-K teacher
logits in host RAM to do it. Precompute now drops a ``<shard>.pt.identities.json`` sidecar
carrying only the wrapper metadata and the identities, and the validator reads that.

The sidecar is a CACHE. The property that matters most is not that it is fast but that it
can never produce a PASS the full load would not: a stale sidecar must be rejected, since
comparing freshly packed sequences against identities from a previous shard generation
would be a silent false PASS on the one gate that guarantees teacher/student alignment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

import scripts.precompute_logits_topk as PC  # noqa: E402
import scripts.validate_logit_alignment as VA  # noqa: E402


def _payload(n_items: int, hash_prefix: str = "h"):
    return {
        "format": "topk_packed_v1",
        "top_k": 8,
        "vocab_size": 128256,
        "max_seq_len": 512,
        "pad_id": 0,
        "eos_id": 2,
        "tokenizer_identity": {"name_or_path": "llama", "vocab_size": 128256},
        "packer_version": "packed_v1",
        "logits": [
            {
                "identity": {"len": 512, "hash": f"{hash_prefix}{i:04d}"},
                "topk_values": torch.zeros(4, 8),
                "topk_indices": torch.zeros(4, 8, dtype=torch.long),
            }
            for i in range(n_items)
        ],
    }


def _write_shard(directory: Path, part: int, n_items: int, hash_prefix: str = "h") -> Path:
    path = directory / f"stage1_train_part_{part}.pt"
    PC._atomic_torch_save(_payload(n_items, hash_prefix), path)
    return path


def test_sidecar_is_written_next_to_the_shard(tmp_path):
    shard = _write_shard(tmp_path, 0, 5)
    sidecar = PC.identity_sidecar_path(shard)

    assert sidecar.exists()
    assert sidecar.name == "stage1_train_part_0.pt.identities.json"

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["sidecar_format"] == "identities_v1"
    assert data["count"] == 5
    assert len(data["identities"]) == 5
    assert data["shard_bytes"] == shard.stat().st_size
    assert data["format"] == "topk_packed_v1"
    assert data["tokenizer_identity"]["vocab_size"] == 128256
    # The whole point: no logit tensors in the sidecar.
    assert "logits" not in data
    assert sidecar.stat().st_size < shard.stat().st_size


def test_sidecar_is_invisible_to_the_shard_glob(tmp_path):
    """A sidecar counted as a shard would corrupt every shard-count gate."""
    _write_shard(tmp_path, 0, 3)
    _write_shard(tmp_path, 1, 3)
    found = [p.name for p in PC._stage_shards(tmp_path, "stage1")]
    assert found == ["stage1_train_part_0.pt", "stage1_train_part_1.pt"]


def test_sidecar_path_and_full_load_agree_exactly(tmp_path):
    """The cache must be indistinguishable from the source of truth."""
    shards = [_write_shard(tmp_path, i, 5) for i in range(3)]

    meta_cached, ids_cached = VA._load_stored_identities(shards)
    for shard in shards:
        PC.identity_sidecar_path(shard).unlink()
    meta_full, ids_full = VA._load_stored_identities(shards)

    assert ids_cached == ids_full
    assert meta_cached == meta_full
    assert len(ids_cached) == 15
    # Identities stay in shard order; they are matched positionally by the validator.
    assert [i["hash"] for i in ids_cached[:3]] == ["h0000", "h0001", "h0002"]


def test_stale_sidecar_is_rejected_not_trusted(tmp_path):
    """The false-PASS guard: a regenerated shard must invalidate its old sidecar."""
    shard = _write_shard(tmp_path, 0, 5, hash_prefix="OLD")
    stale = json.loads(PC.identity_sidecar_path(shard).read_text(encoding="utf-8"))

    # Regenerate the shard with different content, then restore the outdated sidecar.
    _write_shard(tmp_path, 0, 7, hash_prefix="NEW")
    PC.identity_sidecar_path(shard).write_text(json.dumps(stale), encoding="utf-8")

    _, identities = VA._load_stored_identities([shard])
    assert len(identities) == 7, "stale sidecar's item count was trusted"
    assert identities[0]["hash"] == "NEW0000", "validator read stale identities"


def test_corrupt_sidecar_falls_back_instead_of_raising(tmp_path):
    shard = _write_shard(tmp_path, 0, 4)
    PC.identity_sidecar_path(shard).write_text("{ this is not json", encoding="utf-8")

    meta, identities = VA._load_stored_identities([shard])
    assert len(identities) == 4
    assert meta["format"] == "topk_packed_v1"


def test_unknown_sidecar_format_falls_back(tmp_path):
    """A future sidecar_format bump must degrade to a full load, not be misread."""
    shard = _write_shard(tmp_path, 0, 4)
    sidecar = PC.identity_sidecar_path(shard)
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data["sidecar_format"] = "identities_v99"
    sidecar.write_text(json.dumps(data), encoding="utf-8")

    _, identities = VA._load_stored_identities([shard])
    assert len(identities) == 4


def test_count_mismatch_inside_sidecar_falls_back(tmp_path):
    shard = _write_shard(tmp_path, 0, 4)
    sidecar = PC.identity_sidecar_path(shard)
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data["count"] = 99          # disagrees with len(identities)
    sidecar.write_text(json.dumps(data), encoding="utf-8")

    _, identities = VA._load_stored_identities([shard])
    assert len(identities) == 4


def test_mixed_sidecar_and_bare_shards_are_stitched_in_order(tmp_path):
    """Partially-regenerated shard sets are the realistic case after a resume."""
    shards = [_write_shard(tmp_path, i, 4, hash_prefix=f"s{i}_") for i in range(4)]
    PC.identity_sidecar_path(shards[1]).unlink()
    PC.identity_sidecar_path(shards[2]).unlink()

    _, identities = VA._load_stored_identities(shards)
    assert len(identities) == 16
    assert [i["hash"] for i in identities] == [
        f"s{shard}_{item:04d}" for shard in range(4) for item in range(4)
    ]


def test_sidecar_write_failure_does_not_lose_the_shard(tmp_path, monkeypatch):
    """The sidecar is best-effort; a cache write must never break precompute."""
    def _boom(*_args, **_kwargs):
        raise OSError("simulated sidecar write failure")

    monkeypatch.setattr(PC.os, "replace", _boom)
    shard_path = tmp_path / "stage1_train_part_0.pt"
    # os.replace is used for the shard too, so write it first, then break replace.
    monkeypatch.undo()
    shard = _write_shard(tmp_path, 0, 3)
    PC.identity_sidecar_path(shard).unlink()

    original_replace = PC.os.replace
    calls = {"n": 0}

    def _fail_second(src, dst):
        calls["n"] += 1
        if calls["n"] >= 2:      # let the shard land, break only the sidecar
            raise OSError("simulated sidecar write failure")
        return original_replace(src, dst)

    monkeypatch.setattr(PC.os, "replace", _fail_second)
    PC._atomic_torch_save(_payload(3), shard_path)

    assert shard_path.exists(), "shard was lost because its sidecar failed"
    assert not PC.identity_sidecar_path(shard_path).exists()
    # And the validator still works, via the full-load fallback.
    _, identities = VA._load_stored_identities([shard_path])
    assert len(identities) == 3


def test_non_dict_payload_is_skipped_without_error(tmp_path):
    """_atomic_torch_save is also used for payloads that carry no identities."""
    shard_path = tmp_path / "stage1_train_part_0.pt"
    PC._atomic_torch_save({"logits": [1, 2, 3]}, shard_path)

    sidecar = PC.identity_sidecar_path(shard_path)
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    # Items that are not dicts have no identity; the sidecar records that faithfully.
    assert data["identities"] == [None, None, None]

    _, identities = VA._load_stored_identities([shard_path])
    assert identities == [None, None, None]
