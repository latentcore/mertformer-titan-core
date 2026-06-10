"""B2 gate: validate_logit_alignment must PASS aligned shards and FAIL drift."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

from train import packing  # noqa: E402
import scripts.validate_logit_alignment as VLA  # noqa: E402
from utils.tokenizer_resolver import tokenizer_identity  # noqa: E402


class _StubTok:
    name_or_path = "stub-tok"
    pad_token_id = 0
    eos_token_id = 2

    def __len__(self):
        return 64

    def __call__(self, text, add_special_tokens=True, truncation=True, max_length=None):
        ids = [ord(c) % 40 + 3 for c in text]
        if max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def _write_stage(tmp_path, rows):
    p = tmp_path / "stage1.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"text": r}) + "\n")
    return p


def _write_shard(logits_dir, tok, jsonl, max_seq, corrupt=False):
    eos_id, pad_id = tok.eos_token_id, tok.pad_token_id
    rows = [(i, r) for i, r in enumerate(json.loads(l)["text"] for l in jsonl.open())]
    seqs = list(packing.iter_packed_sequences(rows, tok, max_seq, eos_id, pad_id))
    items = []
    for s in seqs:
        ident = dict(s["identity"])
        if corrupt:
            ident["hash"] = "deadbeef" * 4
        items.append({
            "indices": torch.zeros(max_seq, 4, dtype=torch.int32),
            "values": torch.zeros(max_seq, 4, dtype=torch.bfloat16),
            "true_len": s["true_len"],
            "seq_index": s["seq_index"],
            "row_span": s["row_span"],
            "identity": ident,
        })
    payload = {
        "format": packing.TOPK_PACKED_FORMAT,
        "top_k": 4,
        "vocab_size": len(tok),
        "max_seq_len": max_seq,
        "pad_id": pad_id,
        "eos_id": eos_id,
        "tokenizer_identity": tokenizer_identity(tok, None),
        "packer_version": "packed_v1",
        "logits": items,
    }
    logits_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, logits_dir / "stage1_train_part_0.pt")


def test_aligned_stage_passes(monkeypatch, tmp_path):
    tok = _StubTok()
    jsonl = _write_stage(tmp_path, ["alpha", "beta", "gamma delta epsilon"])
    logits_dir = tmp_path / "logits"
    _write_shard(logits_dir, tok, jsonl, max_seq=8, corrupt=False)
    monkeypatch.setattr(VLA, "STAGE_FILES", {1: jsonl})

    res = VLA.validate_stage(1, logits_dir, tok)
    assert res["status"] == "PASS", res
    assert res["reason_code"] == "ALIGNED"


def test_identity_mismatch_fails(monkeypatch, tmp_path):
    tok = _StubTok()
    jsonl = _write_stage(tmp_path, ["alpha", "beta", "gamma delta epsilon"])
    logits_dir = tmp_path / "logits"
    _write_shard(logits_dir, tok, jsonl, max_seq=8, corrupt=True)
    monkeypatch.setattr(VLA, "STAGE_FILES", {1: jsonl})

    res = VLA.validate_stage(1, logits_dir, tok)
    assert res["status"] == "FAIL"
    assert res["reason_code"] == "IDENTITY_MISMATCH"


def test_missing_shards_reports_missing(monkeypatch, tmp_path):
    tok = _StubTok()
    jsonl = _write_stage(tmp_path, ["alpha"])
    monkeypatch.setattr(VLA, "STAGE_FILES", {1: jsonl})
    res = VLA.validate_stage(1, tmp_path / "empty", tok)
    assert res["status"] == "MISSING"
