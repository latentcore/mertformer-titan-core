"""Integration smoke (B2): precompute_stage WRITER <-> train READER seam.

Proves the packed teacher-logit shard written by scripts.precompute_logits_topk is
consumed by train.PrecomputedCurriculumDataset with the per-sequence identity HARD-
asserting cleanly -- the one seam unit tests don't cover. Uses a tiny stub teacher +
stub tokenizer (no HF, no real model): a few hundred KB of RAM, sub-second, CPU-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

from scripts import precompute_logits_topk as P  # noqa: E402
from orchestrator.distillation_manager import PrecomputedLogitsIterable  # noqa: E402
from train import train as train_mod  # noqa: E402


class _StubTokenizer:
    name_or_path = "stub-integration-tok"
    pad_token_id = 0
    eos_token_id = 2
    pad_token = "[PAD]"

    def __len__(self):
        return 50

    def __call__(self, text, add_special_tokens=True, truncation=True, max_length=None):
        ids = [ord(c) % 40 + 3 for c in text]
        if max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


class _StubTeacher(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self._p = torch.nn.Linear(1, 1)  # gives .parameters() a device

    def forward(self, ids):
        b, s = ids.shape
        torch.manual_seed(int(ids.sum().item()) % 100000)
        return SimpleNamespace(logits=torch.randn(b, s, self.vocab))


class _DM:
    def __init__(self, logits_dir):
        self.logits_dir = logits_dir

    def get_precomputed_loader(self, stage_name):
        return PrecomputedLogitsIterable(self.logits_dir, stage_name, subset="train")


def test_precompute_writer_matches_train_reader(monkeypatch, tmp_path):
    tok = _StubTokenizer()
    teacher = _StubTeacher(vocab=len(tok))

    jsonl = tmp_path / "stage1.jsonl"
    rows = ["alpha beta", "gamma", "delta epsilon zeta", "eta", "theta iota kappa lambda"]
    with jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"text": r}) + "\n")

    logits_dir = tmp_path / "logits"
    logits_dir.mkdir()
    monkeypatch.setitem(P.STAGE_FILES, 1, jsonl)

    # --- WRITER: real precompute_stage (packed shards + identity) ---
    P.precompute_stage(
        stage_num=1, teacher=teacher, tokenizer=tok, logits_dir=logits_dir,
        top_k=4, chunk_size=2, batch_size=2, max_seq=8,
    )
    shards = sorted(logits_dir.glob("stage1_train_part_*.pt"))
    assert shards, "precompute wrote no shards"

    # --- READER: the real train dataset consumes them, asserting identity ---
    ds = object.__new__(train_mod.PrecomputedCurriculumDataset)
    ds.stage_info = [("stage1", jsonl)]
    ds.max_len = 8
    ds.tokenizer = tok
    ds.distill_manager = _DM(logits_dir)
    ds.current_stage = 1
    ds.pad_id = tok.pad_token_id

    produced = list(ds._iter_stage("stage1", jsonl))
    assert produced, "train reader produced no packed sequences"

    for input_ids, labels, payload in produced:
        assert input_ids.shape == (8,)
        assert labels.shape == (8,)
        # trailing pad is ignored (-100); at least one real supervised token.
        assert int((labels != -100).sum().item()) >= 1
        # teacher payload aligned to the packed length, carrying identity.
        assert payload["format"] == "topk_packed_v1"
        assert "identity" in payload
        assert payload["indices"].shape[0] == 8  # one row per packed position


def test_precompute_resume_produces_aligned_shards(monkeypatch, tmp_path):
    """tier-2 HIGH: an interrupted-then-resumed precompute must still produce shards
    whose per-sequence identity matches a fresh single-pass pack (no resume-seam
    drift). Re-pack-from-0 + skip-already-emitted guarantees this."""
    import scripts.validate_logit_alignment as VLA

    tok = _StubTokenizer()
    teacher = _StubTeacher(vocab=len(tok))
    jsonl = tmp_path / "stage1.jsonl"
    rows = ["one two three four", "five", "six seven eight nine ten", "eleven", "twelve thirteen"]
    with jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"text": r}) + "\n")

    logits_dir = tmp_path / "logits"
    logits_dir.mkdir()
    monkeypatch.setitem(P.STAGE_FILES, 1, jsonl)
    monkeypatch.setitem(VLA.STAGE_FILES, 1, jsonl)

    # First pass: chunk_size=1 flushes after every sequence (many shards + state).
    P.precompute_stage(stage_num=1, teacher=teacher, tokenizer=tok, logits_dir=logits_dir,
                       top_k=4, chunk_size=1, batch_size=1, max_seq=8)
    shards = sorted(logits_dir.glob("stage1_train_part_*.pt"))
    assert len(shards) >= 2

    # Simulate an interruption AFTER the first shard: drop later shards + rewind state.
    import torch as _t
    first = _t.load(shards[0], map_location="cpu", weights_only=False)
    emitted_in_first = len(first["logits"])
    for s in shards[1:]:
        s.unlink()
    P._save_resume_state(logits_dir, "stage1", lines_consumed=1, sequences_emitted=emitted_in_first)

    # Resume: re-pack from 0, skip already-emitted, regenerate the rest.
    P.precompute_stage(stage_num=1, teacher=teacher, tokenizer=tok, logits_dir=logits_dir,
                       top_k=4, chunk_size=1, batch_size=1, max_seq=8)

    # The resumed shard set must align to a fresh single-pass pack.
    res = VLA.validate_stage(1, logits_dir, tok)
    assert res["status"] == "PASS", res
