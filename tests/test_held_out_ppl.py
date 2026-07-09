"""Tests for eval/held_out_ppl.py's deterministic helpers.

These cover the parts that do NOT require loading the canonical model (which is
hardware-bound): the file hash, the corpus row parser's skip-predicate, the schema
constant, and the UTC timestamp format. The end-to-end perplexity path is exercised by
the run itself (checkpoint-bound), not by a unit test.
"""
import hashlib
import json

import eval.held_out_ppl as H


def test_schema_constant():
    assert H.SCHEMA == "held_out_ppl_v1"


def test_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"mertformer-titan")
    assert H._sha256(p) == hashlib.sha256(b"mertformer-titan").hexdigest()


def test_sha256_missing_returns_none(tmp_path):
    assert H._sha256(tmp_path / "does_not_exist.bin") is None


def test_corpus_rows_skips_blank_invalid_and_nondict(tmp_path):
    p = tmp_path / "corpus.jsonl"
    lines = [
        json.dumps({"text": "hello"}),
        "",                       # blank -> skipped
        "{not valid json",        # unparseable -> skipped
        json.dumps(["a", "b"]),   # non-dict -> skipped
        json.dumps({"text": "world"}),
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    rows = list(H._corpus_rows(p))
    texts = [t for _, t in rows]
    assert texts == ["hello", "world"]


def test_utc_now_is_zulu_iso():
    s = H._utc_now()
    assert s.endswith("Z") and "T" in s and len(s) == 20
