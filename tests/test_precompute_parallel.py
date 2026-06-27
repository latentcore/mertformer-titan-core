"""Tests for multi-GPU data-parallel Top-K precompute (block-cyclic sharding).

Proves the parallel lane is byte-compatible with the canonical single-process lane
at the level that matters — the merged on-disk shard stream reads back in exact
global seq_index order with identical per-sequence identities — and that the
orchestrator's coverage/finalize/validation logic is correct. CPU-only, stub
teacher + stub tokenizer (no HF, no GPUs): sub-second.
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
from scripts import precompute_logits_parallel as O  # noqa: E402
import scripts.validate_logit_alignment as VLA  # noqa: E402


class _StubTokenizer:
    name_or_path = "stub-parallel-tok"
    pad_token_id = 0
    eos_token_id = 2
    pad_token = "[PAD]"

    def __len__(self):
        return 60

    def __call__(self, text, add_special_tokens=True, truncation=True, max_length=None):
        ids = [ord(c) % 40 + 3 for c in text]
        if max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


class _StubTeacher(torch.nn.Module):
    """Deterministic logits keyed by the input ids, so any worker computing the
    same sequence produces the same Top-K (lets us compare across shardings)."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self._p = torch.nn.Linear(1, 1)  # gives .parameters() a device

    def forward(self, ids):
        b, s = ids.shape
        torch.manual_seed(int(ids.sum().item()) % 100000)
        return SimpleNamespace(logits=torch.randn(b, s, self.vocab))


# Rows chosen to pack into several sequences at a small max_seq.
_ROWS = [
    "alpha beta gamma delta",
    "epsilon zeta",
    "eta theta iota kappa lambda mu",
    "nu xi",
    "omicron pi rho sigma tau",
    "upsilon phi chi psi omega",
    "aa bb cc dd ee ff gg",
    "hh ii jj",
]


def _write_dataset(tmp_path: Path) -> Path:
    jsonl = tmp_path / "stage1.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in _ROWS:
            f.write(json.dumps({"text": r}) + "\n")
    return jsonl


def _read_merged(logits_dir: Path, stage_name: str = "stage1") -> list[tuple[int, str]]:
    """All shard items flattened in reader order (sorted by integer part index).

    Returns [(seq_index, identity_json), ...] — the exact stream the train reader
    consumes. identity is canonicalised to JSON so it is order-insensitive/comparable.
    """
    from orchestrator.distillation_manager import _part_index

    shards = sorted(logits_dir.glob(f"{stage_name}_train_part_*.pt"),
                    key=lambda p: _part_index(p.name))
    out: list[tuple[int, str]] = []
    for shard in shards:
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        for item in payload["logits"]:
            ident = json.dumps(item["identity"], sort_keys=True)
            out.append((int(item["seq_index"]), ident))
    return out


def _run_all_workers(num_shards, teacher, tok, logits_dir, *, chunk_size, batch_size, max_seq):
    for shard_id in range(num_shards):
        P._precompute_stage_sharded(
            stage_num=1, teacher=teacher, tokenizer=tok, logits_dir=logits_dir,
            top_k=4, chunk_size=chunk_size, batch_size=batch_size, max_seq=max_seq,
            num_shards=num_shards, shard_id=shard_id,
        )


@pytest.fixture
def _wired(tmp_path, monkeypatch):
    jsonl = _write_dataset(tmp_path)
    monkeypatch.setitem(P.STAGE_FILES, 1, jsonl)
    monkeypatch.setitem(VLA.STAGE_FILES, 1, jsonl)
    tok = _StubTokenizer()
    teacher = _StubTeacher(vocab=len(tok))
    return SimpleNamespace(jsonl=jsonl, tok=tok, teacher=teacher, tmp=tmp_path)


def test_single_process_baseline_matches_one_shard(_wired):
    """num_shards=1 sharded path yields the same global stream as precompute_stage."""
    base_dir = _wired.tmp / "base"
    base_dir.mkdir()
    P.precompute_stage(stage_num=1, teacher=_wired.teacher, tokenizer=_wired.tok,
                       logits_dir=base_dir, top_k=4, chunk_size=2, batch_size=2, max_seq=8)
    baseline = _read_merged(base_dir)
    assert baseline, "baseline produced nothing"

    one_dir = _wired.tmp / "one"
    one_dir.mkdir()
    _run_all_workers(1, _wired.teacher, _wired.tok, one_dir, chunk_size=2, batch_size=2, max_seq=8)
    sharded = _read_merged(one_dir)

    # Same sequence order and same per-sequence identities.
    assert [s for s, _ in sharded] == [s for s, _ in baseline]
    assert [i for _, i in sharded] == [i for _, i in baseline]


@pytest.mark.parametrize("num_shards", [2, 3, 4, 5])
def test_block_cyclic_tiles_and_preserves_global_order(_wired, num_shards):
    """N workers tile [0,S) with no gap/overlap; merged read == global seq order
    with identities identical to the single-process baseline."""
    base_dir = _wired.tmp / "base"
    base_dir.mkdir()
    P.precompute_stage(stage_num=1, teacher=_wired.teacher, tokenizer=_wired.tok,
                       logits_dir=base_dir, top_k=4, chunk_size=2, batch_size=2, max_seq=8)
    baseline = _read_merged(base_dir)
    S = len(baseline)
    assert S >= num_shards  # enough sequences to exercise every worker meaningfully

    par_dir = _wired.tmp / f"par{num_shards}"
    par_dir.mkdir()
    _run_all_workers(num_shards, _wired.teacher, _wired.tok, par_dir,
                     chunk_size=2, batch_size=2, max_seq=8)
    merged = _read_merged(par_dir)

    seq_indices = [s for s, _ in merged]
    # exact tiling: every seq_index 0..S-1 exactly once
    assert seq_indices == list(range(S)), seq_indices
    # global order preserved + identity matches baseline byte-for-byte
    assert [i for _, i in merged] == [i for _, i in baseline]


def test_finalize_passes_alignment_and_writes_canonical_state(_wired, monkeypatch):
    """End-to-end: run 4 workers -> finalize verifies coverage, writes canonical
    resume state, and the real alignment validator PASSES."""
    par_dir = _wired.tmp / "par"
    par_dir.mkdir()
    _run_all_workers(4, _wired.teacher, _wired.tok, par_dir, chunk_size=2, batch_size=2, max_seq=8)

    res = O.finalize_stage(par_dir, 1, chunk_size=2, validate=True, tokenizer=_wired.tok)
    assert res["status"] == "PASS", res
    assert res.get("alignment") == "PASS", res

    # canonical resume state present + marks the stage complete for downstream gates.
    state = par_dir / "stage1_train_topk_state.json"
    assert state.exists()
    data = json.loads(state.read_text())
    assert data["lines_consumed"] == P._count_jsonl(_wired.jsonl)
    assert P._stage_complete(par_dir, "stage1", P._count_jsonl(_wired.jsonl))


def test_coverage_gap_is_detected(_wired):
    """A missing block shard must fail coverage (no silent gap)."""
    par_dir = _wired.tmp / "par"
    par_dir.mkdir()
    _run_all_workers(3, _wired.teacher, _wired.tok, par_dir, chunk_size=2, batch_size=2, max_seq=8)

    states = O._collect_worker_states(par_dir, "stage1")
    S = O.aggregate_total_sequences(states)
    ok, missing = O.verify_stage_coverage(par_dir, "stage1", S, chunk_size=2)
    assert ok and not missing, (ok, missing)

    # Delete one block's shard -> coverage must now fail, and finalize reports it.
    victim = sorted(par_dir.glob("stage1_train_part_*.pt"))[1]
    victim.unlink()
    ok2, missing2 = O.verify_stage_coverage(par_dir, "stage1", S, chunk_size=2)
    assert not ok2 and missing2
    res = O.finalize_stage(par_dir, 1, chunk_size=2, validate=False)
    assert res["status"] == "COVERAGE_FAIL", res


def test_resume_skips_existing_blocks(_wired):
    """A re-run over an already-complete shard set recomputes nothing (idempotent)."""
    par_dir = _wired.tmp / "par"
    par_dir.mkdir()
    _run_all_workers(2, _wired.teacher, _wired.tok, par_dir, chunk_size=2, batch_size=2, max_seq=8)
    first = _read_merged(par_dir)
    mtimes = {p.name: p.stat().st_mtime_ns for p in par_dir.glob("stage1_train_part_*.pt")}

    # Re-run identical workers: every block's shard already exists -> skipped.
    _run_all_workers(2, _wired.teacher, _wired.tok, par_dir, chunk_size=2, batch_size=2, max_seq=8)
    second = _read_merged(par_dir)
    mtimes2 = {p.name: p.stat().st_mtime_ns for p in par_dir.glob("stage1_train_part_*.pt")}

    assert first == second
    assert mtimes == mtimes2, "resume must not rewrite existing block shards"


def test_stage_shards_sorted_by_integer_part_index(tmp_path):
    """Regression (review HIGH): _stage_shards must sort by INTEGER part index, not
    lexicographically. Lexicographic order misplaces part_10 before part_2 (>=11
    single-process shards) and part_10000 before part_2000 (>=6 parallel blocks),
    which made the alignment validator false-FAIL a byte-correct shard set."""
    # single-process style indices 0..12 (crosses the 10<2 lexicographic boundary)
    d1 = tmp_path / "single"
    d1.mkdir()
    for i in range(13):
        (d1 / f"stage1_train_part_{i}.pt").write_bytes(b"x")
    got1 = [P._shard_part_index(p) for p in P._stage_shards(d1, "stage1")]
    assert got1 == list(range(13)), got1

    # parallel style first-seq-index indices 0,2000,...,14000 (8 blocks)
    d2 = tmp_path / "parallel"
    d2.mkdir()
    for b in range(8):
        (d2 / f"stage1_train_part_{b * 2000}.pt").write_bytes(b"x")
    got2 = [P._shard_part_index(p) for p in P._stage_shards(d2, "stage1")]
    assert got2 == [b * 2000 for b in range(8)], got2


def test_validator_passes_across_digit_boundary(tmp_path, monkeypatch):
    """End-to-end regression: a parallel run with >=11 blocks (part indices crossing
    the lexicographic 10<2 boundary) must still validate PASS. Before the integer-sort
    fix the validator read identities out of order and false-FAILed here."""
    tok = _StubTokenizer()
    teacher = _StubTeacher(vocab=len(tok))
    # 16 distinct rows -> chunk_size=1 gives >=16 single-sequence blocks
    # (part_0..part_15), crossing the lexicographic part_10 < part_2 boundary.
    jsonl = tmp_path / "stage1.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for i in range(16):
            f.write(json.dumps({"text": f"row number {i} alpha beta"}) + "\n")
    monkeypatch.setitem(P.STAGE_FILES, 1, jsonl)
    import scripts.validate_logit_alignment as _VLA
    monkeypatch.setitem(_VLA.STAGE_FILES, 1, jsonl)

    par_dir = tmp_path / "boundary"
    par_dir.mkdir()
    _run_all_workers(2, teacher, tok, par_dir, chunk_size=1, batch_size=1, max_seq=8)
    n_blocks = len(sorted(par_dir.glob("stage1_train_part_*.pt")))
    assert n_blocks >= 11, f"need >=11 blocks to cross the boundary, got {n_blocks}"
    res = O.finalize_stage(par_dir, 1, chunk_size=1, validate=True, tokenizer=tok)
    assert res["status"] == "PASS", res
    assert res.get("alignment") == "PASS", res


def test_run_stage_honors_preset_interrupt(_wired):
    """Regression (review HIGH): a SIGINT must short-circuit _run_stage's retry loop
    instead of re-launching the workers the user just terminated."""
    res = O._run_stage(
        1, [0], top_k=4, chunk_size=2, batch_size=2, max_seq=8,
        logits_dir=_wired.tmp, model_id="m", max_retries=1, validate=False,
        log_dir=_wired.tmp, interrupted={"flag": True},
    )
    assert res["status"] == "INTERRUPTED", res


def test_orchestrator_pure_helpers():
    assert O.resolve_worker_gpus(None, "0,1,3", 8) == [0, 1, 3]
    assert O.resolve_worker_gpus(4, None, 8) == [0, 1, 2, 3]
    assert O.resolve_worker_gpus(None, None, 2) == [0, 1]
    assert O.aggregate_total_sequences([{"total_sequences": 7}, {"total_sequences": 0}]) == 7
    cmd = O.build_worker_command(2, 1, 4, top_k=256, chunk_size=2000, batch_size=4,
                                 max_seq=4096, logits_dir=Path("/tmp/l"), model_id="m", python_exe="py")
    assert cmd[:5] == ["py", str(O.WORKER_SCRIPT), "--stage", "2", "--num-shards"]
    assert "--shard-id" in cmd
    i = cmd.index("--shard-id")
    assert cmd[i + 1] == "1"
