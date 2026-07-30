"""Tests for train/trainer_data.py::CurriculumDataset — the streaming curriculum loader.

This class had no coverage before [2026-07-29] despite sitting on the real training path.
Two things are pinned here:

* **Y-8** — the per-file line-offset index is ``array.array("q")``, not ``list[int]``.
  Measured 34.8 MiB per 1M offsets as a list against 7.6 MiB as an array (4.56x), and
  because ``CurriculumDataset`` is an ``IterableDataset`` the whole object is copied into
  every DataLoader worker, so the multiplier compounds with ``num_workers``. A future
  refactor that turns the index back into a list would be invisible in behaviour, hence
  the explicit type assertion.
* the pre-existing ``[MED]`` byte-seek fix — the index holds one entry per NON-EMPTY line
  and sampling picks a uniform LINE index. The original code seeked to a random BYTE
  offset, which over-samples long lines (a longer line occupies more byte space) and can
  never select the first line. Both properties are asserted directly.

Uses a stub tokenizer and temp files: no HF, no network, no GPU.
"""
from __future__ import annotations

import array
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

from train.trainer_data import CurriculumDataset  # noqa: E402


class _StubTokenizer:
    """Deterministic char-based tokenizer (no HF, no network)."""
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=True, truncation=True, max_length=None):
        ids = [ord(c) % 50 + 3 for c in text]
        if max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def _write_stages(tmp_path: Path, n_lines: int = 40, blank_every: int = 0):
    """Four stage files; optionally interleave blank lines that must be skipped."""
    paths = []
    for stage in range(4):
        path = tmp_path / f"stage{stage + 1}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for i in range(n_lines):
                handle.write(json.dumps({"text": f"stage{stage} line{i} content here"}) + "\n")
                if blank_every and i % blank_every == 0:
                    handle.write("\n")
        paths.append(path)
    return paths


def test_offset_index_is_a_64bit_array_not_a_list(tmp_path):
    """Y-8: the memory-sensitive index must stay array('q')."""
    paths = _write_stages(tmp_path)
    dataset = CurriculumDataset(paths, max_len=32, tokenizer=_StubTokenizer())

    assert dataset.line_offsets, "no offsets were indexed"
    for key, offsets in dataset.line_offsets.items():
        assert isinstance(offsets, array.array), (
            f"{key}: offset index regressed to {type(offsets).__name__}; a list of ints "
            f"costs ~4.5x the memory and is copied into every DataLoader worker"
        )
        # "q" is signed 64-bit: byte offsets past 2 GiB must not overflow.
        assert offsets.typecode == "q", f"{key}: typecode {offsets.typecode!r}, want 'q'"
        assert offsets.itemsize == 8


def test_blank_lines_are_excluded_from_the_index(tmp_path):
    """[MED] one entry per non-empty line -- a sampled index can never hit a blank."""
    paths = _write_stages(tmp_path, n_lines=40, blank_every=5)
    dataset = CurriculumDataset(paths, max_len=32, tokenizer=_StubTokenizer())

    for path in paths:
        offsets = dataset.line_offsets[str(path)]
        assert len(offsets) == 40, f"{path.name}: indexed {len(offsets)}, want 40 non-empty"
        with open(path, "rb") as handle:
            for idx in range(len(offsets)):
                handle.seek(offsets[idx])
                line = handle.readline()
                assert line.strip(), f"offset {idx} landed on a blank line"
                json.loads(line)  # must be a parseable record, not a mid-line seek


def test_first_line_is_reachable(tmp_path):
    """The byte-offset sampler could never select line 0; the line-index sampler must."""
    paths = _write_stages(tmp_path)
    dataset = CurriculumDataset(paths, max_len=32, tokenizer=_StubTokenizer())
    offsets = dataset.line_offsets[str(paths[0])]
    assert offsets[0] == 0, f"first indexed offset is {offsets[0]}, want 0"


def test_index_is_uniform_over_lines_not_biased_by_length(tmp_path):
    """One index slot per line regardless of its length -> no length over-sampling.

    Written with deliberately lopsided line lengths: byte-offset sampling would give the
    long lines proportionally more of the offset space, while line-index sampling gives
    every line exactly one slot.
    """
    path = tmp_path / "stage1.jsonl"
    lengths = [5, 500, 5, 500, 5]
    with open(path, "w", encoding="utf-8") as handle:
        for i, length in enumerate(lengths):
            handle.write(json.dumps({"text": f"{i}" + "x" * length}) + "\n")

    dataset = CurriculumDataset([path], max_len=32, tokenizer=_StubTokenizer())
    offsets = dataset.line_offsets[str(path)]
    assert len(offsets) == len(lengths), "one slot per line, independent of line length"

    # Gaps between consecutive offsets mirror the lopsided lengths; the point is that the
    # SAMPLER does not see those gaps -- it draws from range(len(offsets)).
    gaps = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
    assert max(gaps) > 10 * min(gaps), "test data is not actually lopsided"


def test_missing_stage_files_are_skipped_not_fatal(tmp_path):
    """Absent stage paths must simply not be indexed."""
    paths = _write_stages(tmp_path)
    paths.append(tmp_path / "stage5_does_not_exist.jsonl")
    dataset = CurriculumDataset(paths, max_len=32, tokenizer=_StubTokenizer())

    assert str(paths[-1]) not in dataset.line_offsets
    assert len(dataset.line_offsets) == 4


@pytest.mark.parametrize("stage", [1, 2, 3, 4])
def test_every_curriculum_stage_yields_padded_pairs(tmp_path, stage):
    """All four stages must produce (input_ids, labels) at exactly max_len."""
    paths = _write_stages(tmp_path)
    max_len = 32
    dataset = CurriculumDataset(paths, max_len=max_len, tokenizer=_StubTokenizer(),
                                current_stage=stage)
    iterator = iter(dataset)
    for _ in range(8):
        sample = next(iterator)
        assert isinstance(sample, tuple) and len(sample) == 2
        input_ids, labels = sample
        assert input_ids.shape == (max_len,), input_ids.shape
        assert labels.shape == (max_len,), labels.shape


def test_set_stage_switches_the_active_mixture(tmp_path):
    paths = _write_stages(tmp_path)
    dataset = CurriculumDataset(paths, max_len=32, tokenizer=_StubTokenizer(), current_stage=1)
    assert dataset.current_stage == 1
    dataset.set_stage(3)
    assert dataset.current_stage == 3
    # Still iterable after the switch (stage 3 mixes several files).
    assert next(iter(dataset))[0].shape == (32,)
