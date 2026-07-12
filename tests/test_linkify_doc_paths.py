from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.linkify_doc_paths as linkify


def test_linkifies_a_bare_tracked_path_mention() -> None:
    text = "See scripts/train_smoke.py for the entry point."
    new_text, count = linkify.linkify_text(text, tracked={"scripts/train_smoke.py"})
    assert count == 1
    assert "[scripts/train_smoke.py](scripts/train_smoke.py)" in new_text


def test_skips_paths_already_inside_a_markdown_link() -> None:
    text = "See [scripts/train_smoke.py](scripts/train_smoke.py) for the entry point."
    new_text, count = linkify.linkify_text(text, tracked={"scripts/train_smoke.py"})
    assert count == 0
    assert new_text == text


def test_skips_paths_inside_inline_code_spans() -> None:
    text = "Run `scripts/train_smoke.py` directly."
    new_text, count = linkify.linkify_text(text, tracked={"scripts/train_smoke.py"})
    assert count == 0
    assert new_text == text


def test_skips_paths_inside_fenced_code_blocks() -> None:
    text = "```bash\npython3 scripts/train_smoke.py\n```"
    new_text, count = linkify.linkify_text(text, tracked={"scripts/train_smoke.py"})
    assert count == 0
    assert new_text == text


def test_skips_untracked_paths() -> None:
    text = "See scripts/does_not_exist.py for details."
    new_text, count = linkify.linkify_text(text, tracked={"scripts/train_smoke.py"})
    assert count == 0
    assert new_text == text


def test_is_idempotent_on_a_second_pass() -> None:
    text = "See scripts/train_smoke.py for the entry point."
    first, count1 = linkify.linkify_text(text, tracked={"scripts/train_smoke.py"})
    second, count2 = linkify.linkify_text(first, tracked={"scripts/train_smoke.py"})
    assert count1 == 1
    assert count2 == 0
    assert first == second
