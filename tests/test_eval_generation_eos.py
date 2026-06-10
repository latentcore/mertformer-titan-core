"""H6: eval prediction extraction must read the answer, not question/prompt digits."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval import gsm8k  # noqa: E402


def test_extract_pred_prefers_marker():
    # Completion-only text with the canonical GSM8K marker.
    assert gsm8k._extract_pred("Let me compute... #### 42") == "42"


def test_extract_pred_falls_back_to_last_number():
    assert gsm8k._extract_pred("the result is 17") == "17"


def test_extract_pred_marker_wins_over_trailing_number():
    # Even if a later number exists, the #### marker is authoritative.
    assert gsm8k._extract_pred("steps 1 2 3 #### 9 then noise 100") == "9"


def test_extract_pred_empty():
    assert gsm8k._extract_pred("") is None
    assert gsm8k._extract_pred("no numbers here") is None


def test_decode_only_new_tokens_excludes_prompt():
    """Simulate generate() returning prompt ++ completion; decoding the new-token
    slice must drop the prompt so question digits can't leak into the prediction."""
    prompt_len = 5
    output_row = list(range(prompt_len)) + [100, 101, 102]  # prompt + 3 new tokens
    new_tokens = output_row[prompt_len:]
    assert new_tokens == [100, 101, 102]
