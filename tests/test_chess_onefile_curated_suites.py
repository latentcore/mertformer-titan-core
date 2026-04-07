from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.chess_5080_onefile as onefile


def test_curated_position_bank_has_required_suites_and_legal_moves() -> None:
    bank = onefile.materialize_curated_position_bank()
    suites = {item['suite'] for item in bank}
    assert {'opening', 'tactical', 'endgame', 'blunder_correction'} <= suites
    assert bank
    for item in bank:
        move = onefile.chess.Move.from_uci(item['expected_move_uci'])
        assert move in item['board'].legal_moves


def test_curated_training_examples_repeat_factor_applies() -> None:
    examples, manifest = onefile.build_curated_training_examples({**onefile.RUN_CONFIG, 'curated_position_repeat': 3})
    assert manifest['enabled'] is True
    assert manifest['repeat_factor'] == 3
    assert manifest['examples_total'] == manifest['positions_total'] * 3
    assert examples


def test_synthetic_teaching_corpus_contains_all_levels() -> None:
    report = onefile.build_synthetic_teaching_corpus(onefile.RUN_CONFIG)
    assert report['enabled'] is True
    assert report['record_count'] > 0
    assert report['level_counts']['advanced'] > 0
    assert report['level_counts']['basic'] > 0
    assert report['level_counts']['club'] > 0


class _DummyModel(torch.nn.Module):
    def __init__(self, preferred_move_id: int, vocab_size: int):
        super().__init__()
        self.preferred_move_id = preferred_move_id
        self.vocab_size = vocab_size

    def forward(self, piece_ids: torch.Tensor, meta_ids: torch.Tensor):
        logits = torch.full((piece_ids.size(0), self.vocab_size), -5.0, dtype=torch.float32, device=piece_ids.device)
        logits[:, self.preferred_move_id] = 5.0
        value = torch.full((piece_ids.size(0),), 0.4, dtype=torch.float32, device=piece_ids.device)
        aux = torch.zeros((), dtype=torch.float32, device=piece_ids.device)
        return logits, value, aux, {}


def test_curated_position_suite_eval_reports_hits() -> None:
    bank = onefile.materialize_curated_position_bank()
    expected_move_id = onefile.MOVE_TO_ID[bank[0]['expected_move_uci']]
    model = _DummyModel(expected_move_id, len(onefile.MOVE_VOCAB))
    report = onefile.evaluate_curated_position_suites(model, onefile.RUN_CONFIG, torch.device('cpu'))
    assert report['status'] == 'completed'
    assert report['position_count'] == len(bank)
    assert 0.0 <= report['top3_hit_rate'] <= 1.0
    assert 0.0 <= report['teaching_length_monotonic_rate'] <= 1.0
