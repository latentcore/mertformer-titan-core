from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.chess_5080_onefile as onefile


def test_move_vocab_contains_common_uci_moves() -> None:
    assert onefile.MOVE_TO_ID["e2e4"] >= 0
    assert onefile.MOVE_TO_ID["g1f3"] >= 0
    assert onefile.MOVE_TO_ID["a7a8q"] >= 0


def test_legal_move_ids_start_position_is_non_empty() -> None:
    board = onefile.chess.Board()
    ids = onefile.legal_move_ids(board)
    assert ids
    assert onefile.MOVE_TO_ID["e2e4"] in ids
    assert onefile.MOVE_TO_ID["g1f3"] in ids


def test_resolve_runtime_config_test_mode_caps_runtime(monkeypatch) -> None:
    monkeypatch.setenv("MERTFORMER_CHESS_TEST_MODE", "1")
    cfg = onefile.resolve_runtime_config(dict(onefile.RUN_CONFIG))
    assert cfg["offline_seed_only"] is True
    assert cfg["auto_download_enabled"] is False
    assert cfg["max_steps"] == 6
    assert cfg["batch_size"] <= 8


def test_embedded_seed_games_and_example_builder() -> None:
    logger = onefile.JSONLLogger(ROOT / "reports" / "_pytest_chess_5080_log.jsonl")
    examples, stats = onefile.build_examples_from_games(onefile.embedded_seed_games(), {
        **onefile.RUN_CONFIG,
        "offline_seed_only": True,
        "auto_download_enabled": False,
        "max_games": 3,
        "max_positions": 24,
        "max_positions_per_game": 3,
        "min_elo": 1800,
    }, logger)
    assert stats["games_kept"] >= 2
    assert examples
    assert all(example.target_move_id in example.legal_move_ids for example in examples)


def test_export_bundle_layout_uses_desktop(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(onefile, "detect_desktop_dir", lambda: tmp_path)
    cfg = onefile.resolve_runtime_config({**onefile.RUN_CONFIG, "artifact_root": str(tmp_path / "artifacts")})
    layout = onefile.make_layout(cfg)
    assert layout.final_zip_path.parent == tmp_path
    assert layout.run_dir.exists()
