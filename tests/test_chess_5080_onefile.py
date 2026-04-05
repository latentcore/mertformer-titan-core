from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.chess_5080_onefile as onefile


def make_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        'mode': onefile.RUN_CONFIG['mode'],
        'profile': onefile.RUN_CONFIG['profile'],
        'baseline': onefile.RUN_CONFIG['baseline'],
        'resume_from': None,
        'artifact_root': None,
        'stockfish_path': None,
        'no_download': False,
        'allow_install': False,
        'share_mode': False,
        'enable_self_delete': False,
        'offline_seed_only': False,
        'test_mode': False,
        'max_steps': None,
        'max_wall_hours': None,
        'batch_size': None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_move_vocab_contains_common_uci_moves() -> None:
    assert onefile.MOVE_TO_ID['e2e4'] >= 0
    assert onefile.MOVE_TO_ID['g1f3'] >= 0
    assert onefile.MOVE_TO_ID['a7a8q'] >= 0


def test_resolve_runtime_config_verify_mode_uses_embedded_seed() -> None:
    cfg = onefile.resolve_runtime_config(make_args(mode='verify'), onefile.RUN_CONFIG)
    assert cfg['mode'] == 'verify'
    assert cfg['offline_seed_only'] is True
    assert cfg['auto_download_enabled'] is False


def test_deterministic_seed_sets_strict_flags() -> None:
    onefile.deterministic_seed(123, strict=True)
    assert onefile.torch.backends.cudnn.deterministic is True
    assert onefile.torch.backends.cudnn.benchmark is False


def test_embedded_seed_games_and_example_builder() -> None:
    logger = onefile.JSONLLogger(ROOT / 'reports' / '_pytest_chess_5080_log.jsonl')
    examples, stats = onefile.build_examples_from_games(
        [('embedded_seed', onefile.embedded_seed_games())],
        {
            **onefile.RUN_CONFIG,
            'offline_seed_only': True,
            'auto_download_enabled': False,
            'max_games': 3,
            'max_positions': 24,
            'max_positions_per_game': 3,
            'min_elo': 1800,
        },
        logger,
    )
    assert stats['games_accepted'] >= 2
    assert examples
    assert all(example.target_move_id in example.legal_move_ids for example in examples)


def test_prepare_layout_reuses_existing_package_run(tmp_path: Path) -> None:
    run_dir = tmp_path / 'MertFormer_Chess_5080_Delivery_20260405_120000'
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True)
    resume_path = checkpoint_dir / 'latest.pt'
    resume_path.write_bytes(b'checkpoint')
    cfg = {
        **onefile.RUN_CONFIG,
        'mode': 'package',
        'resume_from': str(resume_path),
        'artifact_root': str(tmp_path / 'other_root'),
    }
    layout = onefile.prepare_layout(cfg)
    assert layout.run_dir == run_dir
    assert layout.checkpoints_dir == checkpoint_dir


def test_create_result_bundle_single_output_mode_omits_sha(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(onefile, 'detect_desktop_dir', lambda: tmp_path)
    cfg = {
        **onefile.RUN_CONFIG,
        'artifact_root': str(tmp_path / 'artifacts'),
        'single_output_only': True,
        'zip_outputs': True,
    }
    layout = onefile.make_layout(cfg)
    report_path = layout.reports_dir / 'demo.json'
    report_path.write_text(json.dumps({'ok': True}), encoding='utf-8')
    bundle = onefile.create_result_bundle(layout, {'config': cfg})
    assert bundle['zip_path']
    assert bundle['sha256_path'] == ''
    assert not layout.final_sha_path.exists()
