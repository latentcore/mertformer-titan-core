from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

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
        'self_delete_target': None,
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


def test_embedded_seed_games_and_example_builder(tmp_path: Path) -> None:
    logger = onefile.JSONLLogger(tmp_path / 'pytest_chess_5080_log.jsonl')
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


def test_validate_runtime_config_requires_explicit_self_delete_target() -> None:
    with pytest.raises(onefile.ConfigValidationError, match='self_delete_target'):
        onefile.resolve_runtime_config(
            make_args(share_mode=True, enable_self_delete=True),
            onefile.RUN_CONFIG,
        )


def test_schedule_self_delete_ignores_canonical_repo_script(monkeypatch) -> None:
    popen_calls: list[object] = []

    def fake_popen(*args: object, **kwargs: object) -> object:
        popen_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(onefile.subprocess, 'Popen', fake_popen)
    onefile.schedule_self_delete_if_needed(
        {
            **onefile.RUN_CONFIG,
            'share_mode': True,
            'enable_self_delete': True,
            'self_delete_target': str(Path(onefile.__file__).resolve()),
        },
        True,
        None,
    )
    assert popen_calls == []


def test_evaluate_model_uses_single_forward_per_batch_and_example_weighting() -> None:
    move_a = onefile.MOVE_TO_ID['e2e4']
    move_b = onefile.MOVE_TO_ID['d2d4']

    class CountingModel(onefile.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_calls = 0

        def forward(self, piece_ids, meta_ids):
            del meta_ids
            self.forward_calls += 1
            batch = piece_ids.size(0)
            logits = onefile.torch.full((batch, len(onefile.MOVE_VOCAB)), -5.0)
            for idx in range(batch):
                predicted = move_a if int(piece_ids[idx, 0].item()) == 0 else move_b
                logits[idx, predicted] = 5.0
            value = onefile.torch.zeros(batch, dtype=onefile.torch.float32)
            aux = onefile.torch.tensor(0.0, dtype=onefile.torch.float32)
            return logits, value, aux, {}

    def make_batch(marker_values: list[int], targets: list[int]) -> dict[str, object]:
        batch = len(marker_values)
        legal_mask = onefile.torch.zeros((batch, len(onefile.MOVE_VOCAB)), dtype=onefile.torch.bool)
        legal_mask[:, move_a] = True
        legal_mask[:, move_b] = True
        return {
            'piece_ids': onefile.torch.tensor([[value] + [0] * 63 for value in marker_values], dtype=onefile.torch.long),
            'meta_ids': onefile.torch.zeros((batch, 8), dtype=onefile.torch.long),
            'legal_mask': legal_mask,
            'move_targets': onefile.torch.tensor(targets, dtype=onefile.torch.long),
            'value_targets': onefile.torch.zeros(batch, dtype=onefile.torch.float32),
            'phases': onefile.torch.zeros(batch, dtype=onefile.torch.long),
        }

    model = CountingModel()
    cfg = {**onefile.RUN_CONFIG, 'seed': 7}
    evaluation = onefile.evaluate_model(
        model,
        [
            make_batch([0, 0], [move_a, move_a]),
            make_batch([1], [move_a]),
        ],
        onefile.torch.device('cpu'),
        cfg,
    )
    assert model.forward_calls == 2
    assert evaluation['metrics']['masked_policy_accuracy'] == pytest.approx(2 / 3, abs=1e-6)
    assert evaluation['per_phase']['opening']['masked_policy_accuracy'] == pytest.approx(2 / 3, abs=1e-6)


def test_run_pipeline_verify_mode_skips_strength_surfaces(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(onefile, 'detect_desktop_dir', lambda: tmp_path)

    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError('verify mode must skip strength surfaces')

    monkeypatch.setattr(onefile, 'evaluate_model', fail)
    monkeypatch.setattr(onefile, 'run_legality_report', fail)
    monkeypatch.setattr(onefile, 'generate_demo_replay', fail)
    monkeypatch.setattr(onefile, 'play_stockfish_gauntlet', fail)

    cfg = onefile.apply_profile(onefile.RUN_CONFIG, 'smoke')
    cfg['mode'] = 'verify'
    cfg['artifact_root'] = str(tmp_path / 'artifacts')
    cfg['cache_root'] = str(tmp_path / 'cache')
    cfg['offline_seed_only'] = True
    cfg['auto_download_enabled'] = False
    cfg['device'] = 'cpu'
    onefile.validate_runtime_config(cfg)

    layout = onefile.prepare_layout(cfg)
    logger = onefile.JSONLLogger(tmp_path / 'verify_log.jsonl')
    payload = onefile.run_pipeline(cfg, layout=layout, logger=logger)

    assert payload['evaluation_status'] == onefile.EvaluationStatus.UNEVALUATED.value
    assert payload['holdout_validation']['status'] == 'not_run'
    assert payload['legality_report']['status'] == 'not_run'
    assert payload['stockfish']['reason'] == 'verify_mode_runtime_only'
