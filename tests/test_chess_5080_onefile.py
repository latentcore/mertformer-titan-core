from __future__ import annotations

import argparse
import builtins
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import cfg as canonical_cfg
from layers.bitlinear import BitLinear as CanonicalBitLinear
from layers.lifelong_safety import LifelongSafetyLayer as CanonicalLifelongSafetyLayer
from layers.liquid import LiquidMixer as CanonicalLiquidMixer
from layers.mla import MLA as CanonicalMLA
from layers.moe import MoE as CanonicalMoE
from layers.qinn import UnitaryQINN as CanonicalUnitaryQINN
from layers.world_model_head import CausalWorldModelHead as CanonicalWorldModelHead
import scripts.chess_5080_onefile as onefile


def make_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        'mode': onefile.RUN_CONFIG['mode'],
        'profile': onefile.RUN_CONFIG['profile'],
        'baseline': onefile.RUN_CONFIG['baseline'],
        'feature_bundle': None,
        'enable_features': None,
        'disable_features': None,
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


@contextmanager
def patched_canonical_cfg(**updates: object):
    missing = object()
    originals = {name: getattr(canonical_cfg, name, missing) for name in updates}
    try:
        for name, value in updates.items():
            setattr(canonical_cfg, name, value)
        yield
    finally:
        for name, value in originals.items():
            if value is missing:
                delattr(canonical_cfg, name)
            else:
                setattr(canonical_cfg, name, value)


def make_mirror_cfg(**overrides: object) -> dict[str, object]:
    cfg = {
        **onefile.RUN_CONFIG,
        'device': 'cpu',
        'hidden_size': 32,
        'intermediate_size': 64,
        'num_layers': 4,
        'num_heads': 4,
        'num_kv_heads': 2,
        'head_dim': 8,
        'max_seq_len': 32,
        'dropout': 0.0,
        'attention_dropout': 0.0,
        'ffn_dropout': 0.0,
        'use_bitlinear': True,
        'use_moe': True,
        'moe_top_k': 2,
        'num_experts': 4,
        'moe_every_n_layers': 2,
        'moe_intermediate': 64,
        'use_liquid': True,
        'use_liquid_adapter': True,
        'liquid_layers_idx': [1, 3],
        'liquid_fast_path': False,
        'use_qinn': True,
        'use_hierarchical_kv_cache': False,
        'use_global_workspace_broadcast': True,
        'use_neuromodulatory_gain': True,
        'use_latent_ode_state_channel': True,
        'use_world_model_head': True,
        'use_lifelong_safety_layer': True,
        'use_hebbian_plasticity': True,
        'use_neuro_symbolic_layer': True,
        'use_cross_expert_sync_bus': True,
        'use_structural_plasticity': True,
    }
    cfg.update(overrides)
    return cfg


def test_move_vocab_contains_common_uci_moves() -> None:
    assert onefile.MOVE_TO_ID['e2e4'] >= 0
    assert onefile.MOVE_TO_ID['g1f3'] >= 0
    assert onefile.MOVE_TO_ID['a7a8q'] >= 0


def test_choose_move_trace_emits_structured_response_contract() -> None:
    class StubModel:
        def __call__(self, piece: onefile.torch.Tensor, meta: onefile.torch.Tensor):
            logits = onefile.torch.full((1, len(onefile.MOVE_VOCAB)), -10.0, dtype=onefile.torch.float32)
            logits[0, onefile.MOVE_TO_ID['e2e4']] = 4.0
            logits[0, onefile.MOVE_TO_ID['d2d4']] = 2.0
            value = onefile.torch.tensor([0.34], dtype=onefile.torch.float32)
            return logits, value, onefile.torch.tensor(0.0), {}

    trace = onefile.choose_move_trace(
        StubModel(),
        onefile.chess.Board(),
        onefile.torch.device('cpu'),
        mode='teach',
        teaching_level='advanced',
    )
    contract = trace['response_contract']
    assert trace['move'] == 'e2e4'
    assert contract['best_move'] == 'e2e4'
    assert contract['principal_variation'] == ['e2e4']
    assert contract['mode'] == 'teach'
    assert contract['teaching_level'] == 'advanced'
    assert 'center_control' in contract['teaching_tags']
    assert contract['confidence']['tier'] in {'medium', 'high'}
    assert trace['masked_topk_scores'][0] >= trace['masked_topk_scores'][1]


def test_choose_move_trace_surfaces_auxiliary_predictions() -> None:
    class StubModel:
        def __init__(self) -> None:
            self._aux = {
                'phase_logits': onefile.torch.tensor([[3.0, 0.2, -0.4]], dtype=onefile.torch.float32),
                'wdl_logits': onefile.torch.tensor([[-0.1, 0.3, 1.2]], dtype=onefile.torch.float32),
                'legality_logits': onefile.torch.full((1, len(onefile.MOVE_VOCAB)), -3.0, dtype=onefile.torch.float32),
            }
            self._aux['legality_logits'][0, onefile.MOVE_TO_ID['e2e4']] = 2.0
            self._aux['legality_logits'][0, onefile.MOVE_TO_ID['d2d4']] = 1.0

        def __call__(self, piece: onefile.torch.Tensor, meta: onefile.torch.Tensor):
            logits = onefile.torch.full((1, len(onefile.MOVE_VOCAB)), -10.0, dtype=onefile.torch.float32)
            logits[0, onefile.MOVE_TO_ID['e2e4']] = 4.0
            logits[0, onefile.MOVE_TO_ID['d2d4']] = 2.0
            value = onefile.torch.tensor([0.34], dtype=onefile.torch.float32)
            return logits, value, onefile.torch.tensor(0.0), {}

        def get_last_auxiliary_outputs(self):
            return self._aux

    trace = onefile.choose_move_trace(
        StubModel(),
        onefile.chess.Board(),
        onefile.torch.device('cpu'),
        mode='analyze',
        teaching_level='club',
    )
    aux = trace['auxiliary_predictions']
    assert aux['phase_head']['label'] == 'opening'
    assert aux['wdl_head']['label'] == 'win'
    assert aux['legality_head']['top1_is_legal'] is True
    assert trace['response_contract']['auxiliary_predictions']['phase_head']['label'] == 'opening'


class StubInferenceModel:
    def __init__(self) -> None:
        self.training = False
        self._aux = {}

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self

    def __call__(self, piece: onefile.torch.Tensor, meta: onefile.torch.Tensor):
        logits = onefile.torch.full((1, len(onefile.MOVE_VOCAB)), -10.0, dtype=onefile.torch.float32)
        logits[0, onefile.MOVE_TO_ID['e2e4']] = 4.0
        logits[0, onefile.MOVE_TO_ID['d2d4']] = 2.0
        value = onefile.torch.tensor([0.12], dtype=onefile.torch.float32)
        return logits, value, onefile.torch.tensor(0.0), {}

    def get_last_auxiliary_outputs(self):
        return self._aux


def test_resolve_runtime_config_verify_mode_uses_embedded_seed() -> None:
    cfg = onefile.resolve_runtime_config(make_args(mode='verify'), onefile.RUN_CONFIG)
    assert cfg['mode'] == 'verify'
    assert cfg['offline_seed_only'] is True
    assert cfg['auto_download_enabled'] is False


def test_resolve_runtime_config_applies_all_on_profile_bundle() -> None:
    cfg = onefile.resolve_runtime_config(
        make_args(profile='strength_4060_24h_all_on_experimental'),
        onefile.RUN_CONFIG,
    )
    assert cfg['profile'] == 'strength_4060_24h_all_on_experimental'
    assert cfg['feature_bundle'] == 'all_on_experimental'
    assert cfg['use_moe'] is True
    assert cfg['use_liquid'] is True
    assert cfg['use_qinn'] is True
    assert cfg['use_world_model_head'] is True
    assert cfg['use_gradient_checkpointing'] is True
    assert cfg['selfplay_eval_enabled'] is True
    assert cfg['tournament_eval_enabled'] is True
    assert cfg['replay_buffer_enabled'] is True


def test_resolve_runtime_config_omni_max_profile_enables_auxiliary_heads() -> None:
    cfg = onefile.resolve_runtime_config(
        make_args(profile='strength_4060_24h_omni_max'),
        onefile.RUN_CONFIG,
    )
    assert cfg['profile'] == 'strength_4060_24h_omni_max'
    assert cfg['feature_bundle'] == 'all_on_experimental'
    assert cfg['use_phase_head'] is True
    assert cfg['use_wdl_head'] is True
    assert cfg['use_legality_head'] is True
    assert cfg['enabled_features']


def test_resolve_runtime_config_feature_overrides_win_over_bundle() -> None:
    cfg = onefile.resolve_runtime_config(
        make_args(
            profile='strength_4060_24h_all_on_experimental',
            disable_features='use_qinn,use_world_model_head',
            enable_features='use_bitlinear',
        ),
        onefile.RUN_CONFIG,
    )
    assert cfg['feature_bundle'] == 'all_on_experimental'
    assert cfg['use_qinn'] is False
    assert cfg['use_world_model_head'] is False
    assert cfg['use_bitlinear'] is True
    assert cfg['disabled_features'] == ['use_qinn', 'use_world_model_head']
    assert cfg['enabled_features'] == ['use_bitlinear']


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


def test_make_layout_delivery_mode_uses_internal_runtime_roots(tmp_path: Path) -> None:
    cfg = {
        **onefile.RUN_CONFIG,
        'artifact_root': str(tmp_path / 'runtime'),
        'delivery_mode': True,
    }
    layout = onefile.make_layout(cfg)
    assert layout.run_dir.parent == Path(cfg['artifact_root']) / 'runs'
    assert layout.final_zip_path.parent == Path(cfg['artifact_root']) / 'final'


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


def test_resolve_runtime_config_honors_delivery_env_overrides(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(onefile.DEFAULT_PROFILE_ENV, 'delivery_windows_oneclick')
    monkeypatch.setenv(onefile.DEFAULT_ARTIFACT_ROOT_ENV, str(tmp_path / 'runtime'))
    monkeypatch.setenv(onefile.DEFAULT_CACHE_ROOT_ENV, str(tmp_path / 'runtime_cache'))
    cfg = onefile.resolve_runtime_config(make_args(), onefile.RUN_CONFIG)
    assert cfg['profile'] == 'delivery_windows_oneclick'
    assert cfg['delivery_mode'] is True
    assert Path(cfg['artifact_root']) == tmp_path / 'runtime'
    assert Path(cfg['cache_root']) == tmp_path / 'runtime_cache'


def test_detect_stockfish_path_prefers_cached_runtime_binary(tmp_path: Path) -> None:
    cached_dir = tmp_path / 'stockfish_cache' / 'current'
    cached_dir.mkdir(parents=True)
    binary = cached_dir / 'stockfish.exe'
    binary.write_bytes(b'binary')
    cfg = {
        **onefile.RUN_CONFIG,
        'cache_root': str(tmp_path / 'cache'),
        'stockfish_cache_root': str(tmp_path / 'stockfish_cache'),
        'stockfish_auto_fetch': False,
        'auto_download_enabled': False,
    }
    assert onefile.detect_stockfish_path(cfg) == str(binary)


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


def test_download_archive_slices_enforces_byte_budget_when_range_is_ignored(monkeypatch, tmp_path: Path) -> None:
    payload = b'a' * (2 * 1024 * 1024)

    class FakeResponse:
        status = 200

        def __init__(self, blob: bytes) -> None:
            self._blob = blob
            self._offset = 0

        def info(self) -> dict[str, str]:
            return {'Content-Type': 'application/octet-stream'}

        def read(self, size: int) -> bytes:
            if self._offset >= len(self._blob):
                return b''
            chunk = self._blob[self._offset:self._offset + size]
            self._offset += len(chunk)
            return chunk

        def __enter__(self) -> 'FakeResponse':
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    monkeypatch.setattr(onefile.urllib.request, 'urlopen', lambda req, timeout=60: FakeResponse(payload))
    logger = onefile.JSONLLogger(tmp_path / 'download_log.jsonl')
    cfg = {
        **onefile.RUN_CONFIG,
        'download_partial_mb': 1,
        'download_archive_count': 1,
        'download_retries': 0,
        'download_timeout_sec': 5,
    }
    slices = onefile.download_archive_slices(['https://example.com/demo.pgn.zst'], cfg, logger, tmp_path / 'cache')
    assert len(slices) == 1
    assert slices[0].bytes_written == 1024 * 1024
    assert slices[0].path.stat().st_size == 1024 * 1024


def test_compute_score_rate_ci_uses_non_degenerate_bounds_for_small_samples() -> None:
    ci = onefile.compute_score_rate_ci(0.0, 12)
    assert ci['low'] == 0.0
    assert 0.0 < ci['high'] < 0.3


def test_play_human_vs_model_arena_accepts_human_moves_and_can_abort(monkeypatch) -> None:
    responses = iter(['w', 'e2e4', 'quit'])

    class StubModel:
        def __init__(self) -> None:
            self.training = False

        def eval(self) -> 'StubModel':
            self.training = False
            return self

        def train(self, mode: bool = True) -> 'StubModel':
            self.training = mode
            return self

    monkeypatch.setattr(onefile, 'ensure_interactive_console', lambda: None)
    monkeypatch.setattr(builtins, 'input', lambda prompt='': next(responses))
    monkeypatch.setattr(
        onefile,
        'choose_move_trace',
        lambda model, board, device, cfg=None, **kwargs: {
            'move': 'e7e5',
            'value': 0.12,
            'latency_ms': 1.5,
            'raw_top1_is_legal': True,
            'raw_topk': ['e7e5'],
            'masked_topk': ['e7e5'],
        },
    )
    report = onefile.play_human_vs_model_arena(StubModel(), onefile.RUN_CONFIG, onefile.torch.device('cpu'))
    assert report['status'] == 'aborted_by_user'
    assert report['human_color'] == 'white'
    assert report['plies_played'] == 2
    assert [item['move'] for item in report['transcript']] == ['e2e4', 'e7e5']


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
    assert (layout.reports_dir / 'logging_contract.json').exists()
    assert (layout.reports_dir / 'observability_report.json').exists()


def test_run_pipeline_arena_mode_skips_dataset_and_strength_surfaces(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(onefile, 'detect_desktop_dir', lambda: tmp_path)

    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError('arena mode must skip dataset and strength surfaces')

    monkeypatch.setattr(onefile, 'maybe_collect_dataset', fail)
    monkeypatch.setattr(onefile, 'collect_verify_examples', fail)
    monkeypatch.setattr(onefile, 'evaluate_model', fail)
    monkeypatch.setattr(onefile, 'run_legality_report', fail)
    monkeypatch.setattr(onefile, 'generate_demo_replay', fail)
    monkeypatch.setattr(onefile, 'play_stockfish_gauntlet', fail)
    monkeypatch.setattr(
        onefile,
        'play_human_vs_model_arena',
        lambda *args, **kwargs: {
            'status': 'completed',
            'interactive_only': True,
            'human_color': 'white',
            'result': '1-0',
            'plies_played': 5,
            'final_fen': onefile.chess.Board().fen(),
            'transcript': [],
            'note': 'test arena',
        },
    )

    cfg = onefile.apply_profile(onefile.RUN_CONFIG, 'smoke')
    cfg['mode'] = 'arena'
    cfg['artifact_root'] = str(tmp_path / 'artifacts')
    cfg['cache_root'] = str(tmp_path / 'cache')
    cfg['device'] = 'cpu'
    cfg['zip_outputs'] = False
    onefile.validate_runtime_config(cfg)

    layout = onefile.prepare_layout(cfg)
    logger = onefile.JSONLLogger(tmp_path / 'arena_log.jsonl')
    payload = onefile.run_pipeline(cfg, layout=layout, logger=logger)

    assert payload['evaluation_status'] == onefile.EvaluationStatus.UNEVALUATED.value
    assert payload['holdout_validation']['status'] == 'not_run'
    assert payload['stockfish']['reason'] == 'arena_mode_interactive_only'
    assert payload['arena_session']['status'] == 'completed'


def test_jsonl_logger_writes_structured_schema_and_redacts(tmp_path: Path) -> None:
    log_path = tmp_path / 'run_log.jsonl'
    logger = onefile.JSONLLogger(
        log_path,
        run_id='run-1',
        mode='verify',
        profile='smoke',
        artifact_root='/tmp/artifacts',
    )
    logger.write(
        'test_event',
        {
            'status': 'ok',
            'api_key': 'raw-secret',
            'nested': {'token': 'opaque-token-value'},
            'message': 'sk-abc123',
        },
    )
    logger.finalize('completed')

    rows = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines()]
    assert rows[0]['event'] == 'test_event'
    assert rows[0]['kind'] == 'test_event'
    assert rows[0]['schema_version'] == onefile.LOG_SCHEMA_VERSION
    assert rows[0]['run_id'] == 'run-1'
    assert rows[0]['mode'] == 'verify'
    assert rows[0]['profile'] == 'smoke'
    assert rows[0]['component'] == 'chess_onefile'
    assert rows[0]['artifact_root'] == '/tmp/artifacts'
    assert rows[0]['payload']['api_key'] == '[REDACTED]'
    assert rows[0]['payload']['nested']['token'] == '[REDACTED]'
    assert rows[0]['payload']['message'] == 'sk-[REDACTED]'
    assert rows[-1]['event'] == 'logger_finalize'
    assert rows[-1]['payload']['status'] == 'completed'


def test_jsonl_logger_rotates_when_size_limit_is_hit(tmp_path: Path) -> None:
    log_path = tmp_path / 'rotate.jsonl'
    logger = onefile.JSONLLogger(
        log_path,
        run_id='run-rotate',
        mode='verify',
        profile='smoke',
        artifact_root=str(tmp_path),
        max_bytes=512,
        backup_count=2,
    )
    for idx in range(16):
        logger.write('rotation_probe', {'idx': idx, 'blob': 'x' * 180})
    logger.finalize('completed')
    rotated = sorted(tmp_path.glob('rotate.jsonl*'))
    assert len(rotated) >= 2


def test_gradient_checkpointing_feature_path_runs_backward() -> None:
    cfg = make_mirror_cfg(
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        use_moe=False,
        use_liquid=False,
        use_liquid_adapter=False,
        use_qinn=False,
        use_global_workspace_broadcast=True,
        use_world_model_head=False,
        use_gradient_checkpointing=True,
        dropout=0.0,
        attention_dropout=0.0,
        ffn_dropout=0.0,
    )
    model = onefile.ChessPolicyValueNet(cfg, vocab_size=len(onefile.MOVE_VOCAB))
    model.train()
    batch = onefile.collate_examples(
        [
            onefile.ChessExample(
                piece_ids=[0] * 64,
                meta_ids=[0] * len(onefile.ChessPolicyValueNet.META_CARDINALITIES),
                legal_move_ids=[onefile.MOVE_TO_ID['e2e4'], onefile.MOVE_TO_ID['d2d4']],
                target_move_id=onefile.MOVE_TO_ID['e2e4'],
                value_target=0.2,
                phase=0,
                source_game_id='g1',
                ply=0,
                total_plies=1,
                turn=1,
                has_eval=False,
                opening_prefix='',
                value_source='test',
                source_archive='test',
                position_hash='p1',
                move_uci='e2e4',
            )
        ]
    )
    loss, metrics, _ = onefile.compute_loss(model, batch, cfg)
    loss.backward()
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.detach().abs().sum().item())
    assert metrics['loss'] >= 0.0
    assert grad_norm > 0.0


def test_forward_batch_metrics_reports_auxiliary_head_metrics() -> None:
    cfg = make_mirror_cfg(
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        use_moe=False,
        use_liquid=False,
        use_liquid_adapter=False,
        use_qinn=False,
        use_global_workspace_broadcast=False,
        use_world_model_head=False,
        use_phase_head=True,
        use_wdl_head=True,
        use_legality_head=True,
        dropout=0.0,
        attention_dropout=0.0,
        ffn_dropout=0.0,
    )
    model = onefile.ChessPolicyValueNet(cfg, vocab_size=len(onefile.MOVE_VOCAB))
    batch = onefile.collate_examples(
        [
            onefile.ChessExample(
                piece_ids=[0] * 64,
                meta_ids=[0] * len(onefile.ChessPolicyValueNet.META_CARDINALITIES),
                legal_move_ids=[onefile.MOVE_TO_ID['e2e4'], onefile.MOVE_TO_ID['d2d4']],
                target_move_id=onefile.MOVE_TO_ID['e2e4'],
                value_target=0.8,
                phase=0,
                source_game_id='g1',
                ply=0,
                total_plies=1,
                turn=1,
                has_eval=False,
                opening_prefix='',
                value_source='test',
                source_archive='test',
                position_hash='p1',
                move_uci='e2e4',
            )
        ]
    )
    loss, metrics, _, _, _ = onefile.forward_batch_metrics(model, batch, cfg)
    assert float(loss.detach().item()) >= 0.0
    assert 'phase_loss' in metrics
    assert 'wdl_loss' in metrics
    assert 'legality_loss' in metrics
    assert 'phase_accuracy' in metrics
    assert 'wdl_accuracy' in metrics
    assert 'legality_head_top1_is_legal_rate' in metrics


def test_generate_selfplay_report_runs_with_stub_model(tmp_path: Path) -> None:
    cfg = {
        **onefile.RUN_CONFIG,
        'artifact_root': str(tmp_path),
        'selfplay_eval_enabled': True,
        'selfplay_games': 1,
        'selfplay_max_plies': 4,
        'selfplay_opening_prefix_plies': 1,
    }
    layout = onefile.make_layout(cfg)
    report = onefile.generate_selfplay_report(StubInferenceModel(), cfg, onefile.torch.device('cpu'), layout)
    assert report['status'] == 'completed'
    assert report['games_played'] == 1
    assert report['games'][0]['pgn_path']


def test_build_replay_buffer_report_collects_positions() -> None:
    selfplay_report = {
        'status': 'completed',
        'games': [
            {
                'game_index': 0,
                'moves': [
                    {'ply': 1, 'fen_before': 'fen_a', 'move': 'e2e4', 'value': 0.1, 'confidence': {'tier': 'medium'}, 'auxiliary_predictions': {}},
                    {'ply': 2, 'fen_before': 'fen_b', 'move': 'e7e5', 'value': 0.0, 'confidence': {'tier': 'medium'}, 'auxiliary_predictions': {}},
                ],
            }
        ],
    }
    report = onefile.build_replay_buffer_report(selfplay_report, {'replay_buffer_enabled': True, 'replay_buffer_max_positions': 8})
    assert report['status'] == 'completed'
    assert report['positions'] == 2
    assert report['records'][0]['move'] == 'e2e4'


def test_play_inference_mode_tournament_runs_with_stub_model(tmp_path: Path) -> None:
    cfg = {
        **onefile.RUN_CONFIG,
        'artifact_root': str(tmp_path),
        'tournament_eval_enabled': True,
        'tournament_games': 2,
        'tournament_max_plies': 4,
        'selfplay_opening_prefix_plies': 1,
        'search_enabled': True,
    }
    layout = onefile.make_layout(cfg)
    report = onefile.play_inference_mode_tournament(StubInferenceModel(), cfg, onefile.torch.device('cpu'), layout)
    assert report['status'] == 'completed'
    assert report['games_played'] == 2
    assert 'search_assisted' in report['players']


def test_write_closure_manifests_marks_closure_artifacts_present(tmp_path: Path) -> None:
    cfg = {
        **onefile.RUN_CONFIG,
        'artifact_root': str(tmp_path),
        'mode': 'train',
        'profile': 'strength_4060_24h_all_on_experimental',
        'feature_bundle': 'all_on_experimental',
    }
    layout = onefile.make_layout(cfg)
    required_reports = [
        'resolved_config.json',
        'run_summary.json',
        'run_summary.md',
        'data_card.json',
        'model_card.json',
        'eval_card.json',
        'benchmark_protocol.json',
        'feature_flag_report.json',
        'feature_flag_report.md',
        'observability_report.json',
        'artifact_manifest_with_hashes.json',
        'selfplay_report.json',
        'inference_mode_tournament_report.json',
        'replay_buffer_manifest.json',
    ]
    for name in required_reports:
        path = layout.reports_dir / name
        if path.suffix == '.md':
            path.write_text(f'# {name}\n', encoding='utf-8')
        else:
            path.write_text(json.dumps({'ok': True}), encoding='utf-8')
    (layout.logs_dir / 'run_log.jsonl').write_text('', encoding='utf-8')
    best_ckpt = layout.checkpoints_dir / 'best_by_val_loss.pt'
    latest_ckpt = layout.checkpoints_dir / 'latest.pt'
    best_ckpt.write_bytes(b'checkpoint')
    latest_ckpt.write_bytes(b'checkpoint')
    layout.final_zip_path.write_bytes(b'zip')
    layout.final_sha_path.write_text('deadbeef  bundle.zip\n', encoding='utf-8')
    payload = {
        'run_id': 'pytest-run',
        'script_version': onefile.SCRIPT_VERSION,
        'config': cfg,
        'execution_status': 'completed',
        'evaluation_status': 'completed',
        'rating_claim_status': 'internal_only',
        'selfplay_report': {'status': 'completed', 'games_played': 1, 'average_plies': 8.0},
        'tournament_report': {'status': 'completed', 'games_played': 2, 'players': {'search_assisted': {}, 'pure_policy': {}}},
        'replay_buffer_report': {'status': 'completed', 'positions': 12, 'games_used': 1, 'truncated': False},
        'curated_position_suite': {'status': 'completed', 'exact_hit_rate': 0.5, 'top3_hit_rate': 1.0},
        'stockfish': {'status': 'completed', 'games_total': 4, 'elo_proxy_internal': 1820.0},
        'compile_report': {'status': 'completed'},
        'forward_verify': {'status': 'completed'},
        'holdout_validation': {'status': 'completed'},
        'locked_test': {'status': 'completed'},
        'legality_report': {'status': 'completed'},
        'bundle': {},
    }
    onefile.write_closure_manifests(layout, payload)
    truth = json.loads((layout.reports_dir / 'artifact_truth_matrix.json').read_text(encoding='utf-8'))
    entries = {entry['label']: entry for entry in truth['entries']}
    assert entries['run_status_manifest']['exists'] is True
    assert entries['postrun_analysis_manifest']['exists'] is True
    assert entries['artifact_truth_matrix']['exists'] is True
    assert entries['run_contract']['exists'] is False
    assert entries['release_snapshot']['exists'] is False
    assert entries['evidence_pack_stub']['exists'] is False
    assert entries['rc_stub']['exists'] is False
    assert entries['golden_stub']['exists'] is False
    assert entries['handoff_pack_manifest']['exists'] is False
    assert entries['operator_handoff_summary']['exists'] is False
    assert entries['external_repro_stub']['exists'] is False
    assert entries['pilot_stub']['exists'] is False
    assert entries['security_stub']['exists'] is False
    assert entries['legal_stub']['exists'] is False
    assert entries['operator_handbook_stub']['exists'] is False
    assert entries['dr_evidence_stub']['exists'] is False
    assert entries['backup_retention_stub']['exists'] is False
    assert entries['blind_handoff_stub']['exists'] is False
    assert entries['release_notes_stub']['exists'] is False
    assert entries['freeze_manifest_stub']['exists'] is False
    assert entries['changelog_snapshot']['exists'] is False
    assert entries['maintenance_policy_stub']['exists'] is False
    assert entries['export_truth_stub']['exists'] is False
    assert entries['device_validation_stub']['exists'] is False
    assert entries['packaging_closure_stub']['exists'] is False
    assert entries['installer_validation_stub']['exists'] is False
    assert entries['benchmark_raw_outputs_stub']['exists'] is False
    assert entries['benchmark_compare_report_stub']['exists'] is False
    assert entries['benchmark_summary_stub']['exists'] is False
    assert entries['benchmark_manifest_stub']['exists'] is False
    assert entries['training_report_stub']['exists'] is False
    assert entries['token_accounting_stub']['exists'] is False
    assert entries['compute_accounting_stub']['exists'] is False
    assert entries['cost_report_stub']['exists'] is False
    assert entries['final_weights_truth_stub']['exists'] is False
    assert entries['best_checkpoint_truth_stub']['exists'] is False
    assert entries['latest_checkpoint_truth_stub']['exists'] is False
    assert entries['trained_artifact_registry_stub']['exists'] is False
    assert entries['core_complete_decision_stub']['exists'] is False
    assert entries['research_continues_stub']['exists'] is False
    assert entries['product_maintenance_only_stub']['exists'] is False
    assert entries['closure_decision_record_stub']['exists'] is False
    assert entries['master_closure_table']['exists'] is False
    assert entries['remaining_core_blockers']['exists'] is False
    assert entries['repo_side_completion_summary']['exists'] is False
    assert entries['readiness_snapshot']['exists'] is False
    assert entries['aggregated_master_table']['exists'] is False
    assert entries['real_remaining_core_work']['exists'] is False
    assert entries['repo_truth_inventory']['exists'] is False
    assert entries['closure_gap_summary']['exists'] is False
    assert entries['project_master_truth_reference']['exists'] is False
    assert entries['project_remaining_real_blockers']['exists'] is False
    assert entries['truth_docs_index']['exists'] is False
    assert entries['truth_docs_drift_report']['exists'] is False
    assert entries['project_blocker_action_plan']['exists'] is False
    assert entries['project_blocker_dependency_graph']['exists'] is False
    assert entries['project_execution_sequence']['exists'] is False
    assert entries['project_lane_status_board']['exists'] is False
    assert entries['generated_truth_consistency_report']['exists'] is False
    assert entries['generated_truth_crosscheck_matrix']['exists'] is False
    assert truth['present_required_count'] < truth['required_count']


def test_write_release_evidence_reports_writes_release_surfaces(tmp_path: Path) -> None:
    cfg = {
        **onefile.RUN_CONFIG,
        'artifact_root': str(tmp_path),
        'mode': 'train',
        'profile': 'strength_4060_24h_all_on_experimental',
        'feature_bundle': 'all_on_experimental',
    }
    layout = onefile.make_layout(cfg)
    required_reports = [
        'resolved_config.json',
        'run_summary.json',
        'run_summary.md',
        'data_card.json',
        'model_card.json',
        'eval_card.json',
        'benchmark_protocol.json',
        'feature_flag_report.json',
        'feature_flag_report.md',
        'observability_report.json',
        'artifact_manifest_with_hashes.json',
        'selfplay_report.json',
        'inference_mode_tournament_report.json',
        'replay_buffer_manifest.json',
    ]
    for name in required_reports:
        path = layout.reports_dir / name
        if path.suffix == '.md':
            path.write_text(f'# {name}\n', encoding='utf-8')
        else:
            path.write_text(json.dumps({'ok': True}), encoding='utf-8')
    (layout.logs_dir / 'run_log.jsonl').write_text('', encoding='utf-8')
    best_ckpt = layout.checkpoints_dir / 'best_by_val_loss.pt'
    latest_ckpt = layout.checkpoints_dir / 'latest.pt'
    best_ckpt.write_bytes(b'checkpoint')
    latest_ckpt.write_bytes(b'checkpoint')
    layout.final_zip_path.write_bytes(b'zip')
    layout.final_sha_path.write_text('deadbeef  bundle.zip\n', encoding='utf-8')
    payload = {
        'run_id': 'pytest-run',
        'script_version': onefile.SCRIPT_VERSION,
        'config': cfg,
        'execution_status': 'completed',
        'evaluation_status': 'completed',
        'rating_claim_status': onefile.RatingClaimStatus.PROXY_ONLY.value,
        'selfplay_report': {'status': 'completed', 'games_played': 1, 'average_plies': 8.0},
        'tournament_report': {'status': 'completed', 'games_played': 2, 'players': {'search_assisted': {}, 'pure_policy': {}}},
        'replay_buffer_report': {'status': 'completed', 'positions': 12, 'games_used': 1, 'truncated': False},
        'curated_position_suite': {'status': 'completed', 'exact_hit_rate': 0.5, 'top3_hit_rate': 1.0},
        'stockfish': {'status': 'completed', 'games_total': 4, 'elo_proxy_internal': 1820.0},
        'compile_report': {'status': 'completed'},
        'forward_verify': {'status': 'completed'},
        'holdout_validation': {'status': 'completed'},
        'locked_test': {'status': 'completed'},
        'legality_report': {'status': 'completed'},
        'dataset_provenance': {'sampling_strategy': 'seeded', 'source_mode': 'embedded_seed'},
        'notes': {'replay_is_demo_only': True, 'internal_proxy_only': True},
        'best_checkpoint': str(best_ckpt),
        'latest_checkpoint': str(latest_ckpt),
        'bundle': {'zip_path': str(layout.final_zip_path), 'sha256_path': str(layout.final_sha_path), 'encrypted': False},
    }
    onefile.write_closure_manifests(layout, payload)
    onefile.write_release_evidence_reports(layout, payload)
    truth = json.loads((layout.reports_dir / 'artifact_truth_matrix.json').read_text(encoding='utf-8'))
    run_contract = json.loads((layout.reports_dir / 'run_contract.json').read_text(encoding='utf-8'))
    release_snapshot = json.loads((layout.reports_dir / 'release_snapshot.json').read_text(encoding='utf-8'))
    evidence_pack = json.loads((layout.reports_dir / 'evidence_pack_stub.json').read_text(encoding='utf-8'))
    truth_registry = json.loads((layout.reports_dir / 'final_truth_registry.json').read_text(encoding='utf-8'))
    claim_registry = json.loads((layout.reports_dir / 'claim_registry.json').read_text(encoding='utf-8'))
    known_limits = json.loads((layout.reports_dir / 'known_limits.json').read_text(encoding='utf-8'))
    support_matrix = json.loads((layout.reports_dir / 'support_matrix.json').read_text(encoding='utf-8'))
    release_gate_summary = json.loads((layout.reports_dir / 'release_gate_summary.json').read_text(encoding='utf-8'))
    rc_stub = json.loads((layout.reports_dir / 'rc_stub.json').read_text(encoding='utf-8'))
    golden_stub = json.loads((layout.reports_dir / 'golden_stub.json').read_text(encoding='utf-8'))
    handoff_pack_manifest = json.loads((layout.reports_dir / 'handoff_pack_manifest.json').read_text(encoding='utf-8'))
    operator_handoff_summary = json.loads((layout.reports_dir / 'operator_handoff_summary.json').read_text(encoding='utf-8'))
    external_repro_stub = json.loads((layout.reports_dir / 'external_repro_stub.json').read_text(encoding='utf-8'))
    pilot_stub = json.loads((layout.reports_dir / 'pilot_stub.json').read_text(encoding='utf-8'))
    security_stub = json.loads((layout.reports_dir / 'security_stub.json').read_text(encoding='utf-8'))
    legal_stub = json.loads((layout.reports_dir / 'legal_stub.json').read_text(encoding='utf-8'))
    operator_handbook_stub = json.loads((layout.reports_dir / 'operator_handbook_stub.json').read_text(encoding='utf-8'))
    dr_evidence_stub = json.loads((layout.reports_dir / 'dr_evidence_stub.json').read_text(encoding='utf-8'))
    backup_retention_stub = json.loads((layout.reports_dir / 'backup_retention_stub.json').read_text(encoding='utf-8'))
    blind_handoff_stub = json.loads((layout.reports_dir / 'blind_handoff_stub.json').read_text(encoding='utf-8'))
    release_notes_stub = json.loads((layout.reports_dir / 'release_notes_stub.json').read_text(encoding='utf-8'))
    freeze_manifest_stub = json.loads((layout.reports_dir / 'freeze_manifest_stub.json').read_text(encoding='utf-8'))
    changelog_snapshot = json.loads((layout.reports_dir / 'changelog_snapshot.json').read_text(encoding='utf-8'))
    maintenance_policy_stub = json.loads((layout.reports_dir / 'maintenance_policy_stub.json').read_text(encoding='utf-8'))
    export_truth_stub = json.loads((layout.reports_dir / 'export_truth_stub.json').read_text(encoding='utf-8'))
    device_validation_stub = json.loads((layout.reports_dir / 'device_validation_stub.json').read_text(encoding='utf-8'))
    packaging_closure_stub = json.loads((layout.reports_dir / 'packaging_closure_stub.json').read_text(encoding='utf-8'))
    installer_validation_stub = json.loads((layout.reports_dir / 'installer_validation_stub.json').read_text(encoding='utf-8'))
    benchmark_raw_outputs_stub = json.loads((layout.reports_dir / 'benchmark_raw_outputs_stub.json').read_text(encoding='utf-8'))
    benchmark_compare_report_stub = json.loads((layout.reports_dir / 'benchmark_compare_report_stub.json').read_text(encoding='utf-8'))
    benchmark_summary_stub = json.loads((layout.reports_dir / 'benchmark_summary_stub.json').read_text(encoding='utf-8'))
    benchmark_manifest_stub = json.loads((layout.reports_dir / 'benchmark_manifest_stub.json').read_text(encoding='utf-8'))
    training_report_stub = json.loads((layout.reports_dir / 'training_report_stub.json').read_text(encoding='utf-8'))
    token_accounting_stub = json.loads((layout.reports_dir / 'token_accounting_stub.json').read_text(encoding='utf-8'))
    compute_accounting_stub = json.loads((layout.reports_dir / 'compute_accounting_stub.json').read_text(encoding='utf-8'))
    cost_report_stub = json.loads((layout.reports_dir / 'cost_report_stub.json').read_text(encoding='utf-8'))
    final_weights_truth_stub = json.loads((layout.reports_dir / 'final_weights_truth_stub.json').read_text(encoding='utf-8'))
    best_checkpoint_truth_stub = json.loads((layout.reports_dir / 'best_checkpoint_truth_stub.json').read_text(encoding='utf-8'))
    latest_checkpoint_truth_stub = json.loads((layout.reports_dir / 'latest_checkpoint_truth_stub.json').read_text(encoding='utf-8'))
    trained_artifact_registry_stub = json.loads((layout.reports_dir / 'trained_artifact_registry_stub.json').read_text(encoding='utf-8'))
    core_complete_decision_stub = json.loads((layout.reports_dir / 'core_complete_decision_stub.json').read_text(encoding='utf-8'))
    research_continues_stub = json.loads((layout.reports_dir / 'research_continues_stub.json').read_text(encoding='utf-8'))
    product_maintenance_only_stub = json.loads((layout.reports_dir / 'product_maintenance_only_stub.json').read_text(encoding='utf-8'))
    closure_decision_record_stub = json.loads((layout.reports_dir / 'closure_decision_record_stub.json').read_text(encoding='utf-8'))
    master_closure_table = json.loads((layout.reports_dir / 'master_closure_table.json').read_text(encoding='utf-8'))
    remaining_core_blockers = json.loads((layout.reports_dir / 'remaining_core_blockers.json').read_text(encoding='utf-8'))
    repo_side_completion_summary = json.loads((layout.reports_dir / 'repo_side_completion_summary.json').read_text(encoding='utf-8'))
    readiness_snapshot = json.loads((layout.reports_dir / 'readiness_snapshot.json').read_text(encoding='utf-8'))
    aggregated_master_table = json.loads((layout.reports_dir / 'aggregated_master_table.json').read_text(encoding='utf-8'))
    real_remaining_core_work = json.loads((layout.reports_dir / 'real_remaining_core_work.json').read_text(encoding='utf-8'))
    repo_truth_inventory = json.loads((layout.reports_dir / 'repo_truth_inventory.json').read_text(encoding='utf-8'))
    closure_gap_summary = json.loads((layout.reports_dir / 'closure_gap_summary.json').read_text(encoding='utf-8'))
    project_master_truth_reference = json.loads((layout.reports_dir / 'project_master_truth_reference.json').read_text(encoding='utf-8'))
    project_remaining_real_blockers = json.loads((layout.reports_dir / 'project_remaining_real_blockers.json').read_text(encoding='utf-8'))
    truth_docs_index = json.loads((layout.reports_dir / 'truth_docs_index.json').read_text(encoding='utf-8'))
    truth_docs_drift_report = json.loads((layout.reports_dir / 'truth_docs_drift_report.json').read_text(encoding='utf-8'))
    project_blocker_action_plan = json.loads((layout.reports_dir / 'project_blocker_action_plan.json').read_text(encoding='utf-8'))
    project_blocker_dependency_graph = json.loads((layout.reports_dir / 'project_blocker_dependency_graph.json').read_text(encoding='utf-8'))
    project_execution_sequence = json.loads((layout.reports_dir / 'project_execution_sequence.json').read_text(encoding='utf-8'))
    project_lane_status_board = json.loads((layout.reports_dir / 'project_lane_status_board.json').read_text(encoding='utf-8'))
    generated_truth_consistency_report = json.loads((layout.reports_dir / 'generated_truth_consistency_report.json').read_text(encoding='utf-8'))
    generated_truth_crosscheck_matrix = json.loads((layout.reports_dir / 'generated_truth_crosscheck_matrix.json').read_text(encoding='utf-8'))
    entries = {entry['label']: entry for entry in truth['entries']}
    assert run_contract['schema'] == 'chess_run_contract_v1'
    assert run_contract['feature_bundle'] == 'all_on_experimental'
    assert release_snapshot['schema'] == 'chess_release_snapshot_v1'
    assert release_snapshot['release_surface_status'] == 'candidate_internal_only'
    assert evidence_pack['schema'] == 'chess_evidence_pack_stub_v1'
    assert evidence_pack['status'] == 'partial_internal_only'
    assert truth_registry['schema'] == 'chess_final_truth_registry_v1'
    assert any(claim['label'] == 'external_strength_claim' for claim in truth_registry['claims'])
    assert claim_registry['schema'] == 'chess_claim_registry_v1'
    assert known_limits['schema'] == 'chess_known_limits_v1'
    assert support_matrix['schema'] == 'chess_support_matrix_v1'
    assert release_gate_summary['schema'] == 'chess_release_gate_summary_v1'
    assert rc_stub['schema'] == 'chess_rc_stub_v1'
    assert golden_stub['schema'] == 'chess_golden_stub_v1'
    assert handoff_pack_manifest['schema'] == 'chess_handoff_pack_manifest_v1'
    assert operator_handoff_summary['schema'] == 'chess_operator_handoff_summary_v1'
    assert external_repro_stub['schema'] == 'chess_external_repro_stub_v1'
    assert pilot_stub['schema'] == 'chess_pilot_stub_v1'
    assert security_stub['schema'] == 'chess_security_stub_v1'
    assert legal_stub['schema'] == 'chess_legal_stub_v1'
    assert operator_handbook_stub['schema'] == 'chess_operator_handbook_stub_v1'
    assert dr_evidence_stub['schema'] == 'chess_dr_evidence_stub_v1'
    assert backup_retention_stub['schema'] == 'chess_backup_retention_stub_v1'
    assert blind_handoff_stub['schema'] == 'chess_blind_handoff_stub_v1'
    assert release_notes_stub['schema'] == 'chess_release_notes_stub_v1'
    assert freeze_manifest_stub['schema'] == 'chess_freeze_manifest_stub_v1'
    assert changelog_snapshot['schema'] == 'chess_changelog_snapshot_v1'
    assert maintenance_policy_stub['schema'] == 'chess_maintenance_policy_stub_v1'
    assert export_truth_stub['schema'] == 'chess_export_truth_stub_v1'
    assert device_validation_stub['schema'] == 'chess_device_validation_stub_v1'
    assert packaging_closure_stub['schema'] == 'chess_packaging_closure_stub_v1'
    assert installer_validation_stub['schema'] == 'chess_installer_validation_stub_v1'
    assert benchmark_raw_outputs_stub['schema'] == 'chess_benchmark_raw_outputs_stub_v1'
    assert benchmark_compare_report_stub['schema'] == 'chess_benchmark_compare_report_stub_v1'
    assert benchmark_summary_stub['schema'] == 'chess_benchmark_summary_stub_v1'
    assert benchmark_manifest_stub['schema'] == 'chess_benchmark_manifest_stub_v1'
    assert training_report_stub['schema'] == 'chess_training_report_stub_v1'
    assert token_accounting_stub['schema'] == 'chess_token_accounting_stub_v1'
    assert compute_accounting_stub['schema'] == 'chess_compute_accounting_stub_v1'
    assert cost_report_stub['schema'] == 'chess_cost_report_stub_v1'
    assert final_weights_truth_stub['schema'] == 'chess_final_weights_truth_stub_v1'
    assert best_checkpoint_truth_stub['schema'] == 'chess_best_checkpoint_truth_stub_v1'
    assert latest_checkpoint_truth_stub['schema'] == 'chess_latest_checkpoint_truth_stub_v1'
    assert trained_artifact_registry_stub['schema'] == 'chess_trained_artifact_registry_stub_v1'
    assert core_complete_decision_stub['schema'] == 'chess_core_complete_decision_stub_v1'
    assert research_continues_stub['schema'] == 'chess_research_continues_stub_v1'
    assert product_maintenance_only_stub['schema'] == 'chess_product_maintenance_only_stub_v1'
    assert closure_decision_record_stub['schema'] == 'chess_closure_decision_record_stub_v1'
    assert master_closure_table['schema'] == 'chess_master_closure_table_v1'
    assert remaining_core_blockers['schema'] == 'chess_remaining_core_blockers_v1'
    assert repo_side_completion_summary['schema'] == 'chess_repo_side_completion_summary_v1'
    assert readiness_snapshot['schema'] == 'chess_readiness_snapshot_v1'
    assert aggregated_master_table['schema'] == 'chess_aggregated_master_table_v1'
    assert real_remaining_core_work['schema'] == 'chess_real_remaining_core_work_v1'
    assert repo_truth_inventory['schema'] == 'chess_repo_truth_inventory_v1'
    assert closure_gap_summary['schema'] == 'chess_closure_gap_summary_v1'
    assert project_master_truth_reference['schema'] == 'chess_project_master_truth_reference_v1'
    assert project_remaining_real_blockers['schema'] == 'chess_project_remaining_real_blockers_v1'
    assert truth_docs_index['schema'] == 'chess_truth_docs_index_v1'
    assert truth_docs_drift_report['schema'] == 'chess_truth_docs_drift_report_v1'
    assert project_blocker_action_plan['schema'] == 'chess_project_blocker_action_plan_v1'
    assert project_blocker_dependency_graph['schema'] == 'chess_project_blocker_dependency_graph_v1'
    assert project_execution_sequence['schema'] == 'chess_project_execution_sequence_v1'
    assert project_lane_status_board['schema'] == 'chess_project_lane_status_board_v1'
    assert generated_truth_consistency_report['schema'] == 'chess_generated_truth_consistency_report_v1'
    assert generated_truth_crosscheck_matrix['schema'] == 'chess_generated_truth_crosscheck_matrix_v1'
    assert entries['run_contract']['exists'] is True
    assert entries['release_snapshot']['exists'] is True
    assert entries['evidence_pack_stub']['exists'] is True
    assert entries['claim_registry']['exists'] is True
    assert entries['known_limits']['exists'] is True
    assert entries['support_matrix']['exists'] is True
    assert entries['release_gate_summary']['exists'] is True
    assert entries['rc_stub']['exists'] is True
    assert entries['golden_stub']['exists'] is True
    assert entries['handoff_pack_manifest']['exists'] is True
    assert entries['operator_handoff_summary']['exists'] is True
    assert entries['external_repro_stub']['exists'] is True
    assert entries['pilot_stub']['exists'] is True
    assert entries['security_stub']['exists'] is True
    assert entries['legal_stub']['exists'] is True
    assert entries['operator_handbook_stub']['exists'] is True
    assert entries['dr_evidence_stub']['exists'] is True
    assert entries['backup_retention_stub']['exists'] is True
    assert entries['blind_handoff_stub']['exists'] is True
    assert entries['release_notes_stub']['exists'] is True
    assert entries['freeze_manifest_stub']['exists'] is True
    assert entries['changelog_snapshot']['exists'] is True
    assert entries['maintenance_policy_stub']['exists'] is True
    assert entries['export_truth_stub']['exists'] is True
    assert entries['device_validation_stub']['exists'] is True
    assert entries['packaging_closure_stub']['exists'] is True
    assert entries['installer_validation_stub']['exists'] is True
    assert entries['benchmark_raw_outputs_stub']['exists'] is True
    assert entries['benchmark_compare_report_stub']['exists'] is True
    assert entries['benchmark_summary_stub']['exists'] is True
    assert entries['benchmark_manifest_stub']['exists'] is True
    assert entries['training_report_stub']['exists'] is True
    assert entries['token_accounting_stub']['exists'] is True
    assert entries['compute_accounting_stub']['exists'] is True
    assert entries['cost_report_stub']['exists'] is True
    assert entries['final_weights_truth_stub']['exists'] is True
    assert entries['best_checkpoint_truth_stub']['exists'] is True
    assert entries['latest_checkpoint_truth_stub']['exists'] is True
    assert entries['trained_artifact_registry_stub']['exists'] is True
    assert entries['core_complete_decision_stub']['exists'] is True
    assert entries['research_continues_stub']['exists'] is True
    assert entries['product_maintenance_only_stub']['exists'] is True
    assert entries['closure_decision_record_stub']['exists'] is True
    assert entries['master_closure_table']['exists'] is True
    assert entries['remaining_core_blockers']['exists'] is True
    assert entries['repo_side_completion_summary']['exists'] is True
    assert entries['readiness_snapshot']['exists'] is True
    assert entries['aggregated_master_table']['exists'] is True
    assert entries['real_remaining_core_work']['exists'] is True
    assert entries['repo_truth_inventory']['exists'] is True
    assert entries['closure_gap_summary']['exists'] is True
    assert entries['project_master_truth_reference']['exists'] is True
    assert entries['project_remaining_real_blockers']['exists'] is True
    assert entries['truth_docs_index']['exists'] is True
    assert entries['truth_docs_drift_report']['exists'] is True
    assert entries['project_blocker_action_plan']['exists'] is True
    assert entries['project_blocker_dependency_graph']['exists'] is True
    assert entries['project_execution_sequence']['exists'] is True
    assert entries['project_lane_status_board']['exists'] is True
    assert entries['generated_truth_consistency_report']['exists'] is True
    assert entries['generated_truth_crosscheck_matrix']['exists'] is True
    assert truth['present_required_count'] == truth['required_count']
    assert rc_stub['status'] == 'candidate_internal_only'
    assert golden_stub['status'] == 'not_ready'
    assert operator_handoff_summary['handoff_surface_status'] == 'internal_ready'
    assert operator_handoff_summary['external_stub_count'] == 4
    assert external_repro_stub['status'] == 'pending_external_reproduction'
    assert pilot_stub['status'] == 'pending_pilot_validation'
    assert security_stub['status'] == 'pending_security_review'
    assert legal_stub['status'] == 'pending_legal_review'
    assert operator_handbook_stub['status'] == 'pending_operator_handbook_validation'
    assert dr_evidence_stub['status'] == 'pending_dr_validation'
    assert backup_retention_stub['status'] == 'pending_retention_policy_finalization'
    assert blind_handoff_stub['status'] == 'pending_blind_handoff_rehearsal'
    assert release_notes_stub['status'] == 'pending_release_note_curation'
    assert freeze_manifest_stub['status'] == 'pending_freeze_signoff'
    assert maintenance_policy_stub['status'] == 'pending_maintenance_policy_finalization'
    assert export_truth_stub['status'] == 'pending_export_truth_validation'
    assert device_validation_stub['status'] == 'pending_device_validation'
    assert packaging_closure_stub['status'] == 'pending_packaging_closure'
    assert installer_validation_stub['status'] == 'pending_installer_validation'
    assert benchmark_raw_outputs_stub['status'] == 'pending_benchmark_raw_output_capture'
    assert benchmark_compare_report_stub['status'] == 'pending_benchmark_compare_report'
    assert benchmark_summary_stub['status'] == 'pending_benchmark_summary_closure'
    assert benchmark_manifest_stub['status'] == 'pending_benchmark_manifest_lock'
    assert training_report_stub['status'] == 'pending_training_report_closure'
    assert token_accounting_stub['status'] == 'pending_token_accounting'
    assert compute_accounting_stub['status'] == 'pending_compute_accounting'
    assert cost_report_stub['status'] == 'pending_cost_report'
    assert final_weights_truth_stub['status'] == 'pending_final_weights_truth'
    assert final_weights_truth_stub['bundle_zip_present'] is True
    assert best_checkpoint_truth_stub['status'] == 'pending_best_checkpoint_truth'
    assert best_checkpoint_truth_stub['best_checkpoint_present'] is True
    assert latest_checkpoint_truth_stub['status'] == 'pending_latest_checkpoint_truth'
    assert latest_checkpoint_truth_stub['latest_checkpoint_present'] is True
    assert trained_artifact_registry_stub['status'] == 'pending_trained_artifact_registry_lock'
    assert 'best_checkpoint' in trained_artifact_registry_stub['tracked_labels']
    assert 'latest_checkpoint' in trained_artifact_registry_stub['tracked_labels']
    assert core_complete_decision_stub['status'] == 'pending_core_complete_decision'
    assert research_continues_stub['status'] == 'pending_research_separation_decision'
    assert product_maintenance_only_stub['status'] == 'pending_maintenance_only_decision'
    assert closure_decision_record_stub['status'] == 'pending_management_closure_record'
    assert 'release_gate_summary' in closure_decision_record_stub['tracked_labels']
    assert master_closure_table['row_count'] >= 12
    assert any(row['label'] == 'management_closure' and row['complete'] is True for row in master_closure_table['rows'])
    assert any(row['label'] == 'truth_docs_alignment' and row['complete'] is True for row in master_closure_table['rows'])
    assert any(row['label'] == 'project_actionability' and row['complete'] is True for row in master_closure_table['rows'])
    assert any(row['label'] == 'generated_truth_consistency' and row['complete'] is True for row in master_closure_table['rows'])
    assert remaining_core_blockers['blocker_count'] >= 1
    assert any(blocker['label'] == 'management_closure_pending' for blocker in remaining_core_blockers['blockers'])
    assert repo_side_completion_summary['repo_side_complete'] is True
    assert repo_side_completion_summary['missing_required_count'] == 0
    assert readiness_snapshot['release_surface_status'] == 'candidate_internal_only'
    assert readiness_snapshot['rc_status'] == 'candidate_internal_only'
    assert readiness_snapshot['golden_status'] == 'not_ready'
    assert aggregated_master_table['row_count'] >= 12
    assert any(row['label'] == 'management_closure' and row['real_closure_blocked'] is True for row in aggregated_master_table['rows'])
    assert any(row['label'] == 'truth_docs_alignment' and row['real_closure_blocked'] is False for row in aggregated_master_table['rows'])
    assert any(row['label'] == 'project_actionability' and row['real_closure_blocked'] is False for row in aggregated_master_table['rows'])
    assert any(row['label'] == 'generated_truth_consistency' and row['real_closure_blocked'] is False for row in aggregated_master_table['rows'])
    assert real_remaining_core_work['item_count'] == remaining_core_blockers['blocker_count']
    assert repo_truth_inventory['entry_count'] >= truth['required_count']
    assert closure_gap_summary['repo_side_complete'] is True
    assert closure_gap_summary['missing_required_count'] == 0
    assert closure_gap_summary['blocker_count'] == remaining_core_blockers['blocker_count']
    assert closure_gap_summary['truth_docs_status'] == 'in_sync'
    assert closure_gap_summary['project_actionability_status'] == 'ready'
    assert closure_gap_summary['generated_truth_crosscheck_status'] == 'consistent'
    assert project_master_truth_reference['doc_exists'] is True
    assert project_master_truth_reference['doc_tr_exists'] is True
    assert project_master_truth_reference['row_count'] >= 10
    assert any(row['label'] == 'train_readiness_45k' for row in project_master_truth_reference['rows'])
    assert project_remaining_real_blockers['item_count'] >= 10
    assert any(item['label'] == 'rc_golden_final_release_pending' for item in project_remaining_real_blockers['items'])
    assert truth_docs_index['item_count'] >= 8
    assert all(item['exists'] is True for item in truth_docs_index['items'])
    assert truth_docs_drift_report['status'] == 'in_sync'
    assert truth_docs_drift_report['missing_truth_index_items'] == []
    assert project_blocker_action_plan['item_count'] == project_remaining_real_blockers['item_count']
    assert any(item['label'] == 'external_strength_unproven' and item['owner_domain'] == 'training_eval' for item in project_blocker_action_plan['items'])
    assert project_blocker_dependency_graph['node_count'] == project_remaining_real_blockers['item_count']
    assert project_blocker_dependency_graph['root_count'] >= 1
    assert project_blocker_dependency_graph['terminal_count'] >= 1
    assert any(
        node['label'] == 'external_strength_unproven' and 'benchmark_evidence_pending' in node['depends_on']
        for node in project_blocker_dependency_graph['nodes']
    )
    assert project_execution_sequence['item_count'] == project_remaining_real_blockers['item_count']
    assert project_execution_sequence['cycle_detected'] is False
    assert project_execution_sequence['items'][0]['label'] == 'real_training_outputs_pending'
    assert project_lane_status_board['lane_count'] == project_master_truth_reference['row_count']
    assert project_lane_status_board['covered_blocker_count'] == project_remaining_real_blockers['item_count']
    assert project_lane_status_board['status'] == 'ready'
    assert any(row['label'] == 'management_finalization' and row['blocker_count'] >= 1 for row in project_lane_status_board['rows'])
    assert generated_truth_consistency_report['status'] == 'consistent'
    assert generated_truth_consistency_report['failed_checks'] == []
    assert generated_truth_crosscheck_matrix['status'] == 'consistent'
    assert generated_truth_crosscheck_matrix['failed_checks'] == []
    assert changelog_snapshot['execution_status'] == 'completed'
    assert changelog_snapshot['evaluation_status'] == 'completed'
    assert 'release_gate_summary' in changelog_snapshot['included_labels']
    known_limit_labels = {item['label'] for item in known_limits['limits']}
    assert 'external_reproduction_pending' in known_limit_labels
    assert 'security_legal_pilot_pending' in known_limit_labels
    assert 'operator_handoff_dr_pending' in known_limit_labels
    assert 'release_governance_pending' in known_limit_labels
    assert 'device_export_packaging_pending' in known_limit_labels
    assert 'benchmark_closure_pending' in known_limit_labels
    assert 'training_accounting_pending' in known_limit_labels
    assert 'trained_artifact_truth_pending' in known_limit_labels
    assert 'management_closure_pending' in known_limit_labels
    assert 'truth_docs_drift_pending' not in known_limit_labels
    assert 'generated_truth_consistency_pending' not in known_limit_labels
    assert 'generated_truth_crosscheck_pending' not in known_limit_labels
    gate_labels = {item['label']: item['passed'] for item in release_gate_summary['gates']}
    assert gate_labels['external_closure_stubs_present'] is True
    assert gate_labels['operational_stub_surfaces_present'] is True
    assert gate_labels['release_governance_surfaces_present'] is True
    assert gate_labels['device_packaging_surfaces_present'] is True
    assert gate_labels['benchmark_closure_surfaces_present'] is True
    assert gate_labels['training_accounting_surfaces_present'] is True
    assert gate_labels['trained_artifact_surfaces_present'] is True
    assert gate_labels['management_closure_surfaces_present'] is True
    assert gate_labels['master_summary_surfaces_present'] is True
    assert gate_labels['aggregate_truth_surfaces_present'] is True
    assert gate_labels['project_truth_surfaces_present'] is True
    assert gate_labels['project_actionability_surfaces_present'] is True
    assert gate_labels['generated_truth_consistency_present'] is True
    assert gate_labels['generated_truth_crosscheck_present'] is True
    assert gate_labels['truth_docs_drift_clear'] is True
    assert gate_labels['generated_truth_consistency_clear'] is True
    assert gate_labels['generated_truth_crosscheck_clear'] is True
    handoff_labels = {item['label'] for item in handoff_pack_manifest['items']}
    assert {'external_repro_stub', 'pilot_stub', 'security_stub', 'legal_stub'} <= handoff_labels
    assert {'operator_handbook_stub', 'dr_evidence_stub', 'backup_retention_stub', 'blind_handoff_stub'} <= handoff_labels
    assert {'release_notes_stub', 'freeze_manifest_stub', 'changelog_snapshot', 'maintenance_policy_stub'} <= handoff_labels
    assert {'export_truth_stub', 'device_validation_stub', 'packaging_closure_stub', 'installer_validation_stub'} <= handoff_labels
    assert {'benchmark_raw_outputs_stub', 'benchmark_compare_report_stub', 'benchmark_summary_stub', 'benchmark_manifest_stub'} <= handoff_labels
    assert {'training_report_stub', 'token_accounting_stub', 'compute_accounting_stub', 'cost_report_stub'} <= handoff_labels
    assert {'final_weights_truth_stub', 'best_checkpoint_truth_stub', 'latest_checkpoint_truth_stub', 'trained_artifact_registry_stub'} <= handoff_labels
    assert {'core_complete_decision_stub', 'research_continues_stub', 'product_maintenance_only_stub', 'closure_decision_record_stub'} <= handoff_labels
    assert {'master_closure_table', 'remaining_core_blockers', 'repo_side_completion_summary', 'readiness_snapshot'} <= handoff_labels
    assert {'aggregated_master_table', 'real_remaining_core_work', 'repo_truth_inventory', 'closure_gap_summary'} <= handoff_labels
    assert {'project_master_truth_reference', 'project_remaining_real_blockers', 'truth_docs_index', 'truth_docs_drift_report'} <= handoff_labels
    assert {'project_blocker_action_plan', 'project_blocker_dependency_graph', 'project_execution_sequence', 'project_lane_status_board'} <= handoff_labels
    assert {'generated_truth_consistency_report', 'generated_truth_crosscheck_matrix'} <= handoff_labels
    assert release_gate_summary['overall_internal_ready'] is True
    assert release_gate_summary['overall_external_ready'] is False
    assert operator_handoff_summary['operational_stub_count'] == 4
    assert operator_handoff_summary['release_governance_count'] == 4
    assert operator_handoff_summary['device_packaging_count'] == 4
    assert operator_handoff_summary['benchmark_closure_count'] == 4
    assert operator_handoff_summary['training_accounting_count'] == 4
    assert operator_handoff_summary['trained_artifact_count'] == 4
    assert operator_handoff_summary['management_closure_count'] == 4
    assert operator_handoff_summary['master_summary_count'] == 4
    assert operator_handoff_summary['aggregate_truth_count'] == 4
    assert operator_handoff_summary['truth_docs_count'] == 4
    assert operator_handoff_summary['project_actionability_count'] == 4
    assert operator_handoff_summary['generated_truth_count'] == 2


def test_main_logs_fatal_exception_to_run_log(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(onefile, 'detect_desktop_dir', lambda: tmp_path)

    class DummyGuard:
        def __init__(self, logger, enabled=True):
            self.logger = logger

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def blow_up(*args: object, **kwargs: object) -> object:
        raise onefile.ConfigValidationError('boom')

    monkeypatch.setattr(onefile, 'WindowsExecutionGuard', DummyGuard)
    monkeypatch.setattr(onefile, 'run_pipeline', blow_up)

    exit_code = onefile.main(
        [
            '--mode',
            'verify',
            '--profile',
            'smoke',
            '--artifact-root',
            str(tmp_path / 'artifacts'),
            '--offline-seed-only',
        ]
    )

    assert exit_code == 1
    run_logs = list((tmp_path / 'artifacts').rglob('run_log.jsonl'))
    assert len(run_logs) == 1
    rows = [json.loads(line) for line in run_logs[0].read_text(encoding='utf-8').splitlines()]
    assert any(row['event'] == 'fatal_exception' for row in rows)
    assert any(row['event'] == 'logger_finalize' and row['payload']['status'] == 'failed' for row in rows)
    failed_artifacts = list(tmp_path.glob('MertFormer_Chess_5080_Result_FAILED_*.json'))
    assert failed_artifacts


def test_mirror_parity_report_declares_required_families() -> None:
    report = onefile.build_mirror_parity_report(make_mirror_cfg())
    assert report['audit_status'] == 'ok'
    assert 'mla' in report['required_families']
    assert 'moe_liquid_router' in report['required_families']
    assert report['embedding_strategy'] == 'onefile_only'


def test_bitlinear_matches_canonical_forward() -> None:
    torch.manual_seed(7)
    mirrored = onefile.BitLinear(16, 16, bias=True, enabled=True)
    canonical = CanonicalBitLinear(16, 16, bias=True)
    canonical.load_state_dict(mirrored.state_dict())
    x = torch.randn(2, 5, 16)
    out_mirror = mirrored(x)
    out_canonical = canonical(x)
    assert torch.allclose(out_mirror, out_canonical, atol=1e-6, rtol=1e-5)


def test_import_optional_sdk_module_returns_none_when_unavailable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(onefile, "REPO_ROOT", tmp_path)

    def fake_import_module(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(onefile.importlib, "import_module", fake_import_module)
    assert onefile._import_optional_sdk_module("mertformer_sdk.kernels.dispatcher") is None


def test_try_lowbit_kernel_uses_dynamic_sdk_loader(monkeypatch) -> None:
    class DispatcherModule:
        @staticmethod
        def select_backend(x: object, w: object) -> str:
            del x, w
            return "mps_optimized"

    monkeypatch.setattr(onefile, "_LOWBIT_KERNEL_ENABLED", True)
    monkeypatch.setattr(
        onefile,
        "_import_optional_sdk_module",
        lambda name: DispatcherModule if name == "mertformer_sdk.kernels.dispatcher" else None,
    )
    x = torch.randn(2, 3, 8)
    w = torch.randn(8, 8)
    out = onefile._try_lowbit_kernel(x, w, None)
    expected = onefile.F.linear(onefile.activation_quant(x), onefile.weight_quant(w), None)
    assert out is not None
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-5)


def test_world_model_head_matches_canonical() -> None:
    torch.manual_seed(11)
    mirrored = onefile.CausalWorldModelHead(32, horizon=2)
    canonical = CanonicalWorldModelHead(32, horizon=2)
    canonical.load_state_dict(mirrored.state_dict())
    x = torch.randn(3, 6, 32)
    mirror_out = mirrored(x).to_dict()
    canonical_out = canonical(x).to_dict()
    for key in mirror_out:
        assert torch.allclose(mirror_out[key], canonical_out[key], atol=1e-6, rtol=1e-5)


def test_lifelong_safety_matches_canonical() -> None:
    torch.manual_seed(13)
    mirrored = onefile.LifelongSafetyLayer(24, ema_decay=0.97, max_adaptation_gain=0.04, drift_threshold=0.3)
    canonical = CanonicalLifelongSafetyLayer(24, ema_decay=0.97, max_adaptation_gain=0.04, drift_threshold=0.3)
    canonical.load_state_dict(mirrored.state_dict())
    x = torch.randn(2, 4, 24)
    out_mirror = mirrored(x)
    out_canonical = canonical(x)
    assert torch.allclose(out_mirror, out_canonical, atol=1e-6, rtol=1e-5)
    assert mirrored.safety_metrics() == canonical.safety_metrics()


def test_unitary_qinn_matches_canonical_when_enabled() -> None:
    torch.manual_seed(17)
    mirrored = onefile.UnitaryQINN(16, enabled=True)
    with patched_canonical_cfg(use_qinn=True):
        canonical = CanonicalUnitaryQINN(16)
        canonical.load_state_dict(mirrored.state_dict())
        x = torch.randn(2, 3, 16)
        out_mirror = mirrored(x)
        out_canonical = canonical(x)
    assert torch.allclose(out_mirror, out_canonical, atol=1e-6, rtol=1e-5)


def test_mla_matches_canonical_forward_and_cache() -> None:
    torch.manual_seed(19)
    run_cfg = make_mirror_cfg(
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=24,
        use_moe=False,
        use_liquid=False,
        use_liquid_adapter=False,
        use_qinn=False,
        use_global_workspace_broadcast=False,
        use_neuromodulatory_gain=False,
        use_latent_ode_state_channel=False,
        use_world_model_head=False,
        use_lifelong_safety_layer=False,
        use_hebbian_plasticity=False,
        use_neuro_symbolic_layer=False,
        use_cross_expert_sync_bus=False,
        use_structural_plasticity=False,
    )
    mirrored = onefile.MLA(onefile.build_mirror_model_config(run_cfg))
    with patched_canonical_cfg(
        hidden_size=32,
        num_heads=4,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=24,
        rope_theta=100000.0,
        rope_base=100000.0,
        attention_dropout=0.0,
        use_flash_attn_inference=False,
        use_hierarchical_kv_cache=False,
    ):
        canonical = CanonicalMLA()
        canonical.load_state_dict(mirrored.state_dict())
        x = torch.randn(2, 5, 32)
        out_mirror, kv_mirror = mirrored(x, use_cache=True)
        out_canonical, kv_canonical = canonical(x, use_cache=True)
        step = torch.randn(2, 1, 32)
        step_mirror, _ = mirrored(step, past_key_value=kv_mirror, use_cache=True)
        step_canonical, _ = canonical(step, past_key_value=kv_canonical, use_cache=True)
    assert torch.allclose(out_mirror, out_canonical, atol=1e-5, rtol=1e-5)
    assert torch.allclose(step_mirror, step_canonical, atol=1e-5, rtol=1e-5)


def test_liquid_mixer_matches_canonical_eval_path() -> None:
    torch.manual_seed(23)
    mirrored = onefile.LiquidMixer(16, use_bitnet=True, fast_path=False).eval()
    canonical = CanonicalLiquidMixer(16, fast_path=False).eval()
    canonical.load_state_dict(mirrored.state_dict())
    x = torch.randn(2, 4, 16)
    out_mirror = mirrored(x)
    out_canonical = canonical(x)
    assert torch.allclose(out_mirror, out_canonical, atol=1e-5, rtol=1e-5)


def test_moe_matches_canonical_eval_path() -> None:
    torch.manual_seed(29)
    run_cfg = make_mirror_cfg(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate=32,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        num_experts=4,
        moe_top_k=2,
        use_switch_loss=True,
        router_jitter=0.02,
        router_jitter_boost=0.1,
        z_loss_coef=1e-4,
        use_moe=True,
        use_liquid=False,
        use_liquid_adapter=False,
        use_qinn=False,
        use_global_workspace_broadcast=False,
        use_neuromodulatory_gain=False,
        use_latent_ode_state_channel=False,
        use_world_model_head=False,
        use_lifelong_safety_layer=False,
        use_hebbian_plasticity=False,
        use_neuro_symbolic_layer=False,
        use_cross_expert_sync_bus=False,
        use_structural_plasticity=False,
        use_expert_paging=False,
    )
    mirrored = onefile.MoE(onefile.build_mirror_model_config(run_cfg)).eval()
    with patched_canonical_cfg(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate=32,
        num_experts=4,
        num_experts_per_tok=2,
        active_experts=2,
        router_temperature=1.0,
        router_jitter=0.02,
        router_jitter_boost=0.1,
        z_loss_coef=1e-4,
        shared_expert_gate=0.0,
        use_switch_loss=True,
        moe_capacity_enforce=True,
        moe_capacity_factor=1.25,
        moe_dispatch_mode='sequential',
        use_cross_expert_sync_bus=False,
        cross_expert_sync_gain=0.05,
        use_structural_plasticity=False,
        use_expert_paging=False,
        expert_paging_inference_only=True,
        expert_paging_lazy_init=True,
        expert_paging_cache_size=2,
        expert_paging_offload_device='cpu',
        expert_paging_verbose=False,
    ):
        canonical = CanonicalMoE().eval()
        canonical.load_state_dict(mirrored.state_dict())
        x = torch.randn(2, 3, 16)
        out_mirror, aux_mirror = mirrored(x)
        out_canonical, aux_canonical = canonical(x)
    assert torch.allclose(out_mirror, out_canonical, atol=1e-5, rtol=1e-5)
    assert torch.allclose(aux_mirror, aux_canonical, atol=1e-6, rtol=1e-5)
