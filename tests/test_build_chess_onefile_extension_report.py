from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    script_path = ROOT / 'scripts' / 'build_chess_onefile_extension_report.py'
    spec = importlib.util.spec_from_file_location('build_chess_onefile_extension_report_module', script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_payload_is_ready_with_required_suites() -> None:
    module = _load_module()
    payload = module.build_payload()
    assert payload['final_status'] == 'READY'
    assert payload['summary']['all_green'] is True
    assert payload['summary']['missing_suites'] == []
    assert payload['summary']['curated_position_count'] > 0
    assert payload['summary']['curated_training_examples'] > 0
    assert payload['summary']['synthetic_teaching_records'] > 0


def test_build_markdown_mentions_core_counts() -> None:
    module = _load_module()
    payload = module.build_payload()
    text = module.build_markdown(payload)
    assert 'Chess Onefile Extension Report' in text
    assert 'curated_position_count' in text
    assert 'synthetic_teaching_records' in text
